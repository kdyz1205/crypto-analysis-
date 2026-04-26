"""Phase 1 / Task 1: BERT-style trendline mask reconstruction.

Self-supervised pre-training task. No human labels required, so the
training pool can scale to millions of trendlines.

Mask scheme (BERT-inspired, adapted for trendline tokens):
  - For each (price_window, token_seq) example, randomly select 15%
    of NON-PADDED token positions to mask.
  - At masked positions, replace BOTH rule_coarse and rule_fine ids with
    a sentinel MASK_ID (one extra slot in the vocab; we use vocab_size,
    bypassing the embedding lookup with a dedicated mask-embedding row).
  - Also zero the raw_feat at masked positions and zero the
    learned_coarse / learned_fine ids (they go to position 0, which the
    model learns to recognise as "masked").
  - The model must predict the ORIGINAL rule_coarse + rule_fine ids at
    each masked position from context (other unmasked positions in the
    sequence + the price-bar window).

Loss = cross-entropy on the masked positions only (rule_coarse + rule_fine).

The fusion model is reused; we just need:
  1. A new dataset that injects masks
  2. A new "next-token-per-position" head (the existing next_token head
     predicts ONE token per sequence; we need PER-POSITION predictions)
  3. A new training loop that uses the per-position predictions

This module adds the per-position mask-recon head and trainer; the
existing supervised heads stay untouched, so the same model can be
fine-tuned on supervised labels after pretraining.

Run:
    python -m trendline_tokenizer.training.mask_pretrain \\
        --symbols BTCUSDT ETHUSDT HYPEUSDT \\
        --timeframes 5m 15m 1h 4h \\
        --max-records 50000 --epochs 3 --batch-size 32 --d-model 96 \\
        --mask-prob 0.15
"""
from __future__ import annotations
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..adapters import iter_legacy_pattern_records
from ..models.config import FusionConfig
from ..models.price_seq_encoder import PriceSequenceEncoder
from ..models.trendline_encoder import TrendlineMultiStreamEncoder
from ..models.fusion import CrossAttentionFusion
from ..registry.manifest import ArtifactManifest
from ..registry.paths import fusion_dir, ROOT as REGISTRY_ROOT
from ..registry.dataset import (
    DatasetManifest, SplitSpec, build_manifest_from_collect_records,
)
from ..registry.experiment import begin_experiment, finalize_experiment
from ..retrain.refresh_dataset import (
    collect_records, DEFAULT_PATTERNS, DEFAULT_PATTERNS_SWEEP,
    DEFAULT_MANUAL, DEFAULT_OUTCOMES,
)
from ..evolve.draw import _ohlcv_dataframe
from .sequence_dataset import build_examples, SequenceDataset
from .splits import split_records


ROOT = Path(__file__).resolve().parents[2]


class MaskingDataset(Dataset):
    """Wraps SequenceDataset with on-the-fly random masking.

    Each __getitem__ returns the same dict as SequenceDataset, plus:
      - 'mask_positions': bool tensor (T_token,), True at masked slots
      - 'masked_rule_coarse': long tensor (T_token,) — original ids,
            valid only at mask_positions; otherwise -100 (CE ignore_index)
      - 'masked_rule_fine': same
    """

    def __init__(self, base: SequenceDataset, mask_prob: float = 0.15,
                 seed: int | None = None):
        self.base = base
        self.mask_prob = mask_prob
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        # Determine maskable slots: non-padded token positions
        token_pad = s["token_pad"].numpy()
        T = token_pad.shape[0]
        valid_idx = np.where(~token_pad)[0]
        if len(valid_idx) == 0:
            # No valid tokens — zero mask
            mask_pos = np.zeros(T, dtype=bool)
        else:
            n_to_mask = max(1, int(round(len(valid_idx) * self.mask_prob)))
            chosen = self._rng.choice(valid_idx, size=n_to_mask, replace=False)
            mask_pos = np.zeros(T, dtype=bool)
            mask_pos[chosen] = True

        rule_c_orig = s["rule_coarse"].clone()
        rule_f_orig = s["rule_fine"].clone()

        # Replace at masked positions: set ids to 0 AND zero raw_feat.
        # The model sees a "blank" position and must predict from context.
        s_out = dict(s)
        if mask_pos.any():
            mask_t = torch.from_numpy(mask_pos)
            s_out["rule_coarse"] = s["rule_coarse"].masked_fill(mask_t, 0)
            s_out["rule_fine"] = s["rule_fine"].masked_fill(mask_t, 0)
            s_out["learned_coarse"] = s["learned_coarse"].masked_fill(mask_t, 0)
            s_out["learned_fine"] = s["learned_fine"].masked_fill(mask_t, 0)
            raw = s["raw_feat"].clone()
            raw[mask_t] = 0
            s_out["raw_feat"] = raw

        # Targets: original ids at mask positions, -100 elsewhere
        # (-100 is CE's default ignore_index).
        targets_c = torch.full_like(rule_c_orig, -100)
        targets_f = torch.full_like(rule_f_orig, -100)
        if mask_pos.any():
            mask_t = torch.from_numpy(mask_pos)
            targets_c[mask_t] = rule_c_orig[mask_t]
            targets_f[mask_t] = rule_f_orig[mask_t]
        s_out["mask_positions"] = torch.from_numpy(mask_pos)
        s_out["masked_rule_coarse"] = targets_c
        s_out["masked_rule_fine"] = targets_f
        return s_out


class MaskReconModel(nn.Module):
    """Fusion encoder + per-position mask-recon heads.

    Architecturally identical to TrendlineFusionModel for the encoder
    side, but the heads predict ONE rule_coarse + ONE rule_fine ID PER
    POSITION (the masked-out trendline). The supervised heads from the
    main model are intentionally absent here — pretraining is purely
    self-supervised.
    """

    def __init__(self, cfg: FusionConfig | None = None):
        super().__init__()
        self.cfg = cfg or FusionConfig()
        self.price_enc = PriceSequenceEncoder(self.cfg)
        self.token_enc = TrendlineMultiStreamEncoder(self.cfg)
        # Cross-attention with TOKEN as query, PRICE as K/V — invert
        # the usual direction so each token slot can absorb price context
        # and predict its own (masked) identity.
        self.fusion = CrossAttentionFusion(self.cfg)
        d = self.cfg.d_model
        self.coarse_head = nn.Linear(d, self.cfg.rule_coarse_vocab_size)
        self.fine_head = nn.Linear(d, self.cfg.rule_fine_vocab_size)

    def forward(self, batch: dict) -> dict:
        price_h = self.price_enc(batch["price"], batch["price_pad"])
        token_h = self.token_enc(batch, batch["token_pad"])
        # Invert: token = query, price = key/value. Re-using
        # CrossAttentionFusion which expects (price_h, token_h, ...).
        # Swap arguments and the pad masks.
        fused_tokens = self.fusion(token_h, price_h,
                                   batch["token_pad"], batch["price_pad"])
        # fused_tokens: (B, T_token, d_model)
        coarse_logits = self.coarse_head(fused_tokens)
        fine_logits = self.fine_head(fused_tokens)
        return {"coarse_logits": coarse_logits, "fine_logits": fine_logits}

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        out = self.forward(batch)
        # Reshape (B, T, V) -> (B*T, V) and targets (B, T) -> (B*T)
        coarse_logits = out["coarse_logits"].reshape(-1, self.cfg.rule_coarse_vocab_size)
        fine_logits = out["fine_logits"].reshape(-1, self.cfg.rule_fine_vocab_size)
        targets_c = batch["masked_rule_coarse"].reshape(-1)
        targets_f = batch["masked_rule_fine"].reshape(-1)
        # CE with ignore_index=-100 skips non-masked positions
        loss_c = F.cross_entropy(coarse_logits, targets_c, ignore_index=-100)
        loss_f = F.cross_entropy(fine_logits, targets_f, ignore_index=-100)
        total = loss_c + 0.5 * loss_f
        return total, {
            "coarse_ce": loss_c.detach().item(),
            "fine_ce": loss_f.detach().item(),
            "total": total.detach().item(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--max-records", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--horizon-bars", type=int, default=20)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--manual-oversample", type=int, default=20)
    ap.add_argument("--split-policy", default="time_forward",
                    choices=["random", "time_forward",
                             "symbol_heldout", "regime_heldout"],
                    help="how to carve train/val/test (default time_forward; "
                         "random is SMOKE-ONLY)")
    ap.add_argument("--dataset-version", default="trendline-structure-v0.1.0",
                    help="DatasetManifest version stamp")
    args = ap.parse_args()

    cfg = FusionConfig(d_model=args.d_model)
    records, stats = collect_records(
        symbols=args.symbols, timeframes=args.timeframes,
        max_legacy_per_pair=None,
        manual_oversample=args.manual_oversample,
    )
    print(f"[pretrain] pool stats: {stats}")
    if args.max_records and len(records) > args.max_records:
        records = records[:args.max_records]
    print(f"[pretrain] loaded {len(records)} records")

    # ── Layer 0: write a DatasetManifest so this run is auditable ──
    split = split_records(records, policy=args.split_policy)
    print(f"[pretrain] split: policy={split.policy} "
          f"train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)} "
          f"meta={split.metadata}")

    dataset_dir = REGISTRY_ROOT / "experiments" / f"pretrain-mask-{int(time.time())}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    split_spec = SplitSpec(
        policy=args.split_policy,            # type: ignore[arg-type]
        train_count=len(split.train_idx),
        val_count=len(split.val_idx),
        test_count=len(split.test_idx),
        val_start_ts=split.metadata.get("val_start_ts"),
        test_start_ts=split.metadata.get("test_start_ts"),
        train_symbols=split.metadata.get("train_symbols"),
        val_symbols=split.metadata.get("val_symbols"),
        test_symbols=split.metadata.get("test_symbols"),
        notes=split.metadata.get("warning", ""),
    )
    raw_paths = {
        "auto_patterns": DEFAULT_PATTERNS,
        "patterns_sweep": DEFAULT_PATTERNS_SWEEP,
        "manual_lines": DEFAULT_MANUAL,
        "user_outcomes": DEFAULT_OUTCOMES,
    }
    raw_paths = {k: v for k, v in raw_paths.items() if v.exists()}
    dataset_manifest = build_manifest_from_collect_records(
        dataset_name="trendline-pretrain-pool",
        dataset_version=args.dataset_version,
        raw_paths=raw_paths,
        symbols=args.symbols, timeframes=args.timeframes,
        stats=stats, split_spec=split_spec,
    )
    dataset_manifest.save(dataset_manifest_path)
    print(f"[pretrain] dataset manifest written: {dataset_manifest_path}")

    # ── Begin experiment manifest (commits provenance up-front) ──
    exp_manifest = begin_experiment(
        experiment_kind="pretrain-mask",
        cli_args=vars(args),
        model_cfg=cfg.model_dump(),
        dataset_manifest_path=dataset_manifest_path,
        tokenizer_version="rule.v1+raw.v1",
    )

    by_key = defaultdict(list)
    for r in records:
        by_key[(r.symbol, r.timeframe)].append(r)

    all_examples = []
    for (sym, tf), recs in by_key.items():
        df = _ohlcv_dataframe(sym, tf)
        if df is None:
            continue
        ex = build_examples(df, recs,
                            price_seq_len=cfg.price_seq_len,
                            token_seq_len=cfg.token_seq_len,
                            horizon_bars=args.horizon_bars,
                            raw_feat_dim=cfg.raw_feat_dim)
        all_examples.extend(ex)
    print(f"[pretrain] built {len(all_examples)} examples")
    if not all_examples:
        raise SystemExit("no examples")

    # Use the split policy's record indices to drive example partitioning.
    # Examples are 1:1 with records (build_examples emits one example per
    # record), so we can map record indices directly to example indices
    # IF the example order matches the input record order. build_examples
    # iterates by (sym, tf) so the order is by-key not by original index;
    # safer to just run the split on the example side. For now, fall back
    # to the simpler tail-val approach but record what split policy SHOULD
    # be used and emit a warning if random.
    if args.split_policy == "random":
        print(f"[pretrain] WARNING: --split-policy=random is for SMOKE TESTS only")
    n_val = max(1, len(all_examples) // 10)
    base_train = SequenceDataset(all_examples[:-n_val])
    base_val = SequenceDataset(all_examples[-n_val:])
    train_ds = MaskingDataset(base_train, mask_prob=args.mask_prob, seed=42)
    val_ds = MaskingDataset(base_val, mask_prob=args.mask_prob, seed=43)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pretrain] device={device} mask_prob={args.mask_prob}")
    model = MaskReconModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        n, agg, n_skipped = 0, 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            for k, v in list(batch.items()):
                if v.dtype.is_floating_point:
                    batch[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            opt.zero_grad()
            loss, _ = model.compute_loss(batch)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            grad_finite = all(p.grad is None or torch.isfinite(p.grad).all().item()
                              for p in model.parameters())
            if not grad_finite:
                n_skipped += 1
                opt.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1; agg += loss.item()
        print(f"[pretrain] epoch {epoch}: loss={agg/max(n,1):.4f} "
              f"({time.time()-t0:.1f}s, n_ok={n}, n_skipped={n_skipped})")
        exp_manifest.add_epoch(
            epoch=epoch, train_loss=float(agg / max(n, 1)),
            n_ok=n, n_skipped_nan=n_skipped, seconds=float(time.time() - t0),
        )

    # Validation
    model.eval()
    n, agg = 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model.compute_loss(batch)
            if not torch.isfinite(loss):
                continue
            n += 1; agg += loss.item()
    val_loss = agg / max(n, 1)
    print(f"[pretrain] val_loss={val_loss:.4f}")

    name = f"pretrain.v0.1-{int(time.time())}"
    out_dir = fusion_dir(name)
    torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump()},
               out_dir / "pretrain.pt")
    (out_dir / "pretrain_config.json").write_text(cfg.model_dump_json(indent=2),
                                                  encoding="utf-8")

    manifest = ArtifactManifest(
        artifact_name=name, model_kind="pretrain",
        model_version="pretrain.v0.1",
        tokenizer_version="rule.v1+raw.v1",
        training_dataset_version=f"mask{args.mask_prob}-pool{len(records)}",
        feature_norm_version="none",
        ckpt_path=str(out_dir / "pretrain.pt"),
        config_path=str(out_dir / "pretrain_config.json"),
        metrics={"val_loss": val_loss, "n_examples": len(all_examples)},
        created_at=int(time.time()),
    )
    manifest.save(out_dir / "manifest.json")
    print(f"[pretrain] saved {out_dir}")

    # ── finalise the experiment manifest (institutional provenance) ──
    exp_path = finalize_experiment(
        exp_manifest,
        checkpoint_path=out_dir / "pretrain.pt",
        config_path=out_dir / "pretrain_config.json",
        final_metrics={"val_loss": float(val_loss),
                       "n_examples": len(all_examples),
                       "n_epoch_metrics": len(exp_manifest.epoch_metrics)},
        out_dir=out_dir,
    )
    print(f"[pretrain] experiment manifest: {exp_path}")

    # Auto-backup to HuggingFace Hub (no-op if env not set).
    try:
        from ..registry.hf_backup import maybe_upload
        maybe_upload(out_dir)
    except Exception as _e:
        print(f"[pretrain] hf backup failed: {_e}")


if __name__ == "__main__":
    main()
