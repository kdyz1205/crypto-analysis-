"""End-to-end training for the multi-stream fusion model.

Smoke-run usage:
    python -m trendline_tokenizer.training.train_fusion \\
        --symbols BTCUSDT --timeframes 5m \\
        --max-records 200 --epochs 1 --batch-size 8

Production usage: drop --max-records, raise epochs.
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..adapters import iter_legacy_pattern_records
from ..models.config import FusionConfig
from ..models.full_model import TrendlineFusionModel
from ..learned.vqvae import HierarchicalVQVAE, VQVAEConfig
from ..registry.manifest import ArtifactManifest
from ..registry.paths import fusion_dir
from ..evolve.draw import _ohlcv_dataframe
from ..retrain.refresh_dataset import collect_records
from .sequence_dataset import build_examples, SequenceDataset


ROOT = Path(__file__).resolve().parents[2]


def _load_vqvae(ckpt_path: Path | None, fusion_cfg: FusionConfig | None = None):
    """Load a HierarchicalVQVAE checkpoint. If a FusionConfig is given,
    validate the checkpoint's code counts match its learned_* vocabs;
    otherwise inference would silently misalign embedding tables."""
    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"[train] no vqvae checkpoint at {ckpt_path}; learned tokens will be zeros")
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = VQVAEConfig(**state["cfg"])
    if fusion_cfg is not None:
        if cfg.coarse_codes != fusion_cfg.learned_coarse_vocab_size:
            raise ValueError(
                f"VQ-VAE checkpoint coarse_codes={cfg.coarse_codes} "
                f"!= FusionConfig.learned_coarse_vocab_size={fusion_cfg.learned_coarse_vocab_size}. "
                f"Either retrain the VQ-VAE or update FusionConfig."
            )
        if cfg.fine_codes != fusion_cfg.learned_fine_vocab_size:
            raise ValueError(
                f"VQ-VAE checkpoint fine_codes={cfg.fine_codes} "
                f"!= FusionConfig.learned_fine_vocab_size={fusion_cfg.learned_fine_vocab_size}."
            )
        if cfg.feat_dim != fusion_cfg.raw_feat_dim:
            raise ValueError(
                f"VQ-VAE feat_dim={cfg.feat_dim} != raw_feat_dim={fusion_cfg.raw_feat_dim}"
            )
    model = HierarchicalVQVAE(cfg)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"[train] loaded VQ-VAE from {ckpt_path} "
          f"(coarse_codes={cfg.coarse_codes}, fine_codes={cfg.fine_codes})")
    return model


def _load_records(symbols, timeframes, max_records, *,
                  use_manual: bool = True,
                  manual_oversample: int = 50):
    """Load training records from EVERY source.

    Returns the merged pool (manual oversampled + auto). max_records
    caps the AUTO portion only - the manual gold is always fully
    included so style alignment doesn't get throttled by the cap.
    """
    records, stats = collect_records(
        symbols=symbols, timeframes=timeframes,
        max_legacy_per_pair=None,    # honour max_records globally below instead
        manual_oversample=manual_oversample if use_manual else 0,
    )
    print(f"[train] pool stats: {stats}")
    if max_records and len(records) > max_records:
        # Keep all manual (front of list), then truncate auto
        n_manual = stats["manual_total_in_pool"]
        keep_auto = max(0, max_records - n_manual)
        records = records[:n_manual + keep_auto]
        print(f"[train] truncated to {len(records)} (manual={n_manual}, auto={keep_auto})")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--max-records", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon-bars", type=int, default=20)
    ap.add_argument("--d-model", type=int, default=64,
                    help="smoke runs use 64 to fit on CPU; bump for prod")
    ap.add_argument("--n-layers-price", type=int, default=2)
    ap.add_argument("--n-layers-token", type=int, default=2)
    ap.add_argument("--n-layers-fusion", type=int, default=2)
    ap.add_argument("--no-manual", action="store_true",
                    help="skip manual_trendlines.json (auto-only training)")
    ap.add_argument("--manual-oversample", type=int, default=50,
                    help="duplicate each manual record N times in the training pool")
    args = ap.parse_args()

    cfg = FusionConfig(
        d_model=args.d_model,
        n_layers_price=args.n_layers_price,
        n_layers_token=args.n_layers_token,
        n_layers_fusion=args.n_layers_fusion,
    )
    vqvae = _load_vqvae(Path(cfg.vqvae_checkpoint_path) if cfg.vqvae_checkpoint_path else None,
                        fusion_cfg=cfg)

    records = _load_records(args.symbols, args.timeframes, args.max_records,
                            use_manual=(not args.no_manual),
                            manual_oversample=args.manual_oversample)
    print(f"[train] loaded {len(records)} records")
    if not records:
        raise SystemExit("no records - populate data/patterns/*.jsonl first")

    by_key = defaultdict(list)
    for r in records:
        by_key[(r.symbol, r.timeframe)].append(r)

    all_examples = []
    for (sym, tf), recs in by_key.items():
        df = _ohlcv_dataframe(sym, tf)
        if df is None:
            print(f"[train] skip {sym} {tf}: no OHLCV cache")
            continue
        ex = build_examples(df, recs,
                            price_seq_len=cfg.price_seq_len,
                            token_seq_len=cfg.token_seq_len,
                            horizon_bars=args.horizon_bars,
                            raw_feat_dim=cfg.raw_feat_dim,
                            vqvae=vqvae)
        all_examples.extend(ex)
        print(f"[train] {sym} {tf}: {len(ex)} examples")
    print(f"[train] total {len(all_examples)} examples")
    if not all_examples:
        raise SystemExit("no examples - check OHLCV availability for the records")

    n_val = max(1, len(all_examples) // 10)
    train_ds = SequenceDataset(all_examples[:-n_val])
    val_ds = SequenceDataset(all_examples[-n_val:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")
    model = TrendlineFusionModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    target_keys = ("next_coarse", "next_fine", "bounce", "brk", "cont", "buffer_pct")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        n, agg = 0, 0.0
        n_skipped_nan = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {k: batch.pop(k) for k in target_keys}
            # Defensively scrub NaN/inf from float inputs/targets
            for k, v in list(batch.items()):
                if v.dtype.is_floating_point:
                    batch[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            for k, v in list(targets.items()):
                if v.dtype.is_floating_point:
                    targets[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            opt.zero_grad()
            loss, _ = model.compute_loss(batch, targets)
            if not torch.isfinite(loss):
                n_skipped_nan += 1
                continue
            loss.backward()
            # Skip step if any gradient is NaN — model would diverge otherwise
            grad_finite = all(
                p.grad is None or torch.isfinite(p.grad).all().item()
                for p in model.parameters()
            )
            if not grad_finite:
                n_skipped_nan += 1
                opt.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1; agg += loss.item()
        print(f"[train] epoch {epoch}: train_loss={agg/max(n,1):.4f} "
              f"({time.time()-t0:.1f}s, n_ok={n}, n_skipped_nan={n_skipped_nan})")

    model.eval()
    n, agg = 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {k: batch.pop(k) for k in target_keys}
            loss, _ = model.compute_loss(batch, targets)
            n += 1; agg += loss.item()
    val_loss = agg / max(n, 1)
    print(f"[train] val_loss={val_loss:.4f}")

    name = f"{cfg.version}-{int(time.time())}"
    out_dir = fusion_dir(name)
    torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump()},
               out_dir / "fusion.pt")
    (out_dir / "fusion.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    tokenizer_versions = []
    if cfg.use_rule_tokens:
        tokenizer_versions.append("rule.v1")
    if cfg.use_learned_tokens and vqvae is not None:
        tokenizer_versions.append("vqvae.v0")
    if cfg.use_raw_features:
        tokenizer_versions.append("raw.v1")
    manifest = ArtifactManifest(
        artifact_name=name, model_kind="fusion", model_version=cfg.version,
        tokenizer_version="+".join(tokenizer_versions) or "none",
        training_dataset_version=f"legacy-patterns-{args.max_records}",
        feature_norm_version="none",
        ckpt_path=str(out_dir / "fusion.pt"),
        config_path=str(out_dir / "fusion.json"),
        metrics={"val_loss": val_loss, "n_train": len(train_ds),
                 "n_val": len(val_ds),
                 "streams": {"rule": cfg.use_rule_tokens,
                             "learned": cfg.use_learned_tokens and vqvae is not None,
                             "raw": cfg.use_raw_features}},
        created_at=int(time.time()),
    )
    manifest.save(out_dir / "manifest.json")
    print(f"[train] saved {out_dir}")


if __name__ == "__main__":
    main()
