"""Run a 3-variant ablation: train rule-only / raw-only / all-three on
the same data, backtest each on the same slice, dump an AblationReport.

Usage:
    python -m trendline_tokenizer.backtest.run_ablation \\
        --symbols BTCUSDT ETHUSDT HYPEUSDT \\
        --timeframes 5m 15m 1h \\
        --max-records 3000 --epochs 2 \\
        --backtest-symbol BTCUSDT --backtest-timeframe 5m \\
        --start-bar 1000 --predict-every 100
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..adapters import iter_legacy_pattern_records
from ..evolve.draw import _ohlcv_dataframe
from ..learned.vqvae import HierarchicalVQVAE, VQVAEConfig
from ..models.config import FusionConfig
from ..models.full_model import TrendlineFusionModel
from ..registry.manifest import ArtifactManifest
from ..registry.paths import fusion_dir
from ..retrain.refresh_dataset import collect_records
from ..training.sequence_dataset import build_examples, SequenceDataset
from ..training.train_fusion import _load_vqvae
from ..inference.inference_service import InferenceService
from ..inference.signal_engine import SignalEngine, SignalEngineConfig
from .ablation import VARIANTS, AblationReport
from .metrics import compute_metrics, summarize_metrics
from .replay_engine import replay
from .strategy_simulator import simulate


ROOT = Path(__file__).resolve().parents[2]


def _train_variant(variant_name: str, stream_cfg: dict, *,
                   symbols, timeframes, max_records, epochs, batch_size,
                   d_model, manual_oversample, device) -> Path:
    cfg = FusionConfig(
        d_model=d_model, n_layers_price=2, n_layers_token=2, n_layers_fusion=2,
        **stream_cfg,
    )
    vqvae = None
    if cfg.use_learned_tokens:
        vqvae = _load_vqvae(Path(cfg.vqvae_checkpoint_path), fusion_cfg=cfg)
    records, stats = collect_records(
        symbols=symbols, timeframes=timeframes,
        manual_oversample=manual_oversample,
    )
    if max_records and len(records) > max_records:
        n_manual = stats["manual_total_in_pool"]
        keep_auto = max(0, max_records - n_manual)
        records = records[:n_manual + keep_auto]

    by_key = defaultdict(list)
    for r in records:
        by_key[(r.symbol, r.timeframe)].append(r)
    all_examples = []
    for (sym, tf), recs in by_key.items():
        df = _ohlcv_dataframe(sym, tf)
        if df is None:
            continue
        ex = build_examples(df, recs, price_seq_len=cfg.price_seq_len,
                            token_seq_len=cfg.token_seq_len, horizon_bars=20,
                            raw_feat_dim=cfg.raw_feat_dim, vqvae=vqvae)
        all_examples.extend(ex)
    print(f"[{variant_name}] {len(all_examples)} examples, "
          f"streams: rule={cfg.use_rule_tokens} learned={cfg.use_learned_tokens} raw={cfg.use_raw_features}")

    n_val = max(1, len(all_examples) // 10)
    train_ds = SequenceDataset(all_examples[:-n_val])
    val_ds = SequenceDataset(all_examples[-n_val:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TrendlineFusionModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    target_keys = ("next_coarse", "next_fine", "bounce", "brk", "cont", "buffer_pct")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        n, agg = 0, 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {k: batch.pop(k) for k in target_keys}
            opt.zero_grad()
            loss, _ = model.compute_loss(batch, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1; agg += loss.item()
        print(f"[{variant_name}] epoch {epoch}: train_loss={agg/max(n,1):.4f} ({time.time()-t0:.1f}s)")

    model.eval()
    n, agg = 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {k: batch.pop(k) for k in target_keys}
            loss, _ = model.compute_loss(batch, targets)
            n += 1; agg += loss.item()
    val_loss = agg / max(n, 1)
    print(f"[{variant_name}] val_loss={val_loss:.4f}")

    name = f"abl-{variant_name}-{int(time.time())}"
    out_dir = fusion_dir(name)
    torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump()},
               out_dir / "fusion.pt")
    (out_dir / "fusion.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    tok = []
    if cfg.use_rule_tokens: tok.append("rule.v1")
    if cfg.use_learned_tokens and vqvae is not None: tok.append("vqvae.v0")
    if cfg.use_raw_features: tok.append("raw.v1")
    manifest = ArtifactManifest(
        artifact_name=name, model_kind="fusion", model_version=cfg.version,
        tokenizer_version="+".join(tok) or "none",
        training_dataset_version=f"abl-{max_records}",
        feature_norm_version="none",
        ckpt_path=str(out_dir / "fusion.pt"),
        config_path=str(out_dir / "fusion.json"),
        metrics={"val_loss": val_loss, "n_train": len(train_ds), "n_val": len(val_ds),
                 "streams": stream_cfg},
        created_at=int(time.time()),
    )
    manifest.save(out_dir / "manifest.json")
    return out_dir / "manifest.json"


def _backtest_variant(manifest_path: Path, *, symbol, timeframe, start_bar,
                      predict_every, hold_bars, min_confidence, device):
    df = _ohlcv_dataframe(symbol, timeframe)
    if df is None or len(df) < start_bar + 10:
        raise SystemExit(f"insufficient OHLCV for {symbol} {timeframe}")
    svc = InferenceService(manifest_path, device=device)
    se = SignalEngine()
    steps = list(replay(df, symbol=symbol, timeframe=timeframe,
                        service=svc, signal_engine=se,
                        predict_every=predict_every, start_bar=start_bar))
    trades = simulate(steps, hold_bars=hold_bars, min_confidence=min_confidence)
    return compute_metrics(trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--max-records", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--manual-oversample", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--variants", nargs="+",
                    default=["rule_only", "raw_only", "all"])
    ap.add_argument("--backtest-symbol", required=True)
    ap.add_argument("--backtest-timeframe", required=True)
    ap.add_argument("--start-bar", type=int, default=1000)
    ap.add_argument("--predict-every", type=int, default=100)
    ap.add_argument("--hold-bars", type=int, default=20)
    ap.add_argument("--min-confidence", type=float, default=0.55)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] device={device}  variants={args.variants}")

    report = AblationReport()
    for v_name in args.variants:
        if v_name not in VARIANTS:
            print(f"[ablation] unknown variant {v_name!r}; skipping")
            continue
        manifest_path = _train_variant(
            v_name, VARIANTS[v_name],
            symbols=args.symbols, timeframes=args.timeframes,
            max_records=args.max_records, epochs=args.epochs,
            batch_size=args.batch_size, d_model=args.d_model,
            manual_oversample=args.manual_oversample, device=device,
        )
        metrics = _backtest_variant(
            manifest_path, symbol=args.backtest_symbol,
            timeframe=args.backtest_timeframe, start_bar=args.start_bar,
            predict_every=args.predict_every, hold_bars=args.hold_bars,
            min_confidence=args.min_confidence, device=device,
        )
        report.add(v_name, manifest_path.parent.name, metrics)
        print(f"[ablation] {v_name} -> {summarize_metrics(metrics)}")

    print()
    print(report.render())

    out = ROOT / "data" / "backtest_runs" / f"ablation_{int(time.time())}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.render() + "\n", encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
