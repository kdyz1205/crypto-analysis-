"""End-to-end backtest CLI.

Usage:
    python -m trendline_tokenizer.backtest.run_backtest \\
        --manifest checkpoints/fusion/<artifact>/manifest.json \\
        --symbol BTCUSDT --timeframe 5m \\
        --start-bar 256 --predict-every 8 \\
        --hold-bars 20 --min-confidence 0.55 \\
        --out-dir data/backtest_runs/

Outputs:
    <out_dir>/<run_id>/trades.jsonl
    <out_dir>/<run_id>/metrics.json
    <out_dir>/<run_id>/summary.txt
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from ..evolve.draw import _ohlcv_dataframe
from ..inference.inference_service import InferenceService
from ..inference.signal_engine import SignalEngine, SignalEngineConfig
from .replay_engine import replay
from .strategy_simulator import simulate
from .metrics import compute_metrics, summarize_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--start-bar", type=int, default=256)
    ap.add_argument("--predict-every", type=int, default=8,
                    help="run model every N bars (1 = every bar; slow)")
    ap.add_argument("--hold-bars", type=int, default=20)
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--bounce-threshold", type=float, default=0.55)
    ap.add_argument("--break-threshold", type=float, default=0.55)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtest_runs"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    df = _ohlcv_dataframe(args.symbol, args.timeframe)
    if df is None or len(df) < args.start_bar + 10:
        raise SystemExit(f"insufficient OHLCV for {args.symbol} {args.timeframe}")

    svc = InferenceService(args.manifest, device=args.device)
    se_cfg = SignalEngineConfig(
        bounce_threshold=args.bounce_threshold,
        break_threshold=args.break_threshold,
    )
    se = SignalEngine(se_cfg)
    print(f"[backtest] manifest={args.manifest.name} bars={len(df)} "
          f"start_bar={args.start_bar} predict_every={args.predict_every}")

    t0 = time.time()
    steps = list(replay(df, symbol=args.symbol, timeframe=args.timeframe,
                        service=svc, signal_engine=se,
                        predict_every=args.predict_every,
                        start_bar=args.start_bar))
    n_signals = sum(1 for s in steps if s.signal is not None and s.signal.action != "WAIT")
    print(f"[backtest] replayed {len(steps)} bars in {time.time()-t0:.1f}s; "
          f"{n_signals} non-WAIT signals")

    trades = simulate(steps, hold_bars=args.hold_bars, min_confidence=args.min_confidence)
    metrics = compute_metrics(trades)
    print(f"[backtest] {summarize_metrics(metrics)}")

    run_id = f"{args.symbol}_{args.timeframe}_{int(time.time())}"
    out = args.out_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trades.jsonl").open("w", encoding="utf-8") as fh:
        for t in trades:
            fh.write(json.dumps(asdict(t)) + "\n")
    (out / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2),
                                      encoding="utf-8")
    (out / "summary.txt").write_text(
        f"manifest: {args.manifest}\n"
        f"symbol: {args.symbol} {args.timeframe}\n"
        f"bars: {len(df)} (start={args.start_bar}, predict_every={args.predict_every})\n"
        f"signals: {n_signals}\n"
        f"{summarize_metrics(metrics)}\n",
        encoding="utf-8",
    )
    print(f"[backtest] saved {out}")


if __name__ == "__main__":
    main()
