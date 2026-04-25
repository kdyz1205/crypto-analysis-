"""Pattern-database scale-up via pivot_window + max_anchor_distance sweep.

Reuses tools.pattern_engine.scan_historical_patterns (the proven generator
that produced the 271k legacy patterns). Each sweep config produces a
separate jsonl under data/patterns_sweep/, so re-running is idempotent.

For each (symbol, timeframe) pair found in data/patterns/*.jsonl, we
also run:
  pivot_window in [2, 3, 5, 7]
  max_anchor_distance in [50, 100, 200]

So 4 x 3 = 12 configs per pair x ~5k records per config = ~60k per pair.
Across 11 sym x 5 tf = 55 pairs => ~3.3M new patterns.

Combined with the existing 271k that's ~3.6M total — a 13x scale-up.

To go further to 50M: add more symbols (need Binance Vision data ingest)
or push max_anchor_distance higher. See trendline_tokenizer/ROADMAP.md.

Usage:
    python -m scripts.scale_patterns_sweep \\
        --symbols BTCUSDT ETHUSDT SOLUSDT HYPEUSDT \\
        --timeframes 5m 15m 1h \\
        --pivot-windows 2 3 5 \\
        --max-anchor-distances 50 100 \\
        --jobs 4

Stop / resume: each config writes to a unique file. Killing & restarting
skips files that already exist + are non-empty.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pattern_engine import scan_historical_patterns, save_patterns  # noqa: E402
from trendline_tokenizer.evolve.draw import _ohlcv_dataframe              # noqa: E402

OUT_DIR = ROOT / "data" / "patterns_sweep"


def _config_tag(pw: int, mad: int) -> str:
    return f"pw{pw}_mad{mad}"


def _process_one(args: tuple) -> tuple[str, int]:
    symbol, tf, pw, mad, lookahead, max_bars, out_dir = args
    tag = _config_tag(pw, mad)
    out_path = Path(out_dir) / f"{symbol}_{tf}__{tag}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        return (str(out_path), -1)
    df = _ohlcv_dataframe(symbol, tf)
    if df is None or len(df) < pw * 3:
        return (str(out_path), 0)
    if max_bars and len(df) > max_bars:
        df = df.iloc[-max_bars:].reset_index(drop=True)
    pdf = df.rename(columns={"open_time": "timestamp"}) if "open_time" in df.columns else df
    for col in ("open", "high", "low", "close", "volume"):
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    try:
        records = scan_historical_patterns(
            pdf, symbol, tf,
            pivot_window=pw,
            max_anchor_distance=mad,
            lookahead_bars=lookahead,
        )
    except Exception as exc:
        print(f"[scale] {symbol} {tf} {tag}: error {exc}")
        return (str(out_path), 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), default=str) + "\n")
    return (str(out_path), len(records))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--pivot-windows", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--max-anchor-distances", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--lookahead-bars", type=int, default=50)
    ap.add_argument("--max-bars-per-tf", type=int, default=20000,
                    help="cap per-timeframe history (sr_patterns.scan is O(N^2))")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for s in args.symbols:
        for tf in args.timeframes:
            for pw in args.pivot_windows:
                for mad in args.max_anchor_distances:
                    tasks.append((s.upper(), tf, pw, mad,
                                  args.lookahead_bars, args.max_bars_per_tf,
                                  str(args.out_dir)))
    print(f"[scale] tasks={len(tasks)}  jobs={args.jobs}")
    print(f"[scale] sym={args.symbols}  tf={args.timeframes}")
    print(f"[scale] pivot_window={args.pivot_windows}  max_anchor_distance={args.max_anchor_distances}")

    t0 = time.time()
    n_done, n_skipped, n_records, n_failed = 0, 0, 0, 0

    if args.jobs <= 1:
        results = (_process_one(t) for t in tasks)
    else:
        with mp.Pool(args.jobs) as pool:
            results = pool.imap_unordered(_process_one, tasks)
            results = list(results)

    for path, n in results:
        n_done += 1
        if n < 0:
            n_skipped += 1
        elif n == 0:
            n_failed += 1
        else:
            n_records += n
        if n_done % 5 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(1e-6, elapsed)
            print(f"[scale] {n_done}/{len(tasks)}  records={n_records:,}  "
                  f"skipped={n_skipped} failed={n_failed}  "
                  f"rate={rate:.2f}/s eta={int((len(tasks)-n_done)/max(1e-6,rate))}s")

    elapsed = time.time() - t0
    print(f"[scale] DONE in {elapsed:.0f}s. records={n_records:,}  "
          f"skipped={n_skipped} failed={n_failed}")
    print(f"[scale] artifacts in {args.out_dir}/")


if __name__ == "__main__":
    main()
