"""SR-Params parameter sweep: explode 271k auto trendlines into millions.

Reuses the existing sr_patterns detector, but iterates over a Cartesian
product of SRParams. Same OHLCV, different params -> different lines.

Output: data/patterns_sweep/<symbol>_<tf>__<params_hash>.jsonl
        (separate files per param combo so re-running is idempotent)

Usage:
    python -m scripts.sweep_sr_params_full \\
        --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --timeframes 5m 15m 1h 4h \\
        --max-trials 1000 --jobs 4

Stop / resume: each (symbol, tf, params_hash) is a separate file. If it
already exists and is non-empty, the script skips it. Killing and
re-running picks up where it left off.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import polars as pl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trendline_tokenizer.evolve.draw import _ohlcv_dataframe   # noqa: E402
from trendline_tokenizer.inference.runtime_detector import detect_lines  # noqa: E402


OUT_DIR = ROOT / "data" / "patterns_sweep"


# Coarse grid by default. Each axis is independently doubled / halved
# from the sr_patterns defaults. Cartesian product = 3^k combos.
DEFAULT_GRID: dict[str, list] = {
    "lookback":         [60, 120, 240, 480],
    "pivot_strength":   [2, 3, 5],
    "min_touches":      [2, 3],
    "tolerance":        [0.001, 0.002, 0.005, 0.01],
    "slope_eps":        [1e-5, 5e-5, 2e-4],
}


def _params_hash(params: dict) -> str:
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _enumerate_grid(grid: dict[str, list], max_trials: int | None) -> list[dict]:
    keys = list(grid.keys())
    pools = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*pools):
        combos.append({k: v for k, v in zip(keys, vals)})
        if max_trials and len(combos) >= max_trials:
            break
    return combos


def _process_one(symbol: str, tf: str, params: dict, out_dir: Path) -> tuple[str, int]:
    h = _params_hash(params)
    out_path = out_dir / f"{symbol}_{tf}__{h}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        return (str(out_path), -1)   # skipped
    df = _ohlcv_dataframe(symbol, tf)
    if df is None or len(df) < 100:
        return (str(out_path), 0)
    lines = detect_lines(df, symbol=symbol, timeframe=tf, sr_params=params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in lines:
            fh.write(r.model_dump_json() + "\n")
    return (str(out_path), len(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--max-trials", type=int, default=None,
                    help="cap total param combos (default: full grid)")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() // 2 - 1))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combos = _enumerate_grid(DEFAULT_GRID, args.max_trials)
    print(f"[sweep] symbols={args.symbols}  tfs={args.timeframes}  param_combos={len(combos)}")
    print(f"[sweep] total tasks={len(args.symbols) * len(args.timeframes) * len(combos)}")

    tasks = [(s, tf, p, args.out_dir)
             for s in args.symbols for tf in args.timeframes for p in combos]
    t0 = time.time()
    n_done, n_skipped, n_lines, n_failed = 0, 0, 0, 0

    if args.jobs <= 1:
        results = (_process_one(*t) for t in tasks)
    else:
        with mp.Pool(args.jobs) as pool:
            results = pool.starmap_async(_process_one, tasks).get()

    for path, n in results:
        n_done += 1
        if n < 0:
            n_skipped += 1
        elif n == 0:
            n_failed += 1
        else:
            n_lines += n
        if n_done % 50 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(1e-6, elapsed)
            print(f"[sweep] {n_done}/{len(tasks)}  lines={n_lines:,}  "
                  f"skipped={n_skipped} failed={n_failed}  "
                  f"rate={rate:.1f}/s eta={int((len(tasks)-n_done)/max(1e-6,rate))}s")

    elapsed = time.time() - t0
    print(f"[sweep] done in {elapsed:.0f}s. lines={n_lines:,}  "
          f"skipped={n_skipped} failed={n_failed}")
    print(f"[sweep] artifacts in {args.out_dir}/")


if __name__ == "__main__":
    main()
