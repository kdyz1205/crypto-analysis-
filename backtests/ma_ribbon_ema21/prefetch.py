"""Prefetch Bitget OHLCV CSVs for a chunk of symbols.

Why this exists: a single CLI run on 45 symbols x 4 TFs would exceed the
10-minute background-bash timeout. Prefetch in 5-symbol chunks, then run
the scan+report once the cache is full (the scan itself is sub-second).

Usage:
    python -m backtests.ma_ribbon_ema21.prefetch --symbols BTCUSDT ETHUSDT --tfs 5m 15m 1h 4h --pages 30
"""
from __future__ import annotations
import argparse
import logging
import sys

from backtests.ma_ribbon_ema21.data_loader import (
    DataLoaderConfig, load_or_fetch, csv_path,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--tfs", nargs="+", default=["5m", "15m", "1h", "4h"])
    p.add_argument("--pages", type=int, default=30)
    p.add_argument("--cache-dir", default="data/csv_cache/ma_ribbon_ema21")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = DataLoaderConfig(cache_dir=args.cache_dir,
                           bitget_pages_per_symbol=args.pages)

    fetched = 0
    skipped = 0
    for sym in args.symbols:
        for tf in args.tfs:
            cp = csv_path(sym, tf, cfg)
            if cp.exists() and cp.stat().st_size > 100:
                print(f"  CACHED  {sym:12s} {tf:4s}  ({cp.stat().st_size // 1024} KB)")
                skipped += 1
                continue
            df = load_or_fetch(sym, tf, cfg)
            n = len(df)
            print(f"  FETCHED {sym:12s} {tf:4s}  {n} bars")
            fetched += 1
    print(f"\nDone. fetched={fetched}, skipped(cached)={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
