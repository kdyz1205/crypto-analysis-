"""Run one evolve round from the command line.

Usage:
    python -m trendline_tokenizer.cli.evolve_round --round 0 --symbols BTCUSDT,HYPEUSDT --timeframes 1h,4h
"""
from __future__ import annotations

import argparse
import json

from ..evolve.rounds import run_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,HYPEUSDT,ZECUSDT")
    ap.add_argument("--timeframes", default="1h,4h")
    ap.add_argument("--max-lines", type=int, default=100)
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    summary = run_round(
        round_id=args.round,
        symbols=symbols,
        timeframes=tfs,
        max_lines_per_sym_tf=args.max_lines,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
