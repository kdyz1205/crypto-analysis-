"""Convert trade_snapshots JSONL files to columnar Parquet.

Run manually or via cron:

    python scripts/convert_snapshots_to_parquet.py [--symbol HYPEUSDT]

For each month-file in data/logs/trade_snapshots/{SYMBOL}/{YYYYMM}.jsonl
(and its companion .outcomes.jsonl), produce a joined Parquet file at

    data/logs/trade_snapshots/{SYMBOL}/{YYYYMM}.parquet

Joined = latest outcome merged into each trade's outcome block, so a
single Parquet row is everything you need for analysis.

Why Parquet: JSONL is append-only and slow to query at scale. Parquet
is columnar — loading "all rows where leverage=10 and touch_number>=3"
only reads the 2 relevant columns, not every byte of every row.

This script is idempotent: re-running overwrites the Parquet with the
latest join. The JSONL files are the source of truth and stay as-is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make server modules importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pandas as pd
except ImportError:
    print("pandas required — pip install pandas", file=sys.stderr)
    sys.exit(1)

from server.conditionals.snapshots import (
    _snapshots_dir,
    iter_snapshots_joined,
)


def _flatten(snap: dict) -> dict:
    """Flatten nested dicts to columns: 'trade_params.entry_price' etc.

    Pandas/Parquet handle nested structs but they're awkward to query;
    flat columns are more useful for analysis.
    """
    out = {}
    def walk(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                walk(key, v)
        elif isinstance(obj, list):
            # Stash lists as JSON strings — Parquet can store them directly
            # but querying is harder. Comma-joined for primitive lists,
            # JSON for complex ones.
            import json
            out[prefix] = json.dumps(obj, ensure_ascii=False)
        else:
            out[prefix] = obj
    walk("", snap)
    return out


def convert_symbol(symbol: str) -> int:
    """Convert all month files for one symbol. Returns row count."""
    sym_dir = _snapshots_dir() / symbol.upper()
    if not sym_dir.exists():
        return 0

    # Group snapshots by YYYYMM
    by_month: dict[str, list[dict]] = {}
    for snap in iter_snapshots_joined(symbol):
        ts = int(snap.get("ts") or 0)
        if ts == 0:
            continue
        from datetime import datetime, timezone
        ym = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m")
        by_month.setdefault(ym, []).append(_flatten(snap))

    total = 0
    for ym, rows in by_month.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_file = sym_dir / f"{ym}.parquet"
        df.to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")
        print(f"  {symbol} {ym}: {len(rows)} row(s) → {out_file.name}")
        total += len(rows)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert trade snapshots to Parquet")
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Only convert this symbol (default: all)",
    )
    args = parser.parse_args()

    root = _snapshots_dir()
    if not root.exists():
        print(f"no snapshots directory at {root}", file=sys.stderr)
        return 0

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = [p.name for p in root.iterdir() if p.is_dir()]

    grand_total = 0
    for sym in symbols:
        n = convert_symbol(sym)
        grand_total += n

    print(f"\nConverted {grand_total} row(s) across {len(symbols)} symbol(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
