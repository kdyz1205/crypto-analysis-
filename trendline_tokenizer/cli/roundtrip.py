"""Run encode → decode on the user's real data, print metrics.

Usage:
    python -m trendline_tokenizer.cli.roundtrip
    python -m trendline_tokenizer.cli.roundtrip --limit 10000
    python -m trendline_tokenizer.cli.roundtrip --out data/tokenized/rule_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from ..adapters import load_manual_records, iter_legacy_pattern_records
from ..tokenizer import encode_rule, decode_rule
from ..tokenizer.metrics import round_trip_error, summarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUAL = PROJECT_ROOT / "data" / "manual_trendlines.json"
DEFAULT_PATTERNS_DIR = PROJECT_ROOT / "data" / "patterns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", default=str(DEFAULT_MANUAL))
    ap.add_argument("--patterns-dir", default=str(DEFAULT_PATTERNS_DIR))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of auto records (default: all)")
    ap.add_argument("--out", default=None,
                    help="write tokenised records as JSONL")
    args = ap.parse_args()

    # Manual set
    manual_recs = load_manual_records(args.manual)
    print(f"[roundtrip] manual records: {len(manual_recs)}")

    manual_tokens = []
    manual_errs = []
    coarse_hist, fine_hist = Counter(), Counter()
    for r in manual_recs:
        tok = encode_rule(r)
        manual_tokens.append(tok)
        dec = decode_rule(tok, reference_record=r)
        manual_errs.append(round_trip_error(r, dec))
        coarse_hist[tok.coarse_token_id] += 1
        fine_hist[tok.fine_token_id] += 1
    if manual_recs:
        print("[roundtrip] manual summary:", summarize(manual_errs))

    # Auto / legacy set — streaming, no full materialisation
    patterns_dir = Path(args.patterns_dir)
    jsonl_files = sorted(patterns_dir.glob("*.jsonl")) if patterns_dir.exists() else []
    print(f"[roundtrip] legacy jsonl files: {len(jsonl_files)}")

    out_fh = None
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_fh = outp.open("w", encoding="utf-8")

    errs = []
    total = 0
    t0 = time.time()
    for f in jsonl_files:
        for r in iter_legacy_pattern_records(f):
            tok = encode_rule(r)
            dec = decode_rule(tok, reference_record=r)
            errs.append(round_trip_error(r, dec))
            coarse_hist[tok.coarse_token_id] += 1
            fine_hist[tok.fine_token_id] += 1
            if out_fh is not None:
                out_fh.write(json.dumps({
                    "record_id": r.id,
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "coarse": tok.coarse_token_id,
                    "fine": tok.fine_token_id,
                    "version": tok.tokenizer_version,
                }) + "\n")
            total += 1
            if args.limit and total >= args.limit:
                break
        if args.limit and total >= args.limit:
            break

    if out_fh is not None:
        out_fh.close()

    dt = time.time() - t0
    rate = total / max(dt, 1e-9)
    print(f"[roundtrip] auto records encoded+decoded: {total} ({dt:.1f}s, {rate:.0f}/s)")
    print(f"[roundtrip] auto summary: {summarize(errs)}")
    print(f"[roundtrip] distinct coarse tokens used: {len(coarse_hist)} / 5040 "
          f"({100*len(coarse_hist)/5040:.1f} %)")
    print(f"[roundtrip] distinct fine tokens used:   {len(fine_hist)} / 21600 "
          f"({100*len(fine_hist)/21600:.1f} %)")

    # Top-10 coarse tokens
    print("[roundtrip] top-10 coarse tokens (freq):")
    for tok_id, n in coarse_hist.most_common(10):
        print(f"    coarse={tok_id:>5}  n={n}")


if __name__ == "__main__":
    main()
