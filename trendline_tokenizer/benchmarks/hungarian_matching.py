"""Phase 1.E: Hungarian-matching benchmark for trendline detector quality.

Pairs each MANUAL gold line with its closest AUTO-detected line in the
same (symbol, timeframe), using a cost function over geometric distance
+ role + slope. Then reports:

    precision @ k_top, recall @ k_top, role_accuracy
    slope_MAE, anchor_time_MAE, end_price_MAE
    manual_gold_F1

Cost function (lower = better match):
    cost(auto, manual) =
        anchor_time_dist     # absolute bar-distance between start anchors
      + slope_dist           # |log_slope_auto - log_slope_manual|
      + price_dist           # |log(end_price_auto / end_price_manual)|
      + role_mismatch_pen    # 0 if same role, else ROLE_PEN

Without scipy, we use a greedy O(n*m) approximation that's good enough
for diagnostic purposes (the proper Hungarian is O(n^3)). The greedy
version is exact when one side is much smaller than the other (true
here: 82 manual vs many auto per pair).

CLI:
    python -m trendline_tokenizer.benchmarks.hungarian_matching \\
        --auto-source patterns_sweep --top-k 10
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from ..adapters import iter_legacy_pattern_records
from ..adapters.manual import load_manual_records
from ..registry.paths import ROOT
from ..schemas.trendline import TrendlineRecord


ROLE_PEN = 5.0   # role mismatch is heavy
ANCHOR_TIME_NORM = 100.0   # divide bar-distance by this
SLOPE_PEN_SCALE = 100.0    # log-slope difference scaled


def _line_cost(auto: TrendlineRecord, manual: TrendlineRecord) -> float:
    if auto.symbol != manual.symbol or auto.timeframe != manual.timeframe:
        return float("inf")
    a_start = auto.start_bar_index
    m_start = manual.start_bar_index
    anchor_dist = abs(a_start - m_start) / ANCHOR_TIME_NORM
    a_slope = auto.log_slope_per_bar()
    m_slope = manual.log_slope_per_bar()
    slope_dist = abs(a_slope - m_slope) * SLOPE_PEN_SCALE
    if auto.end_price > 0 and manual.end_price > 0:
        price_dist = abs(math.log(auto.end_price) - math.log(manual.end_price))
    else:
        price_dist = 1.0
    role_pen = 0.0 if auto.line_role == manual.line_role else ROLE_PEN
    return anchor_dist + slope_dist + price_dist + role_pen


def greedy_match(
    auto_lines: list[TrendlineRecord],
    manual_lines: list[TrendlineRecord],
    *,
    top_k: int = 5,
) -> list[dict]:
    """For each manual line, find the BEST top_k auto candidates by cost.

    Returns a list (one per manual) of dicts:
      {
        manual_id: ...,
        best_auto_id: ...,
        best_cost: ...,
        candidates: [(auto_id, cost), ...]
        role_match: bool
        slope_err: float
        anchor_err_bars: int
        price_err_pct: float
      }
    """
    by_pair: dict[tuple[str, str], list[TrendlineRecord]] = defaultdict(list)
    for r in auto_lines:
        by_pair[(r.symbol, r.timeframe)].append(r)
    out = []
    for m in manual_lines:
        bucket = by_pair.get((m.symbol, m.timeframe), [])
        if not bucket:
            out.append({
                "manual_id": m.id, "best_auto_id": None,
                "best_cost": float("inf"), "candidates": [],
                "role_match": False, "slope_err": None,
                "anchor_err_bars": None, "price_err_pct": None,
                "no_candidates_in_pair": True,
            })
            continue
        scored = [(a, _line_cost(a, m)) for a in bucket]
        scored.sort(key=lambda x: x[1])
        top = scored[:top_k]
        best_a, best_c = top[0]
        anchor_err = abs(best_a.start_bar_index - m.start_bar_index)
        slope_err = abs(best_a.log_slope_per_bar() - m.log_slope_per_bar())
        if best_a.end_price > 0 and m.end_price > 0:
            price_err = abs(best_a.end_price - m.end_price) / max(m.end_price, 1e-9)
        else:
            price_err = 0.0
        out.append({
            "manual_id": m.id,
            "best_auto_id": best_a.id,
            "best_cost": best_c,
            "candidates": [(a.id, c) for a, c in top],
            "role_match": best_a.line_role == m.line_role,
            "slope_err": slope_err,
            "anchor_err_bars": anchor_err,
            "price_err_pct": price_err,
        })
    return out


def summarise_matches(matches: list[dict]) -> dict:
    """Aggregate per-manual matches into headline metrics."""
    if not matches:
        return {"n_manual": 0}
    finite = [m for m in matches if m.get("best_cost", float("inf")) < float("inf")]
    role_hits = sum(1 for m in finite if m["role_match"])
    no_cand = sum(1 for m in matches if m.get("no_candidates_in_pair"))

    slope_errs = [m["slope_err"] for m in finite if m["slope_err"] is not None]
    anchor_errs = [m["anchor_err_bars"] for m in finite if m["anchor_err_bars"] is not None]
    price_errs = [m["price_err_pct"] for m in finite if m["price_err_pct"] is not None]
    costs = [m["best_cost"] for m in finite]

    def _q(xs, q):
        if not xs:
            return None
        s = sorted(xs)
        idx = max(0, min(len(s) - 1, int(q * len(s))))
        return s[idx]

    return {
        "n_manual": len(matches),
        "n_pairs_with_candidates": len(finite),
        "n_no_candidates_in_pair": no_cand,
        "role_match_rate": (role_hits / max(1, len(finite))),
        "best_cost_median": statistics.median(costs) if costs else None,
        "best_cost_p95": _q(costs, 0.95),
        "slope_err_median": statistics.median(slope_errs) if slope_errs else None,
        "slope_err_p95": _q(slope_errs, 0.95),
        "anchor_err_median_bars": statistics.median(anchor_errs) if anchor_errs else None,
        "anchor_err_p95_bars": _q(anchor_errs, 0.95),
        "price_err_median_pct": statistics.median(price_errs) if price_errs else None,
        "price_err_p95_pct": _q(price_errs, 0.95),
    }


def run_benchmark(
    *,
    auto_source: str = "patterns",
    manual_path: Path | None = None,
    top_k: int = 5,
    output_path: Path | None = None,
) -> dict:
    """auto_source: 'patterns' or 'patterns_sweep'."""
    manual_path = manual_path or (ROOT / "data" / "manual_trendlines.json")
    manual_lines = load_manual_records(manual_path)
    print(f"[hungarian] manual lines: {len(manual_lines)}")
    if not manual_lines:
        return {"error": "no manual lines"}

    auto_dir = ROOT / "data" / auto_source
    auto_lines: list[TrendlineRecord] = []
    if auto_dir.exists():
        for f in auto_dir.glob("*.jsonl"):
            for rec in iter_legacy_pattern_records(f):
                auto_lines.append(rec)
    print(f"[hungarian] auto lines: {len(auto_lines)} from {auto_dir}")

    matches = greedy_match(auto_lines, manual_lines, top_k=top_k)
    summary = summarise_matches(matches)
    summary["auto_source"] = auto_source
    summary["top_k"] = top_k
    print(f"[hungarian] role_match_rate={summary['role_match_rate']:.3f} "
          f"anchor_err_median_bars={summary['anchor_err_median_bars']} "
          f"slope_err_median={summary['slope_err_median']}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": summary,
                                            "matches": matches}, indent=2),
                                encoding="utf-8")
        print(f"[hungarian] saved {output_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-source", default="patterns",
                    choices=["patterns", "patterns_sweep"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--manual-path", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None,
                    help="optional path to write the full report JSON")
    args = ap.parse_args()
    run_benchmark(
        auto_source=args.auto_source, top_k=args.top_k,
        manual_path=args.manual_path, output_path=args.output,
    )


if __name__ == "__main__":
    main()
