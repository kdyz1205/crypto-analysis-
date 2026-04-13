"""Inline reflection: read v1 traces, find failure clusters, print targeted insights.

This replaces the subagent-based reflection when API budget is exhausted.
Deterministic, data-driven, no LLM call.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_traces(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict]) -> dict:
    triggered = [r for r in rows if r["fade_triggered"]]
    if not triggered:
        return {"n": 0, "wr": 0.0, "avg_R": 0.0, "total_R": 0.0}
    wins = sum(1 for r in triggered if r["fade_R"] > 0)
    return {
        "n": len(triggered),
        "n_lines": len(rows),
        "wr": wins / len(triggered),
        "avg_R": mean(r["total_R"] for r in triggered),
        "total_R": sum(r["total_R"] for r in triggered),
    }


def slice_by(rows: list[dict], predicate) -> list[dict]:
    return [r for r in rows if predicate(r)]


MIN_BUCKET_N = 20  # sample-size cutoff — below this the signal is noise


def scan_feature(rows: list[dict], feature: str, bins: list[tuple]) -> list[tuple]:
    """Scan a numeric feature across bins. Only returns buckets with
    enough samples to be significant (>= MIN_BUCKET_N triggered).
    """
    out = []
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r.get(feature, 0) < hi]
        agg = aggregate(sub)
        if agg.get("n", 0) >= MIN_BUCKET_N:
            out.append((f"{lo}..{hi}", agg))
    return out


def main():
    path = Path(__file__).parent.parent.parent / "data/evolution/rounds/round_00/v1_clean_traces.jsonl"
    rows = load_traces(path)
    # CRITICAL: reflect on TRAIN only. Using TEST for reflection and then
    # evaluating a variant on the same TEST set is in-sample lookahead.
    # Round 1 filters (v1a) were derived from test-split reflection → any
    # "improvement" on test is fitting the test distribution.
    test = [r for r in rows if r.get("split") == "train"]
    print(f"v1 TRAIN traces (used for reflection — NOT touching test): {len(test)}")

    overall = aggregate(test)
    print(f"overall: {overall}")
    print()

    # Per-touch (we already saw this, confirm)
    print("=== per setup_touch_number ===")
    by_touch: dict[int, list[dict]] = defaultdict(list)
    for r in test:
        by_touch[r.get("setup_touch_number", 0)].append(r)
    for k in sorted(by_touch.keys()):
        agg = aggregate(by_touch[k])
        print(f"  touch={k}: {agg}")
    print()

    # By span_bars buckets
    print("=== span_bars buckets (test, triggered only) ===")
    bins = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 250), (250, 500), (500, 10000)]
    for label, agg in scan_feature([r for r in test if r["fade_triggered"]], "span_bars", bins):
        print(f"  span {label}: {agg}")
    print()

    # By slope_pct_per_bar sign × side (trending vs counter-trend)
    print("=== slope sign × side (triggered only) ===")
    trig = [r for r in test if r["fade_triggered"]]
    for side in ("support", "resistance"):
        for tag, pred in [
            ("upslope",    lambda r: r["slope_pct_per_bar"] > 0.0005),
            ("flat",       lambda r: -0.0005 <= r["slope_pct_per_bar"] <= 0.0005),
            ("downslope",  lambda r: r["slope_pct_per_bar"] < -0.0005),
        ]:
            sub = [r for r in trig if r["side"] == side and pred(r)]
            if sub:
                agg = aggregate(sub)
                print(f"  {side:10s} {tag:10s}: {agg}")
    print()

    # By total_touch_count (quality of source line)
    print("=== total_touch_count (source line) ===")
    for ttc in sorted({r["total_touch_count"] for r in test}):
        sub = [r for r in test if r["total_touch_count"] == ttc]
        agg = aggregate(sub)
        if agg["n"] > 0:
            print(f"  total_touch={ttc}: {agg}")
    print()

    # By vol_regime
    print("=== vol_regime (triggered only) ===")
    for vr in ("low", "normal", "high"):
        sub = [r for r in trig if r["vol_regime"] == vr]
        if sub:
            agg = aggregate(sub)
            print(f"  {vr:6s}: {agg}")
    print()

    # By timeframe
    print("=== timeframe (triggered only) ===")
    for tf in sorted({r["timeframe"] for r in trig}):
        sub = [r for r in trig if r["timeframe"] == tf]
        if sub:
            agg = aggregate(sub)
            print(f"  {tf}: {agg}")
    print()

    # By span_pct_of_available (line length relative to available history)
    print("=== span_pct buckets ===")
    bins_pct = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
    for label, agg in scan_feature(trig, "span_pct_of_available", bins_pct):
        print(f"  span_pct {label}: {agg}")
    print()

    # === Synthetic filter combinations ===
    print("=== targeted filter recommendations ===")

    # Filter 1: drop setups with touch_number >= 5
    sub1 = [r for r in trig if r["setup_touch_number"] <= 4]
    agg1 = aggregate(sub1)
    print(f"  [touch <= 4]: {agg1}")

    # Filter 2: drop counter-trend support (downsloping support)
    sub2 = [r for r in trig if not (r["side"] == "support" and r["slope_pct_per_bar"] < -0.0005)]
    agg2 = aggregate(sub2)
    print(f"  [drop downslope support]: {agg2}")

    # Filter 3: drop too-short spans
    sub3 = [r for r in trig if r["span_bars"] >= 30]
    agg3 = aggregate(sub3)
    print(f"  [span >= 30]: {agg3}")

    # Combined
    sub4 = [r for r in trig
            if r["setup_touch_number"] <= 4
            and not (r["side"] == "support" and r["slope_pct_per_bar"] < -0.0005)
            and not (r["side"] == "resistance" and r["slope_pct_per_bar"] > 0.0005)
            and r["span_bars"] >= 30]
    agg4 = aggregate(sub4)
    print(f"  [ALL combined (touch<=4, same-trend, span>=30)]: {agg4}")

    # Winning: vol normal + touch 3-4 + same-trend
    sub5 = [r for r in trig
            if r["setup_touch_number"] in (3, 4)
            and r["vol_regime"] == "normal"
            and ((r["side"] == "support" and r["slope_pct_per_bar"] >= -0.0005)
                 or (r["side"] == "resistance" and r["slope_pct_per_bar"] <= 0.0005))]
    agg5 = aggregate(sub5)
    print(f"  [touch3-4 + normal vol + same-trend]: {agg5}")


if __name__ == "__main__":
    main()
