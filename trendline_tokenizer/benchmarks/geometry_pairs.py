"""Phase 1.T3 helper: derive geometric-consistency labels from a pool
of trendlines. Pairs two lines from the same (symbol, timeframe) and
classifies their joint structure:

  channel:           parallel, opposite roles (support+resistance)
  triangle:          converging, opposite roles
  wedge:             same-role pair tightening (rare)
  parallel_same:     parallel, same role (e.g. two supports)
  diverging:         opposite roles, slopes diverging
  unrelated:         too far apart in time / inconsistent

This is the LABEL generator for the T3 self-supervised classification
task. The classifier itself goes in models/heads.py later; here we
just produce (line_a, line_b, label) triples that any classifier
can consume.

Used as both:
  - Training labels for a future classifier
  - A standalone benchmark (do auto-paired lines match the manual
    ones the user labels?)
"""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from ..schemas.trendline import TrendlineRecord


GeometryLabel = Literal[
    "channel", "triangle", "wedge",
    "parallel_same", "diverging", "unrelated",
]


@dataclass
class LinePair:
    line_a_id: str
    line_b_id: str
    label: GeometryLabel
    angle_deg: float                   # angle between the two lines
    end_distance_atr: float            # how close their right-edges end up
    overlap_bars: int                  # bars they're both active
    notes: str = ""


def _slope_angle_deg(slope: float) -> float:
    """Convert log-slope-per-bar to angle in degrees (atan(slope*100))."""
    return math.degrees(math.atan(slope * 100.0))


def _bars_overlap(a: TrendlineRecord, b: TrendlineRecord) -> int:
    start = max(a.start_bar_index, b.start_bar_index)
    end = min(a.end_bar_index, b.end_bar_index)
    return max(0, end - start)


def _end_distance(a: TrendlineRecord, b: TrendlineRecord) -> float:
    """Vertical price gap at the right-edge (max of the two ends), in
    fraction of mid-price."""
    end = max(a.end_bar_index, b.end_bar_index)
    # Project both lines to `end` bar
    a_at_end = a.end_price + a.slope_per_bar() * (end - a.end_bar_index)
    b_at_end = b.end_price + b.slope_per_bar() * (end - b.end_bar_index)
    mid = (abs(a_at_end) + abs(b_at_end)) / 2 + 1e-9
    return abs(a_at_end - b_at_end) / mid


def label_pair(a: TrendlineRecord, b: TrendlineRecord) -> LinePair | None:
    """Classify a pair of trendlines from the same (sym, tf).

    Returns None if the pair has insufficient overlap (no joint structure).
    """
    if a.symbol != b.symbol or a.timeframe != b.timeframe:
        return None
    overlap = _bars_overlap(a, b)
    if overlap < 5:
        return None

    a_slope = a.log_slope_per_bar()
    b_slope = b.log_slope_per_bar()
    a_angle = _slope_angle_deg(a_slope)
    b_angle = _slope_angle_deg(b_slope)
    angle_diff = abs(a_angle - b_angle)
    end_dist = _end_distance(a, b)

    same_role = a.line_role == b.line_role
    opposite_role = (
        (a.line_role, b.line_role) in
        [("support", "resistance"), ("resistance", "support"),
         ("channel_lower", "channel_upper"), ("channel_upper", "channel_lower")]
    )

    label: GeometryLabel
    notes = ""

    # Triangle / wedge: lines converging (slopes pointing toward each other)
    converging = (a_slope > 0 and b_slope < 0) or (a_slope < 0 and b_slope > 0)
    parallel = angle_diff < 5.0

    if converging and opposite_role:
        label = "triangle"
    elif converging and same_role:
        label = "wedge"
        notes = "same role + converging is rare; usually rejected"
    elif parallel and opposite_role:
        label = "channel"
    elif parallel and same_role:
        label = "parallel_same"
    elif (a_slope > 0 and b_slope > 0 and same_role) or (
            a_slope < 0 and b_slope < 0 and same_role):
        label = "parallel_same"
    elif opposite_role and not converging and angle_diff > 10:
        label = "diverging"
    else:
        label = "unrelated"

    return LinePair(
        line_a_id=a.id, line_b_id=b.id, label=label,
        angle_deg=angle_diff, end_distance_atr=end_dist,
        overlap_bars=overlap, notes=notes,
    )


def all_pairs_within_pair(records: Sequence[TrendlineRecord],
                          *, max_per_pair: int | None = None) -> list[LinePair]:
    """Generate all valid pairs WITHIN each (symbol, timeframe).

    O(n^2) per pair; capped by max_per_pair to bound compute.
    """
    by_pair: dict[tuple[str, str], list[TrendlineRecord]] = defaultdict(list)
    for r in records:
        by_pair[(r.symbol, r.timeframe)].append(r)
    out: list[LinePair] = []
    for (sym, tf), bucket in by_pair.items():
        # cap per (sym, tf) to keep pair count bounded
        if max_per_pair and len(bucket) > max_per_pair:
            bucket = bucket[:max_per_pair]
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                p = label_pair(bucket[i], bucket[j])
                if p is not None:
                    out.append(p)
    return out


def label_distribution(pairs: Iterable[LinePair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.label] = counts.get(p.label, 0) + 1
    return counts
