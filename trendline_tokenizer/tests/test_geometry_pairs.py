"""Tests for the geometry-pair labeller (Phase 1.T3 label generator)."""
from __future__ import annotations

from trendline_tokenizer.benchmarks.geometry_pairs import (
    label_pair, all_pairs_within_pair, label_distribution,
    _slope_angle_deg, _bars_overlap,
)
from trendline_tokenizer.schemas.trendline import TrendlineRecord


def _r(rid, *, role="support", start_idx=10, end_idx=100,
       start_price=100.0, end_price=102.0,
       symbol="BTC", timeframe="5m") -> TrendlineRecord:
    return TrendlineRecord(
        id=rid, symbol=symbol, exchange="bitget", timeframe=timeframe,
        start_time=start_idx * 60, end_time=end_idx * 60,
        start_bar_index=start_idx, end_bar_index=end_idx,
        start_price=start_price, end_price=end_price,
        line_role=role, direction="up" if end_price > start_price else "down",
        touch_count=2, label_source="auto", created_at=0,
    )


def test_slope_angle_basic():
    assert abs(_slope_angle_deg(0.0)) < 1e-6
    a = _slope_angle_deg(0.01)
    assert 40 < a < 50    # atan(1) ~= 45deg


def test_bars_overlap():
    a = _r("a", start_idx=10, end_idx=50)
    b = _r("b", start_idx=30, end_idx=70)
    assert _bars_overlap(a, b) == 20
    c = _r("c", start_idx=200, end_idx=300)
    assert _bars_overlap(a, c) == 0


def test_label_pair_different_pair_returns_none():
    a = _r("a", symbol="BTC")
    b = _r("b", symbol="ETH")
    assert label_pair(a, b) is None


def test_label_pair_no_overlap_returns_none():
    a = _r("a", start_idx=10, end_idx=20)
    b = _r("b", start_idx=100, end_idx=200)
    assert label_pair(a, b) is None


def test_label_pair_channel_parallel_opposite_roles():
    """Two lines with same slope (parallel) and opposite roles -> channel."""
    sup = _r("sup", role="support", start_idx=10, end_idx=100,
             start_price=100, end_price=110)   # +0.01/bar
    res = _r("res", role="resistance", start_idx=10, end_idx=100,
             start_price=120, end_price=130)   # +0.01/bar
    p = label_pair(sup, res)
    assert p is not None
    assert p.label == "channel"


def test_label_pair_triangle_converging_opposite_roles():
    """Ascending support meets descending resistance -> triangle."""
    sup = _r("sup", role="support", start_idx=10, end_idx=100,
             start_price=100, end_price=110)        # rising
    res = _r("res", role="resistance", start_idx=10, end_idx=100,
             start_price=130, end_price=120)        # falling
    p = label_pair(sup, res)
    assert p is not None
    assert p.label == "triangle"


def test_label_pair_parallel_same_role():
    sup1 = _r("s1", role="support", start_idx=10, end_idx=100,
              start_price=100, end_price=110)
    sup2 = _r("s2", role="support", start_idx=10, end_idx=100,
              start_price=95, end_price=105)
    p = label_pair(sup1, sup2)
    assert p is not None
    assert p.label == "parallel_same"


def test_all_pairs_within_pair_only_same_sym_tf():
    recs = [
        _r("a", symbol="BTC"),
        _r("b", symbol="BTC"),
        _r("c", symbol="ETH"),
    ]
    pairs = all_pairs_within_pair(recs)
    # Only the BTC-BTC pair counts (1)
    assert len(pairs) == 1


def test_label_distribution():
    recs = [
        _r("sup1", role="support", start_idx=10, end_idx=100,
           start_price=100, end_price=110),
        _r("res1", role="resistance", start_idx=10, end_idx=100,
           start_price=130, end_price=120),  # converging w/ sup1 -> triangle
        _r("sup2", role="support", start_idx=10, end_idx=100,
           start_price=95, end_price=105),  # parallel w/ sup1
    ]
    pairs = all_pairs_within_pair(recs)
    dist = label_distribution(pairs)
    assert dist.get("triangle", 0) >= 1
    assert dist.get("parallel_same", 0) >= 1


# ── Per-record pattern labels ─────────────────────────────────────────

def test_per_record_pattern_priority():
    """When a record participates in multiple pair structures, the
    higher-priority (more-distinctive) one wins."""
    from trendline_tokenizer.benchmarks.geometry_pairs import per_record_pattern_labels
    # sup1 + res1 form a triangle (converging opposite roles).
    # sup2 only parallels sup1 (kept far enough from res1 by symbol).
    recs = [
        _r("sup1", role="support", start_idx=10, end_idx=100,
           start_price=100, end_price=110),
        _r("res1", role="resistance", start_idx=10, end_idx=100,
           start_price=130, end_price=120),
        _r("sup2_iso", role="support", start_idx=10, end_idx=100,
           start_price=95, end_price=105, symbol="ETH"),   # different symbol, isolated
    ]
    labels = per_record_pattern_labels(recs)
    # sup1 is in a triangle with res1. Triangle wins.
    assert labels["sup1"] == "triangle"
    assert labels["res1"] == "triangle"
    # sup2 is on a different symbol — no pair, falls to 'unrelated'
    assert labels["sup2_iso"] == "unrelated"


def test_per_record_pattern_priority_when_multiple_pairs():
    """sup1 in BOTH triangle + parallel_same — triangle wins."""
    from trendline_tokenizer.benchmarks.geometry_pairs import per_record_pattern_labels
    recs = [
        _r("sup1", role="support", start_idx=10, end_idx=100,
           start_price=100, end_price=110),
        _r("res1", role="resistance", start_idx=10, end_idx=100,
           start_price=130, end_price=120),  # triangle w/ sup1
        _r("sup2", role="support", start_idx=10, end_idx=100,
           start_price=95, end_price=105),   # parallel w/ sup1
    ]
    labels = per_record_pattern_labels(recs)
    # sup1 in (triangle vs parallel_same) -> triangle wins
    assert labels["sup1"] == "triangle"


def test_per_record_pattern_isolated_record_is_unrelated():
    from trendline_tokenizer.benchmarks.geometry_pairs import per_record_pattern_labels
    recs = [_r("solo", symbol="BTC")]
    labels = per_record_pattern_labels(recs)
    assert labels["solo"] == "unrelated"
