"""Tests for the Hungarian-matching benchmark."""
from __future__ import annotations
import math

from trendline_tokenizer.benchmarks.hungarian_matching import (
    _line_cost, greedy_match, summarise_matches,
)
from trendline_tokenizer.schemas.trendline import TrendlineRecord


def _r(rid: str, *, symbol: str = "BTC", timeframe: str = "5m",
       start_idx: int = 10, end_idx: int = 50,
       start_price: float = 100.0, end_price: float = 102.0,
       role: str = "support") -> TrendlineRecord:
    return TrendlineRecord(
        id=rid, symbol=symbol, exchange="bitget", timeframe=timeframe,
        start_time=start_idx * 60, end_time=end_idx * 60,
        start_bar_index=start_idx, end_bar_index=end_idx,
        start_price=start_price, end_price=end_price,
        line_role=role, direction="up", touch_count=2,
        label_source="manual", created_at=0,
    )


def test_cost_zero_for_identical_lines():
    a = _r("a"); b = _r("b")
    cost = _line_cost(a, b)
    assert cost < 1e-6


def test_cost_infinite_for_different_pair():
    a = _r("a", symbol="BTC")
    b = _r("b", symbol="ETH")
    assert math.isinf(_line_cost(a, b))


def test_cost_role_mismatch_dominates():
    a = _r("a", role="support")
    b = _r("b", role="resistance")
    cost = _line_cost(a, b)
    assert cost >= 5.0   # ROLE_PEN floor


def test_greedy_match_picks_best_per_manual():
    manual = [_r("m1", start_idx=10, end_idx=50, end_price=102)]
    autos = [
        _r("a-good", start_idx=11, end_idx=51, end_price=102.05),
        _r("a-bad", start_idx=200, end_idx=300, end_price=80),
    ]
    matches = greedy_match(autos, manual, top_k=2)
    assert len(matches) == 1
    assert matches[0]["best_auto_id"] == "a-good"
    assert matches[0]["role_match"] is True
    assert matches[0]["anchor_err_bars"] == 1


def test_greedy_match_no_candidates_in_pair():
    manual = [_r("m1", symbol="BTC")]
    autos = [_r("a-eth", symbol="ETH")]   # different symbol, ignored
    matches = greedy_match(autos, manual, top_k=2)
    assert len(matches) == 1
    assert matches[0]["best_auto_id"] is None
    assert matches[0].get("no_candidates_in_pair") is True


def test_summarise_basic_metrics():
    manual = [_r("m1"), _r("m2", role="resistance"), _r("m3", symbol="ETH")]
    autos = [
        _r("a1", start_idx=11, end_idx=51),
        _r("a2", role="resistance", start_idx=12, end_idx=52),
    ]
    matches = greedy_match(autos, manual, top_k=2)
    s = summarise_matches(matches)
    assert s["n_manual"] == 3
    assert s["n_no_candidates_in_pair"] == 1   # m3 (ETH) has no auto
    # Two pairs matched → role_match rate calculated on those
    assert s["role_match_rate"] == 1.0


def test_summarise_handles_empty():
    s = summarise_matches([])
    assert s["n_manual"] == 0
