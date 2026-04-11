"""Tests for multi-timeframe confluence scoring."""
from server.strategy.confluence import confluence_score_for_line, confluence_score_for_zone
from server.strategy.types import Trendline
from server.strategy.zones import HorizontalZone


def _line(side="resistance", price=100.0, timeframe="1h") -> Trendline:
    return Trendline(
        line_id="test", side=side, symbol="TEST", timeframe=timeframe,
        state="confirmed", anchor_pivot_ids=("a", "b"),
        confirming_touch_pivot_ids=("a", "b"),
        anchor_indices=(0, 10), anchor_prices=(price, price),
        slope=0.0, intercept=price,
        confirming_touch_indices=(0, 10), bar_touch_indices=(),
        confirming_touch_count=2, bar_touch_count=0, recent_bar_touch_count=0,
        residuals=(0.1, 0.1), score=70.0, score_components={},
        projected_price_current=price, projected_price_next=price,
        latest_confirming_touch_index=10, latest_confirming_touch_price=price,
        bars_since_last_confirming_touch=5, recent_test_count=0, non_touch_cross_count=0,
    )


_zone_counter = 0

def _zone(side="resistance", center=100.0, width=1.0) -> HorizontalZone:
    global _zone_counter
    _zone_counter += 1
    return HorizontalZone(
        zone_id=f"z{_zone_counter}", side=side,
        price_low=center - width / 2, price_high=center + width / 2,
        price_center=center, width=width,
        touches=3, touch_indices=(5, 15, 25), touch_prices=(center,) * 3,
        first_touch_index=5, last_touch_index=25,
        strength=70.0, strength_components={},
    )


def test_no_other_tf_data_returns_zero():
    score = confluence_score_for_line(_line(), [], [])
    assert score == 0.0


def test_matching_zone_in_other_tf_gives_positive():
    line = _line(side="resistance", price=100.0, timeframe="1h")
    other_zone = _zone(side="resistance", center=100.0)  # overlapping
    score = confluence_score_for_line(line, [other_zone], [])
    assert score >= 0.5


def test_non_matching_zone_gives_zero():
    line = _line(side="resistance", price=100.0, timeframe="1h")
    other_zone = _zone(side="resistance", center=200.0)  # far away
    score = confluence_score_for_line(line, [other_zone], [])
    assert score == 0.0


def test_opposite_side_ignored():
    line = _line(side="resistance", price=100.0)
    other_zone = _zone(side="support", center=100.0)  # same price but support
    score = confluence_score_for_line(line, [other_zone], [])
    assert score == 0.0


def test_multiple_matches_score_higher():
    line = _line(side="resistance", price=100.0, timeframe="1h")
    z1 = _zone(side="resistance", center=100.0)
    z2 = _zone(side="resistance", center=100.2)  # close enough
    score = confluence_score_for_line(line, [z1, z2], [])
    assert score >= 0.8


def test_matching_line_in_other_tf():
    primary = _line(side="support", price=50.0, timeframe="1h")
    other = _line(side="support", price=50.0, timeframe="4h")
    score = confluence_score_for_line(primary, [], [other])
    assert score >= 0.5


def test_same_tf_line_ignored():
    primary = _line(side="support", price=50.0, timeframe="1h")
    same_tf = _line(side="support", price=50.0, timeframe="1h")
    score = confluence_score_for_line(primary, [], [same_tf])
    assert score == 0.0


def test_zone_confluence_overlapping():
    zone = _zone(side="support", center=50.0, width=1.0)
    other = _zone(side="support", center=50.3, width=1.0)  # overlapping
    score = confluence_score_for_zone(zone, [other], [])
    assert score >= 0.5


def test_zone_confluence_no_overlap():
    zone = _zone(side="support", center=50.0, width=1.0)
    other = _zone(side="support", center=60.0, width=1.0)  # far
    score = confluence_score_for_zone(zone, [other], [])
    assert score == 0.0
