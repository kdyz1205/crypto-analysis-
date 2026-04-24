"""Schema sanity: accept valid, reject invalid."""
import pytest
from trendline_tokenizer.schemas.trendline import TrendlineRecord


def _base(**overrides):
    data = dict(
        id="test-1", symbol="BTCUSDT", timeframe="4h",
        start_time=1_770_000_000, end_time=1_770_086_400,
        start_bar_index=0, end_bar_index=6,
        start_price=65_000.0, end_price=67_200.0,
        line_role="support", direction="up",
        touch_count=3, label_source="manual", created_at=1_770_000_000,
    )
    data.update(overrides)
    return data


def test_valid_record_builds():
    r = TrendlineRecord(**_base())
    assert r.duration_bars() == 6
    assert r.log_slope_per_bar() > 0


def test_end_time_before_start_fails():
    with pytest.raises(Exception):
        TrendlineRecord(**_base(end_time=1_760_000_000))


def test_end_bar_before_start_fails():
    with pytest.raises(Exception):
        TrendlineRecord(**_base(end_bar_index=-1))


def test_bad_role_fails():
    with pytest.raises(Exception):
        TrendlineRecord(**_base(line_role="nonsense"))


def test_negative_touch_count_fails():
    with pytest.raises(Exception):
        TrendlineRecord(**_base(touch_count=-1))


def test_slope_direction_agree():
    up = TrendlineRecord(**_base(start_price=100.0, end_price=110.0, direction="up"))
    dn = TrendlineRecord(**_base(id="t2", start_price=110.0, end_price=100.0, direction="down"))
    assert up.log_slope_per_bar() > 0
    assert dn.log_slope_per_bar() < 0
