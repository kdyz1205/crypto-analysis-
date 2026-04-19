import pytest

from server.conditionals.watcher import _compute_trade_prices, _manual_slope_per_bar
from server.conditionals.types import (
    ConditionalOrder,
    OrderConfig,
    TriggerConfig,
)


def _order(**overrides):
    base = dict(
        conditional_id="cond_test",
        manual_line_id="line_test",
        symbol="TESTUSDT",
        timeframe="5m",
        side="support",
        t_start=1_000,
        t_end=1_100,
        price_start=10.0,
        price_end=20.0,
        pattern_stats_at_create={},
        trigger=TriggerConfig(),
        order=OrderConfig(
            direction="long",
            tolerance_pct_of_line=0.1,
            stop_offset_pct_of_line=0.0,
        ),
        status="pending",
        created_at=1_000,
        updated_at=1_000,
    )
    base.update(overrides)
    return ConditionalOrder(**base)


def test_line_price_extends_right_by_default():
    cond = _order()
    assert cond.line_price_at(1_200) == 30.0


def test_line_price_can_clamp_right_when_disabled():
    cond = _order(extend_right=False)
    assert cond.line_price_at(1_200) == 20.0


def test_line_price_extends_left_only_when_enabled():
    assert _order().line_price_at(900) == 10.0
    assert _order(extend_left=True).line_price_at(900) == 0.0


def test_trade_prices_can_place_stop_beyond_line_for_long():
    cond = _order(order=OrderConfig(
        direction="long",
        tolerance_pct_of_line=0.1,
        stop_offset_pct_of_line=0.3,
        rr_target=8.0,
    ))

    entry, stop, tp = _compute_trade_prices(cond, 100.0, 0.0)

    assert entry == pytest.approx(100.1)
    assert stop == pytest.approx(99.7)
    assert tp == pytest.approx(103.3)


def test_trade_prices_can_place_stop_beyond_line_for_short():
    cond = _order(order=OrderConfig(
        direction="short",
        tolerance_pct_of_line=0.1,
        stop_offset_pct_of_line=0.3,
        rr_target=8.0,
    ))

    entry, stop, tp = _compute_trade_prices(cond, 100.0, 0.0)

    assert entry == pytest.approx(99.9)
    assert stop == pytest.approx(100.3)
    assert tp == pytest.approx(96.7)


def test_manual_slope_converts_anchor_slope_to_timeframe_bars():
    cond = _order(
        timeframe="5m",
        t_start=1_000,
        t_end=1_600,
        price_start=10.0,
        price_end=16.0,
    )

    assert _manual_slope_per_bar(cond) == pytest.approx(3.0)
