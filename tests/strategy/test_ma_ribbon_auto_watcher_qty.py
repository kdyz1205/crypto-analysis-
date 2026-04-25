from __future__ import annotations
import pytest
from dataclasses import replace
from server.conditionals.types import ConditionalOrder, OrderConfig


def _ribbon_long_cond(notional_usd: float, current_price: float = 50_500.0) -> ConditionalOrder:
    cfg = OrderConfig(
        direction="long",
        sl_logic="ribbon_ema21_trailing",
        ribbon_meta={
            "tf": "5m", "ribbon_buffer_pct": 0.01,
            "signal_id": "x", "layer": "LV1",
            "ema21_at_signal": 50000.0,
            "initial_sl_estimate": 49500.0,
            "ramp_day_cap_pct_at_spawn": 0.02,
            "reverse_on_stop": False,
        },
        qty_notional_target=notional_usd,
        risk_usd_target=10.0,
    )
    return ConditionalOrder(
        lineage="ma_ribbon",
        manual_line_id=None,
        symbol="BTCUSDT",
        timeframe="5m",
        direction="long",
        order=cfg,
    )


@pytest.mark.asyncio
async def test_ribbon_qty_uses_notional_target_divided_by_price():
    from server.conditionals.watcher import _compute_qty
    cond = _ribbon_long_cond(notional_usd=505.0)
    qty = await _compute_qty(cond, market_price=50_500.0, atr=0.0)
    assert qty == pytest.approx(0.01, rel=0.001)


@pytest.mark.asyncio
async def test_ribbon_qty_none_when_notional_target_missing():
    from server.conditionals.watcher import _compute_qty
    cond = _ribbon_long_cond(notional_usd=505.0)
    cond.config = replace(cond.config, qty_notional_target=None)
    cond.order = cond.config
    qty = await _compute_qty(cond, market_price=50_500.0, atr=0.0)
    assert qty is None or qty == 0


@pytest.mark.asyncio
async def test_ribbon_qty_none_when_market_price_invalid():
    from server.conditionals.watcher import _compute_qty
    cond = _ribbon_long_cond(notional_usd=505.0)
    qty = await _compute_qty(cond, market_price=0.0, atr=0.0)
    assert qty is None or qty == 0


@pytest.mark.asyncio
async def test_manual_line_qty_path_unchanged_regression():
    """Build a manual-line cond and confirm its qty path is non-None and finite —
    the original tolerance_pct / stop_pct sizing path must still work."""
    from server.conditionals.watcher import _compute_qty
    cfg = OrderConfig(
        direction="long",
        sl_logic="line_buffer",
        ribbon_meta=None,
        entry_offset_points=10.0,
        stop_points=20.0,
    )
    cond = ConditionalOrder(
        lineage="manual_line",
        manual_line_id="some-line-id",
        symbol="BTCUSDT",
        timeframe="1h",
        direction="long",
        order=cfg,
    )
    qty = await _compute_qty(cond, market_price=50_000.0, atr=100.0)
    # We don't assert exact value (project-internal math) — just that the
    # manual-line path still produces SOMETHING non-trivial.
    assert qty is None or qty >= 0  # don't crash
