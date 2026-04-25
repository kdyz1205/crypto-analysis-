"""Task 5 of MA-ribbon EMA21 auto-execution: trailing-SL helper for
ribbon_ema21_trailing.

Validates `_compute_ribbon_trailing_sl(cond, current_sl=...)` in
`server/conditionals/watcher.py`. Helper rules:

  long  → sl never loosens: sl = max(current_sl, ema21 * (1 - buffer))
  short → sl never raises:  sl = min(current_sl, ema21 * (1 + buffer))

If EMA21 is None / NaN / non-positive, return current_sl unchanged. If
the cond is not a ribbon conditional or has invalid config, raise.

NOTE: `OrderConfig` is `frozen=True`. Tests that need to flip direction
or sl_logic mid-fixture must use `dataclasses.replace` to swap the
config in. The mutable `cond.direction` attribute on the outer
ConditionalOrder dataclass is the helper's source of truth for
direction (mirrors `cond.order.direction`).
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
from unittest.mock import patch, AsyncMock

from server.conditionals.types import ConditionalOrder, OrderConfig
from server.conditionals.watcher import _compute_ribbon_trailing_sl


def _bull_cond(symbol: str = "BTCUSDT", tf: str = "5m",
               buffer: float = 0.01,
               current_sl: float = 49000.0) -> ConditionalOrder:
    """Build a long ribbon ConditionalOrder fixture.

    `current_sl` is mirrored into `ribbon_meta["initial_sl_estimate"]` so the
    test for the `current_sl=None` fallback path resolves to a known number.
    Tests pass `current_sl` explicitly to the helper, so the meta value only
    matters in the fallback case.
    """
    return ConditionalOrder(
        lineage="ma_ribbon",
        manual_line_id=None,
        symbol=symbol,
        timeframe=tf,
        direction="long",
        config=OrderConfig(
            direction="long",
            sl_logic="ribbon_ema21_trailing",
            ribbon_meta={
                "tf": tf,
                "ribbon_buffer_pct": buffer,
                "signal_id": "x",
                "layer": "LV1",
                "ema21_at_signal": 50000.0,
                "initial_sl_estimate": current_sl,
                "ramp_day_cap_pct_at_spawn": 0.02,
                "reverse_on_stop": False,
            },
        ),
    )


def _flip_to_short(cond: ConditionalOrder) -> ConditionalOrder:
    """Helper: produce a short-direction copy of a long ribbon cond.

    `OrderConfig` is frozen, so we use `dataclasses.replace` to swap in a
    new config with direction=short, then re-sync the outer alias.
    """
    cond.direction = "short"
    cond.order = replace(cond.order, direction="short")
    cond.config = cond.order
    return cond


# ─────────────────────────────────────────────────────────────────────
# Long-side ratchet
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_long_sl_ratchets_up_when_ema21_rises():
    cond = _bull_cond(current_sl=49500.0)
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=51000.0)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=49500.0)
    # candidate = 51000 * (1 - 0.01) = 50490; max(49500, 50490) = 50490
    assert new_sl == pytest.approx(50490.0)


@pytest.mark.asyncio
async def test_long_sl_does_not_loosen_when_ema21_falls():
    cond = _bull_cond(current_sl=49500.0)
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=49000.0)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=49500.0)
    # candidate = 49000 * 0.99 = 48510 < 49500 → keep current_sl
    assert new_sl == pytest.approx(49500.0)


# ─────────────────────────────────────────────────────────────────────
# Short-side ratchet (mirror)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_short_sl_ratchets_down_when_ema21_falls():
    cond = _flip_to_short(_bull_cond())
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=49000.0)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=51000.0)
    # candidate = 49000 * 1.01 = 49490 < 51000 → adopted
    assert new_sl == pytest.approx(49490.0)


@pytest.mark.asyncio
async def test_short_sl_does_not_loosen_when_ema21_rises():
    cond = _flip_to_short(_bull_cond())
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=53000.0)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=51000.0)
    # candidate = 53000 * 1.01 = 53530 > 51000 → keep current
    assert new_sl == pytest.approx(51000.0)


# ─────────────────────────────────────────────────────────────────────
# Buffer comes from ribbon_meta (per-layer)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_uses_buffer_from_ribbon_meta_per_layer():
    cond = _bull_cond(tf="4h", buffer=0.10)
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=50000.0)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=40000.0)
    # candidate = 50000 * (1 - 0.10) = 45000; max(40000, 45000) = 45000
    assert new_sl == pytest.approx(45000.0)


# ─────────────────────────────────────────────────────────────────────
# EMA21 fetch failure fallbacks (None / NaN / non-positive)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_current_sl_when_ema21_unavailable():
    cond = _bull_cond(current_sl=49500.0)
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=None)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=49500.0)
    assert new_sl == pytest.approx(49500.0)


@pytest.mark.asyncio
async def test_returns_current_sl_when_ema21_nan():
    cond = _bull_cond(current_sl=49500.0)
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=math.nan)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=49500.0)
    assert new_sl == pytest.approx(49500.0)


@pytest.mark.asyncio
async def test_returns_current_sl_when_ema21_non_positive():
    cond = _bull_cond(current_sl=49500.0)
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=-5.0)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=49500.0)
    assert new_sl == pytest.approx(49500.0)


# ─────────────────────────────────────────────────────────────────────
# Lineage / config gates — the helper must refuse non-ribbon callers
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_raises_on_non_ribbon_lineage():
    """A line_buffer cond must NOT call this helper. The caller in
    _sync_sl_to_line_now branches on lineage; we still defend the helper.
    """
    cond = _bull_cond(current_sl=49500.0)
    cond.lineage = "manual_line"
    # OrderConfig is frozen → swap a fresh one in via dataclasses.replace
    cond.order = replace(
        cond.order,
        sl_logic="line_buffer",
        ribbon_meta=None,
    )
    cond.config = cond.order
    with pytest.raises((AssertionError, ValueError)):
        await _compute_ribbon_trailing_sl(cond, current_sl=49500.0)


@pytest.mark.asyncio
async def test_raises_on_invalid_buffer():
    """Zero or negative ribbon_buffer_pct is a bug — refuse to compute."""
    cond = _bull_cond(current_sl=49500.0)
    # ribbon_meta is a dict (not a frozen field) so we can mutate it in place.
    cond.config.ribbon_meta["ribbon_buffer_pct"] = 0.0
    with pytest.raises(ValueError, match="ribbon_buffer_pct"):
        await _compute_ribbon_trailing_sl(cond, current_sl=49500.0)


@pytest.mark.asyncio
async def test_raises_on_unknown_direction():
    cond = _bull_cond(current_sl=49500.0)
    cond.direction = "sideways"  # invalid; helper reads cond.direction
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=50000.0)):
        with pytest.raises(ValueError, match="direction"):
            await _compute_ribbon_trailing_sl(cond, current_sl=49500.0)


# ─────────────────────────────────────────────────────────────────────
# current_sl=None fallback to ribbon_meta["initial_sl_estimate"]
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initial_sl_estimate_used_when_current_sl_none():
    """If caller passes current_sl=None, fall back to
    ribbon_meta["initial_sl_estimate"]."""
    cond = _bull_cond(current_sl=49500.0)  # also sets initial_sl_estimate=49500
    with patch("server.conditionals.watcher.fetch_current_ema21",
               new=AsyncMock(return_value=50000.0)):
        new_sl = await _compute_ribbon_trailing_sl(cond, current_sl=None)
    # initial_sl_estimate = 49500; candidate = 50000 * 0.99 = 49500;
    # max(49500, 49500) = 49500
    assert new_sl == pytest.approx(49500.0)
