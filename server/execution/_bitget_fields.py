"""Bitget response field helpers.

Bitget's v2 futures API is inconsistent about naming: position endpoints and
history-position endpoints use different keys for the same concept, and the
older `averageOpenPrice` form also still shows up. Using the wrong key
silently returns 0.0 (not an error), which is how the 2026-04-18 -81% DD
incident went undetected — every trade_log row had `entry_price=0.0`.

This module centralizes the "try these keys in order" logic so the set of
known aliases is in ONE place. When Bitget changes a field name in the
future, updating here fixes every call site.

Primary reference: `server/strategy/mar_bb_history.py` (canonical for
`history-position` endpoint) and `server/execution/live_adapter.py:712`
(canonical for `/position` endpoints).
"""
from __future__ import annotations

from typing import Any


def _first_float(row: dict[str, Any] | None, keys: tuple[str, ...]) -> float:
    """Return the first non-zero, numeric value found under any of `keys`.
    Returns 0.0 if row is falsy, or none of the keys have a valid value.
    Does NOT raise — callers depend on the 0.0 fallback, but pair this with
    the quarantine check in `trade_log.log_close` so zeroes surface loudly.
    """
    if not row:
        return 0.0
    for k in keys:
        v = row.get(k)
        if v in (None, ""):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != 0.0:
            return f
    return 0.0


def open_price(row: dict[str, Any] | None) -> float:
    """Open / entry price from any Bitget row — positions OR history-positions."""
    return _first_float(
        row,
        # Order matters: history-position canonical form first, then /position
        # forms, then legacy alias.
        ("openAvgPrice", "openPriceAvg", "averageOpenPrice", "entryPrice"),
    )


def close_price(row: dict[str, Any] | None) -> float:
    """Close / exit price. Only meaningful for history-position rows."""
    return _first_float(row, ("closeAvgPrice", "closePriceAvg"))


def mark_price(row: dict[str, Any] | None) -> float:
    """Latest mark price for an open position."""
    return _first_float(row, ("markPrice", "mark_price", "indexPrice"))


def position_size(row: dict[str, Any] | None) -> float:
    """Position quantity (contracts or base units, per symbol)."""
    return _first_float(row, ("total", "size", "available", "holdPos"))


def realized_pnl_usd(row: dict[str, Any] | None) -> float:
    """Realized PnL (net USD) for a closed Bitget position row. Respects sign.

    # SAFE: docstring only refers to the legacy name for historical context
    Named `_usd` (not plain realized_pnl) to match the Phase 1 convention
    that splits legacy `realized_pnl` into `_sim` (paper) and `_usd` (real
    exchange fill). Bitget responses are always real fills → always `_usd`.
    """
    if not row:
        return 0.0
    for k in ("netProfit", "achievedProfits", "net_pnl", "pnl", "realizedPnl"):
        v = row.get(k)
        if v in (None, ""):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def margin_used(row: dict[str, Any] | None) -> float:
    """Margin consumed by a position — used to compute pnl%.

    Falls back to computing from (openSize × openAvgPrice / leverage)
    when Bitget's history-position row omits the direct margin field.
    User 2026-04-21: 18 position-close events had pnl_pct=0.00%
    because Bitget's history rows don't always include 'margin' —
    only 'initialMargin' shows up for /position endpoints, not /history.
    """
    direct = _first_float(row, ("margin", "initialMargin", "marginSize", "openingMargin"))
    if direct > 0:
        return direct
    # Fallback: qty × open_price / leverage
    if not row:
        return 0.0
    qty = _first_float(row, ("openTotalPos", "openTotal", "total", "size"))
    op = open_price(row)
    lev = _first_float(row, ("leverage", "openLeverage", "openPositionMode"))
    if qty > 0 and op > 0 and lev > 0:
        return (qty * op) / lev
    return 0.0


def notional_usd(row: dict[str, Any] | None) -> float:
    """Total position size in USD = qty × open_price. Used as a pnl_pct
    denominator when margin_used comes back 0 (pnl_pct then = raw price
    move %, unlevered). Complements margin_used by providing a guaranteed
    non-zero fallback for pnl% display."""
    qty = _first_float(row, ("openTotalPos", "openTotal", "total", "size"))
    op = open_price(row)
    if qty > 0 and op > 0:
        return qty * op
    return 0.0


__all__ = [
    "open_price",
    "close_price",
    "mark_price",
    "position_size",
    "realized_pnl_usd",
    "margin_used",
    "notional_usd",
]
