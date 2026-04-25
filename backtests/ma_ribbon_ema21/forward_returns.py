"""Forward-return computation. Strict: never reach beyond data length."""
from __future__ import annotations
import math
import pandas as pd


def forward_return(close: pd.Series, entry_idx: int, n_bars: int) -> float:
    """Return (close[entry_idx + n_bars] - close[entry_idx]) / close[entry_idx].
    NaN if entry_idx + n_bars exceeds len(close) - 1.
    """
    if n_bars < 1:
        raise ValueError(f"forward_return n_bars must be >= 1, got {n_bars}")
    target = entry_idx + n_bars
    if target >= len(close):
        return math.nan
    entry  = float(close.iloc[entry_idx])
    later  = float(close.iloc[target])
    if entry <= 0 or math.isnan(entry) or math.isnan(later):
        return math.nan
    return (later - entry) / entry


def forward_returns_at(
    close: pd.Series,
    entry_idx: int,
    horizons: list[int],
) -> dict[int, float]:
    """Returns dict horizon → forward return."""
    return {h: forward_return(close, entry_idx, h) for h in horizons}


def apply_round_trip_cost(
    raw_return: float,
    fee_per_side: float = 0.0005,
    slippage_per_fill: float = 0.0001,
) -> float:
    """Subtract a long round-trip cost: 2 fills * (fee + slippage). Long-only."""
    if math.isnan(raw_return):
        return math.nan
    cost_per_fill = fee_per_side + slippage_per_fill
    return raw_return - 2.0 * cost_per_fill
