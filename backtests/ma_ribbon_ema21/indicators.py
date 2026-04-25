"""Pure-function indicators: SMA, EMA. NaN-prefixed to match incomplete windows."""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average. Returns a Series with NaN for the first `period - 1` bars."""
    if period < 1:
        raise ValueError(f"sma period must be >= 1, got {period}")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. Seeded with SMA at index `period - 1`. NaN prefix."""
    if period < 1:
        raise ValueError(f"ema period must be >= 1, got {period}")
    n = len(series)
    out = pd.Series(np.full(n, np.nan, dtype=float), index=series.index)
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    seed = float(series.iloc[:period].mean())
    out.iloc[period - 1] = seed
    prev = seed
    values = series.to_numpy(dtype=float)
    for i in range(period, n):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out.iloc[i] = prev
    return out
