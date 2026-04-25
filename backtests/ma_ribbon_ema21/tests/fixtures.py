"""Synthetic and real-data fixtures for Phase 1 tests."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


_TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}


def make_flat_ohlcv(
    n_bars: int = 100,
    base_price: float = 100.0,
    tf: str = "1h",
    start_ts: int = 1_700_000_000,
) -> pd.DataFrame:
    """A flat-price OHLCV with no alignment events. Useful as a negative control."""
    step = _TF_SECONDS[tf]
    ts = np.arange(n_bars) * step + start_ts
    return pd.DataFrame({
        "timestamp": ts.astype(np.int64),
        "open":   np.full(n_bars, base_price, dtype=float),
        "high":   np.full(n_bars, base_price, dtype=float),
        "low":    np.full(n_bars, base_price, dtype=float),
        "close":  np.full(n_bars, base_price, dtype=float),
        "volume": np.full(n_bars, 1.0, dtype=float),
    })


def make_uptrend_with_formation(
    n_bars: int = 200,
    formation_at_bar: int = 100,
    base_price: float = 100.0,
    tf: str = "1h",
    start_ts: int = 1_700_000_000,
    pre_drift: float = -0.001,
    post_drift: float = +0.005,
    noise_pct: float = 0.0005,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """OHLCV where bullish ribbon forms near `formation_at_bar`.

    Returns (df, formation_bar_idx). `formation_bar_idx` is the SUGGESTED
    formation bar (the regime change). The actual first bar where
    close > MA5 > MA8 > EMA21 > MA55 may shift by a few bars depending on
    the noise and MA periods, but it is guaranteed to be near it given
    enough post-drift bars.
    """
    rng = np.random.default_rng(seed)
    closes = np.empty(n_bars, dtype=float)
    closes[0] = base_price
    for i in range(1, n_bars):
        drift = pre_drift if i < formation_at_bar else post_drift
        noise = rng.normal(0.0, noise_pct)
        closes[i] = closes[i-1] * (1.0 + drift + noise)

    step = _TF_SECONDS[tf]
    ts = np.arange(n_bars) * step + start_ts
    df = pd.DataFrame({
        "timestamp": ts.astype(np.int64),
        "open":   closes,
        "high":   closes * (1.0 + 0.0005),
        "low":    closes * (1.0 - 0.0005),
        "close":  closes,
        "volume": np.full(n_bars, 1.0),
    })
    return df, formation_at_bar


def make_real_csv_path(symbol: str, tf: str) -> str:
    """Returns the canonical path to a small real-data fixture CSV.
    The CSV may not be present on first run — the function only returns the path.
    """
    here = Path(__file__).resolve().parent
    return str(here / "data" / f"{symbol}_{tf}.csv")
