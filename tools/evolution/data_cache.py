"""Cached OHLCV loader so backtests don't re-fetch from API every run.

Determinism: once a (symbol, tf, days) slice is cached, it never changes.
Evolution rounds over the same cached data → reproducible fitness.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import pandas as pd


CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "evolution" / "candles_cache"


def _cache_path(symbol: str, timeframe: str, days: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{timeframe}_{days}d.pkl"


def load_candles(symbol: str, timeframe: str, days: int = 730) -> pd.DataFrame | None:
    """Return cached candles if present, else fetch + cache + return."""
    p = _cache_path(symbol, timeframe, days)
    if p.exists():
        try:
            with p.open("rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return df
        except Exception:
            pass

    try:
        from tools.market import get_ohlcv_sync
        df = get_ohlcv_sync(symbol, timeframe, days)
    except Exception as e:
        print(f"[data_cache] fetch failed {symbol} {timeframe}: {e}")
        return None

    if df is None or df.empty:
        return None

    try:
        with p.open("wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass

    return df


def split_train_test(df: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * train_fraction)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test
