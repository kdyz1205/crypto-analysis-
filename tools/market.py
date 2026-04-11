"""Tool 1: Market Data — fetch OHLCV for any symbol/timeframe."""

from __future__ import annotations
import asyncio
import pandas as pd
from .types import AuditEntry


async def get_ohlcv(symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
    """Fetch OHLCV data. Returns standardized DataFrame."""
    from server.data_service import get_ohlcv_with_df
    df_polars, _ = await get_ohlcv_with_df(symbol, timeframe, None, days, history_mode="fast_window")
    if df_polars is None or df_polars.is_empty():
        return pd.DataFrame()
    pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(pd.Timestamp(v).timestamp()))
    for c in ("open", "high", "low", "close", "volume"):
        pdf[c] = pd.to_numeric(pdf[c], errors="raise")
    return pdf


def get_ohlcv_sync(symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
    """Synchronous wrapper."""
    return asyncio.run(get_ohlcv(symbol, timeframe, days))
