from __future__ import annotations

from typing import Any

import pandas as pd


def build_analysis_history(
    market_payload: dict[str, Any] | None,
    candles_df: pd.DataFrame,
) -> dict[str, Any]:
    payload = dict(market_payload or {})
    loaded_bar_count = int(payload.get("loadedBarCount") or len(candles_df))
    analysis_bar_count = int(len(candles_df))
    timestamps = candles_df["timestamp"].tolist() if not candles_df.empty and "timestamp" in candles_df.columns else []
    payload.update(
        {
            "analysisInputBarCount": analysis_bar_count,
            "analysisEarliestTimestamp": int(timestamps[0]) if timestamps else None,
            "analysisLatestTimestamp": int(timestamps[-1]) if timestamps else None,
            "analysisWasTrimmed": loaded_bar_count > analysis_bar_count,
        }
    )
    return payload


__all__ = ["build_analysis_history"]
