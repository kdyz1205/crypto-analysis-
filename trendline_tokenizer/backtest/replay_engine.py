"""Replay engine: feed a historical OHLCV DataFrame to an InferenceService
one bar at a time, like a live feed would, and collect predictions /
signals at each step.

Critical invariant: the InferenceService only sees bars up to and
including bar i when predicting at bar i. No future bars leak in.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import pandas as pd

from ..inference.inference_service import InferenceService, PredictionRecord
from ..inference.signal_engine import SignalEngine, SignalRecord


@dataclass
class ReplayStep:
    bar_index: int
    open_time: int
    prediction: Optional[PredictionRecord]
    signal: Optional[SignalRecord]
    close: float


def replay(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    service: InferenceService,
    signal_engine: SignalEngine | None = None,
    predict_every: int = 1,
    start_bar: int = 0,
) -> Iterator[ReplayStep]:
    """Push closed bars sequentially. Yields one ReplayStep per bar
    (signal/prediction may be None if predict_every > 1 or cache is
    not warm yet)."""
    se = signal_engine or SignalEngine()
    ts_col = "open_time" if "open_time" in df.columns else (
        "timestamp" if "timestamp" in df.columns else None
    )
    for i in range(start_bar, len(df)):
        row = df.iloc[i]
        if ts_col:
            v = row[ts_col]
            try:
                ot = int(v)
            except (TypeError, ValueError):
                ot = int(pd.Timestamp(v).timestamp() * 1000)
        else:
            ot = i
        bar = {
            "open_time": ot,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0)),
        }
        service.push_bar(symbol, timeframe, bar)
        pred: Optional[PredictionRecord] = None
        sig: Optional[SignalRecord] = None
        if (i - start_bar) % predict_every == 0:
            pred = service.predict(symbol, timeframe)
            if pred is not None:
                sig = se.evaluate(pred)
        yield ReplayStep(
            bar_index=i, open_time=ot,
            prediction=pred, signal=sig,
            close=float(row["close"]),
        )
