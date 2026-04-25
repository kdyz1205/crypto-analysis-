"""Multi-timeframe feature cache: a ring buffer per (symbol, timeframe)
that exposes the same scale-stable price-feature window the training
dataset uses.

CRITICAL: do NOT update higher-TF caches mid-bar. The cache only sees
a candle once it has closed (open_time stable, close finalised).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..training.sequence_dataset import _price_window


@dataclass
class _SymbolTFBuffer:
    symbol: str
    timeframe: str
    capacity: int
    bars: deque = field(default_factory=deque)
    last_close_time: Optional[int] = None

    def add_closed_bar(self, bar: dict) -> bool:
        """Append a CLOSED bar. Idempotent on (open_time). Returns True
        if the bar was new."""
        ot = int(bar["open_time"]) if not isinstance(bar["open_time"], pd.Timestamp) \
             else int(bar["open_time"].timestamp() * 1000)
        if self.last_close_time is not None and ot <= self.last_close_time:
            return False
        self.bars.append({
            "open_time": ot,
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar["volume"]),
        })
        if len(self.bars) > self.capacity:
            self.bars.popleft()
        self.last_close_time = ot
        return True

    def to_df(self) -> pd.DataFrame:
        if not self.bars:
            return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        return pd.DataFrame(list(self.bars))


class FeatureCache:
    """Per-(symbol, tf) ring buffer of recent closed bars + an extractor
    that produces the price feature window for the inference model."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._buffers: dict[tuple[str, str], _SymbolTFBuffer] = {}

    def _buf(self, symbol: str, tf: str) -> _SymbolTFBuffer:
        key = (symbol.upper(), tf)
        if key not in self._buffers:
            self._buffers[key] = _SymbolTFBuffer(symbol.upper(), tf, self.capacity)
        return self._buffers[key]

    def push(self, symbol: str, tf: str, bar: dict) -> bool:
        return self._buf(symbol, tf).add_closed_bar(bar)

    def bars_df(self, symbol: str, tf: str) -> pd.DataFrame:
        return self._buf(symbol, tf).to_df()

    def n_bars(self, symbol: str, tf: str) -> int:
        return len(self._buf(symbol, tf).bars)

    def price_window(self, symbol: str, tf: str, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (window, pad_mask) of shape ((length, 13), (length,)).
        Uses the SAME _price_window helper as training, so train/serve
        feature space is identical by construction."""
        df = self.bars_df(symbol, tf)
        end_idx = len(df)
        return _price_window(df, end_idx, length)
