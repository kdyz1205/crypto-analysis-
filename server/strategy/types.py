from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
LineSide = Literal["resistance", "support"]
PivotKind = Literal["high", "low"]
SignalDirection = Literal["long", "short"]
TriggerMode = Literal["pre_limit", "rejection", "failed_breakout"]


def ensure_candles_df(candles: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(candles, pd.DataFrame):
        df = candles.copy()
    else:
        df = pd.DataFrame(list(candles))

    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")

    df = df.reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    return df


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def project_price(slope: float, intercept: float, bar_index: int) -> float:
    return float((slope * bar_index) + intercept)


@dataclass(frozen=True, slots=True)
class Pivot:
    pivot_id: str
    kind: PivotKind
    index: int
    timestamp: Any
    price: float
    left_bars: int
    right_bars: int
    confirmed_at_index: int


@dataclass(frozen=True, slots=True)
class Trendline:
    line_id: str
    side: LineSide
    symbol: str
    timeframe: str
    state: Literal["candidate", "confirmed", "invalidated", "expired"]
    anchor_pivot_ids: tuple[str, str]
    confirming_touch_pivot_ids: tuple[str, ...]
    anchor_indices: tuple[int, int]
    anchor_prices: tuple[float, float]
    slope: float
    intercept: float
    confirming_touch_indices: tuple[int, ...]
    bar_touch_indices: tuple[int, ...]
    confirming_touch_count: int
    bar_touch_count: int
    recent_bar_touch_count: int
    residuals: tuple[float, ...]
    score: float
    score_components: Mapping[str, float] = field(default_factory=dict)
    projected_price_current: float = 0.0
    projected_price_next: float = 0.0
    latest_confirming_touch_index: int | None = None
    latest_confirming_touch_price: float | None = None
    bars_since_last_confirming_touch: int = 10**9
    recent_test_count: int = 0
    non_touch_cross_count: int = 0
    invalidation_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TrendlineDetectionResult:
    candidate_lines: tuple[Trendline, ...]
    active_lines: tuple[Trendline, ...]


@dataclass(frozen=True, slots=True)
class StrategySignal:
    signal_id: str
    line_id: str
    symbol: str
    timeframe: str
    signal_type: str
    direction: SignalDirection
    trigger_mode: TriggerMode
    timestamp: Any
    trigger_bar_index: int
    score: float
    priority_rank: int | None
    entry_price: float
    stop_price: float
    tp_price: float
    risk_reward: float
    confirming_touch_count: int
    bars_since_last_confirming_touch: int
    distance_to_line: float
    line_side: LineSide
    reason_code: str
    factor_components: Mapping[str, float] = field(default_factory=dict)


__all__ = [
    "LineSide",
    "OHLCV_COLUMNS",
    "Pivot",
    "PivotKind",
    "SignalDirection",
    "StrategySignal",
    "Trendline",
    "TrendlineDetectionResult",
    "TriggerMode",
    "ensure_candles_df",
    "project_price",
    "stable_id",
]
