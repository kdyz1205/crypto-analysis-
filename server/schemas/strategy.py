from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .history import HistoryCoverageModel
from ..strategy.config import StrategyConfig


class StrategyPivotModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pivot_id: str
    kind: str
    index: int
    timestamp: Any
    price: float
    left_bars: int
    right_bars: int
    confirmed_at_index: int


class StrategyTouchPointModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line_id: str
    timestamp: Any
    bar_index: int
    price: float
    touch_type: str
    residual: float
    is_confirming_touch: bool
    side: str
    display_visible: bool = True
    display_class: str = "confirming"


class StrategyLineModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line_id: str
    symbol: str
    timeframe: str
    side: str
    source: str = "auto"
    state: str
    t_start: Any
    t_end: Any
    price_start: float
    price_end: float
    slope: float
    intercept: float
    anchor_indices: list[int]
    anchor_prices: list[float]
    anchor_timestamps: list[Any]
    confirming_touch_indices: list[int]
    bar_touch_indices: list[int]
    touch_count: int
    confirming_touch_count: int
    bar_touch_count: int
    line_score: float
    score_components: dict[str, float] = Field(default_factory=dict)
    projected_price_current: float
    projected_price_next: float
    projected_time_current: Any
    projected_time_next: Any
    is_active: bool
    is_invalidated: bool
    invalidation_reason: str | None = None
    invalidation_bar_index: int | None = None
    invalidation_timestamp: Any | None = None
    display_rank: int | None = None
    display_class: str = "debug"
    line_usability_score: float = 0.0
    last_quality_touch_index: int | None = None
    collapsed_invalidation_count: int = 1


class StrategyLineStateModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line_id: str
    state: str
    previous_state: str | None = None
    side: str
    symbol: str
    timeframe: str
    line_score: float
    confirming_touch_count: int
    bar_touch_count: int
    projected_price_current: float
    projected_price_next: float
    latest_confirming_touch_index: int | None = None
    bars_since_last_confirming_touch: int
    invalidation_reason: str | None = None
    transition_reason: str
    factor_score: float | None = None
    arm_distance: float | None = None
    distance_to_next_projection: float | None = None
    signal_ids: list[str] = Field(default_factory=list)
    invalidation_bar_index: int | None = None
    invalidation_timestamp: Any | None = None
    display_rank: int | None = None
    display_class: str = "debug"
    line_usability_score: float = 0.0
    collapsed_invalidation_count: int = 1


class StrategySignalModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    signal_id: str
    line_id: str
    symbol: str
    timeframe: str
    source: str = "auto"
    signal_type: str
    direction: str
    trigger_mode: str
    timestamp: Any
    trigger_bar_index: int
    score: float
    priority_rank: int | None = None
    entry_price: float
    stop_price: float
    tp_price: float
    risk_reward: float
    confirming_touch_count: int
    bars_since_last_confirming_touch: int
    distance_to_line: float
    line_side: str
    reason_code: str
    factor_components: dict[str, float] = Field(default_factory=dict)


class StrategySignalStateModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    signal_id: str
    line_id: str
    symbol: str
    timeframe: str
    state: str
    priority_rank: int | None = None
    trigger_mode: str
    direction: str
    score: float
    reason: str


class StrategySnapshotModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bar_index: int
    timestamp: Any
    pivots: list[StrategyPivotModel] = Field(default_factory=list)
    candidate_lines: list[StrategyLineModel] = Field(default_factory=list)
    active_lines: list[StrategyLineModel] = Field(default_factory=list)
    line_states: list[StrategyLineStateModel] = Field(default_factory=list)
    touch_points: list[StrategyTouchPointModel] = Field(default_factory=list)
    signals: list[StrategySignalModel] = Field(default_factory=list)
    signal_states: list[StrategySignalStateModel] = Field(default_factory=list)
    invalidations: list[StrategyLineStateModel] = Field(default_factory=list)
    horizontal_zones: list[dict[str, Any]] = Field(default_factory=list)
    market_regime: dict[str, Any] | None = None
    orders: list[dict[str, Any]] = Field(default_factory=list)


class StrategyConfigResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str | None = None
    interval: str | None = None
    pricePrecision: int | None = None
    tickSize: float
    config: dict[str, Any]
    layerDefaults: dict[str, bool]


class StrategySnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    interval: str
    barCount: int
    analysisBarCount: int
    pricePrecision: int | None = None
    tickSize: float
    config: dict[str, Any]
    history: HistoryCoverageModel | None = None
    snapshot: StrategySnapshotModel


class StrategyReplayResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    interval: str
    barCount: int
    analysisBarCount: int
    snapshotCount: int
    pricePrecision: int | None = None
    tickSize: float
    config: dict[str, Any]
    history: HistoryCoverageModel | None = None
    snapshots: list[StrategySnapshotModel]


def strategy_layer_defaults() -> dict[str, bool]:
    return {
        "primaryTrendlines": True,
        "debugTrendlines": False,
        "confirmingTouches": True,
        "barTouches": False,
        "projectedLine": True,
        "signalMarkers": True,
        "collapsedInvalidations": True,
        "orderMarkers": False,
    }


def serialize_config_response(
    config: StrategyConfig,
    *,
    symbol: str | None = None,
    interval: str | None = None,
    price_precision: int | None = None,
) -> StrategyConfigResponse:
    return StrategyConfigResponse(
        symbol=symbol,
        interval=interval,
        pricePrecision=price_precision,
        tickSize=float(config.tick_size),
        config=asdict(config),
        layerDefaults=strategy_layer_defaults(),
    )


__all__ = [
    "StrategyConfigResponse",
    "StrategyLineModel",
    "StrategyLineStateModel",
    "StrategyPivotModel",
    "StrategyReplayResponse",
    "StrategySignalModel",
    "StrategySignalStateModel",
    "StrategySnapshotModel",
    "StrategySnapshotResponse",
    "StrategyTouchPointModel",
    "serialize_config_response",
    "strategy_layer_defaults",
]
