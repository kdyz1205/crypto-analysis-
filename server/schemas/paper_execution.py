from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OrderIntentModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_intent_id: str
    signal_id: str
    line_id: str
    client_order_id: str
    symbol: str
    timeframe: str
    side: str
    order_type: str
    trigger_mode: str
    entry_price: float
    stop_price: float
    tp_price: float
    quantity: float
    status: str
    reason: str = ""
    created_at_bar: int
    created_at_ts: object | None = None


class PaperOrderModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    line_id: str
    client_order_id: str
    signal_id: str
    symbol: str
    timeframe: str
    side: str
    order_type: str
    trigger_mode: str
    price: float
    quantity: float
    filled_quantity: float
    avg_fill_price: float
    status: str
    reason: str = ""
    created_at_bar: int
    updated_at_bar: int


class PaperFillModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    fill_id: str
    order_id: str
    client_order_id: str
    signal_id: str
    line_id: str
    symbol: str
    timeframe: str
    side: str
    fill_price: float
    quantity: float
    filled_at_bar: int
    filled_at_ts: object | None = None
    fill_reason: str


class PaperPositionModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position_id: str
    signal_id: str
    line_id: str
    client_order_id: str
    symbol: str
    timeframe: str
    direction: str
    quantity: float
    entry_price: float
    mark_price: float
    stop_price: float
    tp_price: float
    status: str
    opened_at_bar: int
    closed_at_bar: int | None = None
    opened_at_ts: object | None = None
    closed_at_ts: object | None = None
    realized_pnl: float
    unrealized_pnl: float
    exit_price: float | None = None
    exit_reason: str | None = None


class KillSwitchStateModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    blocked: bool
    manual_blocked: bool
    data_blocked: bool
    risk_blocked: bool
    reason: str = ""


class PaperAccountSummaryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    starting_equity: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    daily_realized_pnl: float
    consecutive_losses: int
    total_exposure: float
    open_order_count: int
    open_position_count: int
    closed_trade_count: int
    last_processed_bar_by_stream: dict[str, int] = Field(default_factory=dict)


class PaperExecutionConfigModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    risk_per_trade: float
    max_concurrent_positions: int
    max_positions_per_symbol: int
    max_total_exposure: float
    max_daily_loss: float
    max_consecutive_losses: int
    cancel_after_bars: int
    cooldown_bars_after_loss: int
    starting_equity: float
    allow_multiple_same_direction_per_symbol: bool


class PaperExecutionStateModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config: PaperExecutionConfigModel
    account: PaperAccountSummaryModel
    kill_switch: KillSwitchStateModel
    intents: list[OrderIntentModel] = Field(default_factory=list)
    open_orders: list[PaperOrderModel] = Field(default_factory=list)
    open_positions: list[PaperPositionModel] = Field(default_factory=list)
    recent_fills: list[PaperFillModel] = Field(default_factory=list)
    recent_closed_positions: list[PaperPositionModel] = Field(default_factory=list)
    cooldowns: dict[str, int] = Field(default_factory=dict)


class PaperExecutionStateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: PaperExecutionStateModel


class PaperExecutionConfigResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config: PaperExecutionConfigModel


class PaperExecutionConfigUpdateRequest(BaseModel):
    risk_per_trade: float | None = None
    max_concurrent_positions: int | None = None
    max_positions_per_symbol: int | None = None
    max_total_exposure: float | None = None
    max_daily_loss: float | None = None
    max_consecutive_losses: int | None = None
    cancel_after_bars: int | None = None
    cooldown_bars_after_loss: int | None = None
    starting_equity: float | None = None
    allow_multiple_same_direction_per_symbol: bool | None = None


class PaperExecutionStepRequest(BaseModel):
    symbol: str
    interval: str = "4h"
    bar_index: int | None = None
    days: int | None = None
    analysis_bars: int = 500
    trigger_modes: list[str] = Field(default_factory=lambda: ["pre_limit"])
    lookback_bars: int | None = 80
    min_touches: int | None = None
    confirm_threshold: float | None = None
    score_threshold: float | None = None
    rr_target: float | None = None
    strategy_window_bars: int | None = 100


class PaperExecutionStepResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stream: str
    processedBars: list[int] = Field(default_factory=list)
    lastProcessedBar: int
    state: PaperExecutionStateModel


class PaperExecutionResetRequest(BaseModel):
    starting_equity: float | None = None


class PaperKillSwitchRequest(BaseModel):
    blocked: bool
    reason: str = ""


__all__ = [
    "PaperAccountSummaryModel",
    "PaperExecutionConfigModel",
    "PaperExecutionConfigResponse",
    "PaperExecutionConfigUpdateRequest",
    "PaperExecutionResetRequest",
    "PaperExecutionStateModel",
    "PaperExecutionStateResponse",
    "PaperExecutionStepRequest",
    "PaperExecutionStepResponse",
    "PaperFillModel",
    "PaperKillSwitchRequest",
    "PaperOrderModel",
    "PaperPositionModel",
]
