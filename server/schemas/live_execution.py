from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LiveExecutionWhitelistModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbols: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    trigger_modes: list[str] = Field(default_factory=list)


class LiveExecutionLimitsModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_live_positions: int
    max_live_notional: float
    reconciliation_max_age_seconds: int | None = None


class LiveExecutionReconciliationModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool | None = None
    blocked: bool | None = None
    mode: str | None = None
    reason: str = ""
    positions: list[dict] = Field(default_factory=list)
    pending_orders: list[dict] = Field(default_factory=list)
    account_accessible: bool | None = None
    total_equity: float | None = None
    usdt_available: float | None = None
    checked_at: int | None = None
    exchange_response_excerpt: dict | None = None


class LiveExecutionResultModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool
    blocked: bool | None = None
    reason: str = ""
    blocking_reasons: list[str] = Field(default_factory=list)
    mode: str
    order_intent_id: str | None = None
    signal_id: str | None = None
    symbol: str
    side: str | None = None
    timeframe: str | None = None
    trigger_mode: str | None = None
    would_submit_symbol: str | None = None
    would_submit_side: str | None = None
    would_submit_notional: float | None = None
    submitted_price: float | None = None
    submitted_notional: float | None = None
    exchange_order_id: str = ""
    exchange_response_excerpt: dict | None = None
    idempotent_replay: bool = False


class LiveExecutionStatusModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled_flags: dict[str, bool]
    default_mode: str
    api_key_ready: bool
    whitelist: LiveExecutionWhitelistModel
    limits: LiveExecutionLimitsModel
    reconciliation: dict[str, LiveExecutionReconciliationModel | None]
    reconciliation_required_by_mode: dict[str, bool] = Field(default_factory=dict)
    submitted_intent_ids_by_mode: dict[str, list[str]] = Field(default_factory=dict)
    last_preview_result: LiveExecutionResultModel | None = None
    last_submission_result: LiveExecutionResultModel | None = None
    blocked_reason: str = ""


class LiveExecutionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: LiveExecutionStatusModel


class LivePreviewRequest(BaseModel):
    order_intent_id: str | None = None
    signal_id: str | None = None
    mode: str = "demo"


class LivePreviewResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    result: LiveExecutionResultModel


class LiveSubmitRequest(BaseModel):
    order_intent_id: str
    mode: str = "demo"
    confirm: bool = False


class LiveSubmitResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    result: LiveExecutionResultModel


class LiveCloseRequest(BaseModel):
    symbol: str
    mode: str = "demo"
    confirm: bool = False


class LiveCloseResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    result: LiveExecutionResultModel


class LiveReconcileRequest(BaseModel):
    mode: str = "demo"


class LiveReconcileResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reconciliation: LiveExecutionReconciliationModel


__all__ = [
    "LiveCloseRequest",
    "LiveCloseResponse",
    "LiveExecutionResultModel",
    "LiveExecutionStatusResponse",
    "LivePreviewRequest",
    "LivePreviewResponse",
    "LiveReconcileRequest",
    "LiveReconcileResponse",
    "LiveSubmitRequest",
    "LiveSubmitResponse",
]
