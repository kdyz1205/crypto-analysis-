from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .paper_execution import PaperExecutionConfigModel, PaperExecutionStateModel


class RuntimeInstanceConfigModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    instance_id: str
    label: str
    symbol: str
    timeframe: str
    subaccount_label: str = ""
    history_mode: str
    analysis_bars: int
    days: int
    tick_interval_seconds: int
    auto_restart_on_boot: bool
    live_mode: str
    auto_live_preview: bool
    auto_live_submit: bool
    notes: str = ""
    paper_config: PaperExecutionConfigModel


class RuntimeInstanceStatusModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    runtime_state: str
    last_tick_at: str | None = None
    last_processed_bar: int | None = None
    last_error: str = ""
    last_runtime_note: str = ""
    last_live_result: dict | None = None
    paper_current_day: str | None = None
    paper_state: PaperExecutionStateModel | None = None
    live_engine_state: dict | None = None


class RuntimeInstanceModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config: RuntimeInstanceConfigModel
    status: RuntimeInstanceStatusModel


class RuntimeInstanceListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    instances: list[RuntimeInstanceModel] = Field(default_factory=list)


class RuntimeInstanceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    instance: RuntimeInstanceModel


class RuntimeInstanceCreateRequest(BaseModel):
    label: str
    symbol: str
    timeframe: str = "4h"
    subaccount_label: str = ""
    history_mode: str = "fast_window"
    analysis_bars: int = 500
    days: int = 365
    tick_interval_seconds: int = 60
    auto_restart_on_boot: bool = False
    live_mode: str = "disabled"
    auto_live_preview: bool = True
    auto_live_submit: bool = False
    notes: str = ""
    paper_config: PaperExecutionConfigModel | None = None


class RuntimeInstanceUpdateRequest(BaseModel):
    label: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    subaccount_label: str | None = None
    history_mode: str | None = None
    analysis_bars: int | None = None
    days: int | None = None
    tick_interval_seconds: int | None = None
    auto_restart_on_boot: bool | None = None
    live_mode: str | None = None
    auto_live_preview: bool | None = None
    auto_live_submit: bool | None = None
    notes: str | None = None
    paper_config: PaperExecutionConfigModel | None = None


class RuntimeTickRequest(BaseModel):
    bar_index: int | None = None


class RuntimeKillSwitchRequest(BaseModel):
    blocked: bool
    reason: str = ""


class RuntimeEventModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: str
    instance_id: str
    event_type: str
    payload: dict = Field(default_factory=dict)


class RuntimeEventsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    events: list[RuntimeEventModel] = Field(default_factory=list)


__all__ = [
    "RuntimeEventModel",
    "RuntimeEventsResponse",
    "RuntimeInstanceCreateRequest",
    "RuntimeInstanceListResponse",
    "RuntimeInstanceModel",
    "RuntimeInstanceResponse",
    "RuntimeInstanceUpdateRequest",
    "RuntimeKillSwitchRequest",
    "RuntimeTickRequest",
]
