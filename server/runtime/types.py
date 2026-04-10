from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..execution.types import PaperExecutionConfig, PaperExecutionState

RuntimeMode = Literal["disabled", "demo", "live"]
RuntimeState = Literal["stopped", "running", "blocked"]


@dataclass(slots=True)
class RuntimeStrategyConfig:
    enabled_trigger_modes: tuple[str, ...] = ("pre_limit",)
    lookback_bars: int | None = 80
    min_touches: int | None = None
    confirm_threshold: float | None = None
    score_threshold: float | None = None
    rr_target: float | None = None
    window_bars: int | None = 100


@dataclass(slots=True)
class RuntimeInstanceConfig:
    instance_id: str
    label: str
    symbol: str
    timeframe: str
    subaccount_label: str = ""
    history_mode: str = "fast_window"
    analysis_bars: int = 500
    days: int = 365
    tick_interval_seconds: int = 60
    auto_restart_on_boot: bool = False
    live_mode: RuntimeMode = "disabled"
    auto_live_preview: bool = True
    auto_live_submit: bool = False
    notes: str = ""
    paper_config: PaperExecutionConfig = field(default_factory=PaperExecutionConfig)
    strategy_config: RuntimeStrategyConfig = field(default_factory=RuntimeStrategyConfig)


@dataclass(slots=True)
class RuntimeInstanceStatus:
    runtime_state: RuntimeState = "stopped"
    last_tick_at: str | None = None
    last_processed_bar: int | None = None
    last_error: str = ""
    last_runtime_note: str = ""
    last_live_result: dict[str, Any] | None = None
    paper_current_day: str | None = None
    last_history: dict[str, Any] | None = None
    paper_state: PaperExecutionState | None = None
    live_engine_state: dict[str, Any] | None = None


@dataclass(slots=True)
class RuntimeInstanceRecord:
    config: RuntimeInstanceConfig
    status: RuntimeInstanceStatus


__all__ = [
    "RuntimeInstanceConfig",
    "RuntimeInstanceRecord",
    "RuntimeInstanceStatus",
    "RuntimeMode",
    "RuntimeState",
    "RuntimeStrategyConfig",
]
