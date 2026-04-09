from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from hashlib import sha1
from typing import Any, Literal


IntentStatus = Literal[
    "created",
    "approved",
    "blocked",
    "submitted",
    "filled",
    "cancelled",
    "rejected",
    "expired",
]
OrderType = Literal["limit", "market"]
OrderStatus = Literal["pending", "filled", "cancelled", "rejected", "expired"]
PositionStatus = Literal["open", "closing", "closed"]
PositionDirection = Literal["long", "short"]


def stable_execution_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class PaperExecutionConfig:
    risk_per_trade: float = 0.003
    max_concurrent_positions: int = 3
    max_positions_per_symbol: int = 1
    max_total_exposure: float = 1.0
    max_daily_loss: float = 0.02
    max_consecutive_losses: int = 3
    cancel_after_bars: int = 3
    cooldown_bars_after_loss: int = 10
    starting_equity: float = 10_000.0
    allow_multiple_same_direction_per_symbol: bool = False


@dataclass(slots=True)
class OrderIntent:
    order_intent_id: str
    signal_id: str
    line_id: str
    client_order_id: str
    symbol: str
    timeframe: str
    side: PositionDirection
    order_type: OrderType
    trigger_mode: str
    entry_price: float
    stop_price: float
    tp_price: float
    quantity: float
    status: IntentStatus
    reason: str = ""
    created_at_bar: int = -1
    created_at_ts: Any = None


@dataclass(slots=True)
class PaperOrder:
    order_id: str
    line_id: str
    client_order_id: str
    signal_id: str
    symbol: str
    timeframe: str
    side: PositionDirection
    order_type: OrderType
    trigger_mode: str
    price: float
    quantity: float
    filled_quantity: float
    avg_fill_price: float
    status: OrderStatus
    reason: str = ""
    created_at_bar: int = -1
    updated_at_bar: int = -1


@dataclass(slots=True)
class PaperFill:
    fill_id: str
    order_id: str
    client_order_id: str
    signal_id: str
    line_id: str
    symbol: str
    timeframe: str
    side: PositionDirection
    fill_price: float
    quantity: float
    filled_at_bar: int
    filled_at_ts: Any = None
    fill_reason: str = "order_fill"


@dataclass(slots=True)
class PaperPosition:
    position_id: str
    signal_id: str
    line_id: str
    client_order_id: str
    symbol: str
    timeframe: str
    direction: PositionDirection
    quantity: float
    entry_price: float
    mark_price: float
    stop_price: float
    tp_price: float
    status: PositionStatus
    opened_at_bar: int
    closed_at_bar: int | None = None
    opened_at_ts: Any = None
    closed_at_ts: Any = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    exit_price: float | None = None
    exit_reason: str | None = None


@dataclass(slots=True)
class RiskDecision:
    signal_id: str
    approved: bool
    blocking_reason: str
    risk_amount: float
    proposed_quantity: float
    stop_distance: float
    exposure_after_fill: float


@dataclass(slots=True)
class KillSwitchState:
    blocked: bool = False
    manual_blocked: bool = False
    data_blocked: bool = False
    risk_blocked: bool = False
    reason: str = ""


@dataclass(slots=True)
class PaperAccountSummary:
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
    last_processed_bar_by_stream: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class PaperExecutionState:
    config: PaperExecutionConfig
    account: PaperAccountSummary
    kill_switch: KillSwitchState
    intents: list[OrderIntent] = field(default_factory=list)
    open_orders: list[PaperOrder] = field(default_factory=list)
    open_positions: list[PaperPosition] = field(default_factory=list)
    recent_fills: list[PaperFill] = field(default_factory=list)
    recent_closed_positions: list[PaperPosition] = field(default_factory=list)
    cooldowns: dict[str, int] = field(default_factory=dict)


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return dataclass_to_dict(asdict(value))
    if isinstance(value, dict):
        return {str(key): dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


__all__ = [
    "IntentStatus",
    "KillSwitchState",
    "OrderIntent",
    "OrderStatus",
    "OrderType",
    "PaperAccountSummary",
    "PaperExecutionConfig",
    "PaperExecutionState",
    "PaperFill",
    "PaperOrder",
    "PaperPosition",
    "PositionDirection",
    "PositionStatus",
    "RiskDecision",
    "dataclass_to_dict",
    "stable_execution_id",
]
