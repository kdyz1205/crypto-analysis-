"""
Event type constants — defined now for Phase 3, used as documentation in Phase 1.

Naming convention: <domain>.<entity>.<verb>
"""


class SignalEvents:
    DETECTED = "signal.detected"
    VALIDATED = "signal.validated"
    REJECTED = "signal.rejected"
    BLOCKED = "signal.blocked"
    EXPIRED = "signal.expired"


class TradeEvents:
    INTENT_CREATED = "trade.intent.created"
    INTENT_CANCELLED = "trade.intent.cancelled"


class OrderEvents:
    SUBMITTED = "order.submitted"
    ACKNOWLEDGED = "order.acknowledged"
    PARTIALLY_FILLED = "order.partially_filled"
    FILLED = "order.filled"
    CANCELLED = "order.cancelled"
    REJECTED = "order.rejected"


class PositionEvents:
    OPENED = "position.opened"
    SCALED = "position.scaled"
    REDUCED = "position.reduced"
    CLOSED = "position.closed"
    STOP_HIT = "position.stop_hit"
    TARGET_HIT = "position.target_hit"


class RiskEvents:
    CHECK_PASSED = "risk.check.passed"
    CHECK_FAILED = "risk.check.failed"
    LIMIT_HIT = "risk.limit.hit"
    COOLDOWN_STARTED = "risk.cooldown.started"
    COOLDOWN_ENDED = "risk.cooldown.ended"
    KILL_SWITCH_ARMED = "risk.kill_switch.armed"
    KILL_SWITCH_RELEASED = "risk.kill_switch.released"


class AgentEvents:
    STARTED = "agent.started"
    STOPPED = "agent.stopped"
    REVIVED = "agent.revived"
    MODE_CHANGED = "agent.mode.changed"
    CONFIG_UPDATED = "agent.config.updated"
    ERROR_RAISED = "agent.error.raised"


class OpsEvents:
    LOG_EMITTED = "ops.log.emitted"
    HEALER_TRIGGERED = "ops.healer.triggered"
    HEALER_COMPLETED = "ops.healer.completed"
    INTEGRATION_FAILED = "ops.integration.failed"


class NotifyEvents:
    TELEGRAM_SENT = "notify.telegram.sent"
    TELEGRAM_FAILED = "notify.telegram.failed"


class MarketEvents:
    TICK_RECEIVED = "market.tick.received"
    SNAPSHOT_UPDATED = "market.snapshot.updated"
