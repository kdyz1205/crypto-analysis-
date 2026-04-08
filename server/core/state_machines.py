"""
Signal / Order / Position lifecycle state machines.

Enforces valid transitions so that code cannot, e.g., move a BLOCKED signal
back to READY or transition from DETECTED directly to FILLED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SignalState(Enum):
    DETECTED = "detected"
    VALIDATED = "validated"
    RISK_CHECKED = "risk_checked"
    READY = "ready"
    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    BLOCKED = "blocked"
    MANAGED = "managed"
    CLOSED = "closed"


# Allowed transitions: current -> set of allowed next states
_SIGNAL_TRANSITIONS: dict[SignalState, set[SignalState]] = {
    SignalState.DETECTED: {SignalState.VALIDATED, SignalState.BLOCKED, SignalState.EXPIRED},
    SignalState.VALIDATED: {SignalState.RISK_CHECKED, SignalState.BLOCKED},
    SignalState.RISK_CHECKED: {SignalState.READY, SignalState.BLOCKED},
    SignalState.READY: {SignalState.SUBMITTED, SignalState.EXPIRED, SignalState.CANCELLED},
    SignalState.SUBMITTED: {SignalState.FILLED, SignalState.REJECTED, SignalState.CANCELLED},
    SignalState.FILLED: {SignalState.MANAGED},
    SignalState.MANAGED: {SignalState.CLOSED},
    # Terminal states (no outgoing transitions):
    SignalState.REJECTED: set(),
    SignalState.CANCELLED: set(),
    SignalState.EXPIRED: set(),
    SignalState.BLOCKED: set(),
    SignalState.CLOSED: set(),
}


def can_transition(current: SignalState, target: SignalState) -> bool:
    return target in _SIGNAL_TRANSITIONS.get(current, set())


def validate_transition(current: SignalState, target: SignalState) -> None:
    if not can_transition(current, target):
        raise ValueError(f"Invalid signal transition: {current.value} -> {target.value}")


def is_terminal(state: SignalState) -> bool:
    return len(_SIGNAL_TRANSITIONS.get(state, set())) == 0


@dataclass
class SignalLifecycle:
    """Tracks a single signal from detection to close."""
    signal_id: str
    symbol: str
    side: str
    state: SignalState = SignalState.DETECTED
    history: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.history:
            self.history = [{"state": self.state.value, "ts": self.created_at, "reason": "created"}]

    def transition(self, target: SignalState, reason: str | None = None) -> "SignalLifecycle":
        validate_transition(self.state, target)
        self.state = target
        self.history.append({
            "state": target.value,
            "ts": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        })
        return self

    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side,
            "state": self.state.value,
            "history": self.history,
            "created_at": self.created_at,
        }
