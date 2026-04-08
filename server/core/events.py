"""
In-process async event bus + typed event factories.

Design:
- Single global `bus` instance, imported by publishers and subscribers.
- Subscribers are async functions. `subscribe('signal.*', handler)` supports prefix wildcards.
- Publishing fires all matching handlers concurrently via asyncio.gather.
- Exceptions in subscribers are logged, not propagated — one bad handler doesn't kill the bus.
- Last 500 events are retained in-memory for debugging / session replay.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


@dataclass(slots=True)
class Event:
    """Base event structure. All domain events flow through this."""
    type: str
    payload: dict
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    correlation_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


Subscriber = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self):
        self._exact: dict[str, list[Subscriber]] = {}
        self._wildcard: list[tuple[str, Subscriber]] = []
        self._history: list[Event] = []
        self._history_limit: int = 500

    def subscribe(self, event_type: str, handler: Subscriber) -> None:
        if event_type.endswith(".*"):
            prefix = event_type[:-2]
            self._wildcard.append((prefix, handler))
        else:
            self._exact.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Subscriber) -> None:
        if event_type.endswith(".*"):
            prefix = event_type[:-2]
            self._wildcard = [(p, h) for (p, h) in self._wildcard if not (p == prefix and h == handler)]
        elif event_type in self._exact:
            try:
                self._exact[event_type].remove(handler)
            except ValueError:
                pass

    async def publish(self, event: Event) -> None:
        """Fan out event to all matching subscribers concurrently."""
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        handlers: list[Subscriber] = list(self._exact.get(event.type, []))
        for (prefix, handler) in self._wildcard:
            if event.type.startswith(prefix + ".") or event.type == prefix:
                handlers.append(handler)

        if not handlers:
            return

        results = await asyncio.gather(
            *(self._safe_invoke(h, event) for h in handlers),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"[EventBus] Subscriber error for {event.type}: {r}")

    async def _safe_invoke(self, handler: Subscriber, event: Event) -> None:
        try:
            await handler(event)
        except Exception as e:
            raise e

    def get_recent(self, limit: int = 50) -> list[Event]:
        return self._history[-limit:]

    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._exact.values()) + len(self._wildcard)


# Global singleton
bus = EventBus()


# ── Event factory helpers ────────────────────────────────────────────────

def make_signal_detected(symbol, side, confidence, reason, price, sl, tp, regime, correlation_id=None) -> Event:
    return Event(
        type="signal.detected",
        payload={
            "symbol": symbol, "side": side, "confidence": confidence,
            "reason": reason, "price": price, "sl": sl, "tp": tp, "regime": regime,
        },
        correlation_id=correlation_id,
    )


def make_signal_blocked(symbol, side, block_reasons, correlation_id=None) -> Event:
    return Event(
        type="signal.blocked",
        payload={"symbol": symbol, "side": side, "block_reasons": list(block_reasons)},
        correlation_id=correlation_id,
    )


def make_signal_validated(symbol, side, correlation_id=None) -> Event:
    return Event(
        type="signal.validated",
        payload={"symbol": symbol, "side": side},
        correlation_id=correlation_id,
    )


def make_order_submitted(symbol, side, size_usd, price, mode, correlation_id=None) -> Event:
    return Event(
        type="order.submitted",
        payload={"symbol": symbol, "side": side, "size_usd": size_usd, "price": price, "mode": mode},
        correlation_id=correlation_id,
    )


def make_order_filled(symbol, side, size_usd, fill_price, mode, correlation_id=None) -> Event:
    return Event(
        type="order.filled",
        payload={"symbol": symbol, "side": side, "size_usd": size_usd, "fill_price": fill_price, "mode": mode},
        correlation_id=correlation_id,
    )


def make_order_rejected(symbol, side, reason, correlation_id=None) -> Event:
    return Event(
        type="order.rejected",
        payload={"symbol": symbol, "side": side, "reason": reason},
        correlation_id=correlation_id,
    )


def make_position_opened(symbol, side, size_usd, entry_price, sl, tp, correlation_id=None) -> Event:
    return Event(
        type="position.opened",
        payload={"symbol": symbol, "side": side, "size_usd": size_usd,
                 "entry_price": entry_price, "sl": sl, "tp": tp},
        correlation_id=correlation_id,
    )


def make_position_closed(symbol, side, entry_price, exit_price, pnl_pct, pnl_usd, reason, correlation_id=None) -> Event:
    return Event(
        type="position.closed",
        payload={
            "symbol": symbol, "side": side,
            "entry_price": entry_price, "exit_price": exit_price,
            "pnl_pct": pnl_pct, "pnl_usd": pnl_usd, "reason": reason,
        },
        correlation_id=correlation_id,
    )


def make_risk_limit_hit(limit_name, current_value, max_value, symbol=None) -> Event:
    return Event(
        type="risk.limit.hit",
        payload={"limit": limit_name, "current": current_value, "max": max_value, "symbol": symbol},
    )


def make_risk_cooldown_started(symbol, duration_sec, reason) -> Event:
    return Event(
        type="risk.cooldown.started",
        payload={"symbol": symbol, "duration_sec": duration_sec, "reason": reason},
    )


def make_agent_started(mode, equity, generation) -> Event:
    return Event(
        type="agent.started",
        payload={"mode": mode, "equity": equity, "generation": generation},
    )


def make_agent_stopped(reason="manual") -> Event:
    return Event(
        type="agent.stopped",
        payload={"reason": reason},
    )


def make_agent_mode_changed(old_mode, new_mode) -> Event:
    return Event(
        type="agent.mode.changed",
        payload={"from": old_mode, "to": new_mode},
    )


def make_agent_regime_changed(from_regime, to_regime, confidence) -> Event:
    return Event(
        type="agent.regime.changed",
        payload={"from": from_regime, "to": to_regime, "confidence": confidence},
    )


def make_agent_error(message, context=None) -> Event:
    return Event(
        type="agent.error.raised",
        payload={"message": message, "context": context or {}},
    )


def make_summary_daily(equity, daily_pnl, total_trades, win_rate, positions) -> Event:
    return Event(
        type="summary.daily",
        payload={
            "equity": equity, "daily_pnl": daily_pnl,
            "total_trades": total_trades, "win_rate": win_rate,
            "positions": positions,
        },
    )


def make_ops_healer_triggered(reason) -> Event:
    return Event(
        type="ops.healer.triggered",
        payload={"reason": reason},
    )
