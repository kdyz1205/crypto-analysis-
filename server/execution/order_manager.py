from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..strategy.types import StrategySignal
from .types import (
    OrderIntent,
    PaperExecutionConfig,
    PaperFill,
    PaperOrder,
    RiskDecision,
    stable_execution_id,
)


def make_client_order_id(signal: StrategySignal) -> str:
    return f"paper-{signal.symbol.lower()}-{signal.signal_id[:10]}"


class PaperOrderManager:
    def __init__(self) -> None:
        self._intents_by_signal_id: dict[str, OrderIntent] = {}
        self._intents_by_client_order_id: dict[str, OrderIntent] = {}
        self._orders_by_id: dict[str, PaperOrder] = {}
        self._order_id_by_signal_id: dict[str, str] = {}
        self._recent_fills: list[PaperFill] = []

    def reset(self) -> None:
        self._intents_by_signal_id.clear()
        self._intents_by_client_order_id.clear()
        self._orders_by_id.clear()
        self._order_id_by_signal_id.clear()
        self._recent_fills.clear()

    def get_intent(self, signal_id: str) -> OrderIntent | None:
        return self._intents_by_signal_id.get(signal_id)

    def get_intents(self) -> list[OrderIntent]:
        return sorted(self._intents_by_signal_id.values(), key=lambda intent: (intent.created_at_bar, intent.signal_id))

    def get_open_orders(self) -> list[PaperOrder]:
        return [
            order for order in sorted(self._orders_by_id.values(), key=lambda item: (item.created_at_bar, item.order_id))
            if order.status == "pending"
        ]

    def get_recent_fills(self) -> list[PaperFill]:
        return list(self._recent_fills[-20:])

    def create_order_intent_from_signal(
        self,
        signal: StrategySignal,
        risk_decision: RiskDecision,
        config: PaperExecutionConfig,
        *,
        current_bar: int,
        current_ts: Any,
    ) -> OrderIntent:
        del config  # Config is carried for API symmetry and future extension.

        existing = self._intents_by_signal_id.get(signal.signal_id)
        if existing is not None:
            return existing

        client_order_id = make_client_order_id(signal)
        existing_client = self._intents_by_client_order_id.get(client_order_id)
        if existing_client is not None:
            return existing_client

        intent = OrderIntent(
            order_intent_id=stable_execution_id("intent", signal.signal_id),
            signal_id=signal.signal_id,
            line_id=signal.line_id,
            client_order_id=client_order_id,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            side=signal.direction,
            order_type="limit" if signal.trigger_mode == "pre_limit" else "market",
            trigger_mode=signal.trigger_mode,
            entry_price=float(signal.entry_price),
            stop_price=float(signal.stop_price),
            tp_price=float(signal.tp_price),
            quantity=float(risk_decision.proposed_quantity),
            status="approved" if risk_decision.approved else "blocked",
            reason=risk_decision.blocking_reason,
            created_at_bar=current_bar,
            created_at_ts=current_ts,
        )
        self._intents_by_signal_id[signal.signal_id] = intent
        self._intents_by_client_order_id[intent.client_order_id] = intent
        return intent

    def submit_paper_order(self, intent: OrderIntent) -> PaperOrder | None:
        if intent.status != "approved":
            return None

        existing_order_id = self._order_id_by_signal_id.get(intent.signal_id)
        if existing_order_id is not None:
            return self._orders_by_id[existing_order_id]

        order = PaperOrder(
            order_id=stable_execution_id("order", intent.signal_id),
            line_id=intent.line_id,
            client_order_id=intent.client_order_id,
            signal_id=intent.signal_id,
            symbol=intent.symbol,
            timeframe=intent.timeframe,
            side=intent.side,
            order_type=intent.order_type,
            trigger_mode=intent.trigger_mode,
            price=float(intent.entry_price),
            quantity=float(intent.quantity),
            filled_quantity=0.0,
            avg_fill_price=0.0,
            status="pending",
            reason="",
            created_at_bar=intent.created_at_bar,
            updated_at_bar=intent.created_at_bar,
        )
        self._orders_by_id[order.order_id] = order
        self._order_id_by_signal_id[intent.signal_id] = order.order_id

        updated_intent = replace(intent, status="submitted")
        self._intents_by_signal_id[intent.signal_id] = updated_intent
        self._intents_by_client_order_id[intent.client_order_id] = updated_intent
        return order

    def cancel_paper_order(self, order_id: str, reason: str) -> PaperOrder | None:
        order = self._orders_by_id.get(order_id)
        if order is None or order.status != "pending":
            return order

        updated = replace(order, status="cancelled", reason=reason, updated_at_bar=order.updated_at_bar)
        self._orders_by_id[order_id] = updated
        intent = self._intents_by_signal_id.get(order.signal_id)
        if intent is not None:
            updated_intent = replace(intent, status="cancelled", reason=reason)
            self._intents_by_signal_id[order.signal_id] = updated_intent
            self._intents_by_client_order_id[intent.client_order_id] = updated_intent
        return updated

    def expire_stale_orders(self, current_bar: int, cancel_after_bars: int) -> list[PaperOrder]:
        expired: list[PaperOrder] = []
        for order in list(self._orders_by_id.values()):
            if order.status != "pending":
                continue
            if (current_bar - order.created_at_bar) < cancel_after_bars:
                continue

            updated = replace(order, status="expired", reason="stale_order", updated_at_bar=current_bar)
            self._orders_by_id[order.order_id] = updated
            expired.append(updated)
            intent = self._intents_by_signal_id.get(order.signal_id)
            if intent is not None:
                updated_intent = replace(intent, status="expired", reason="stale_order")
                self._intents_by_signal_id[order.signal_id] = updated_intent
                self._intents_by_client_order_id[intent.client_order_id] = updated_intent
        return expired

    def advance_orders_for_bar(self, current_bar: int, bar: dict[str, Any], timestamp: Any) -> list[PaperFill]:
        fills: list[PaperFill] = []
        for order in list(self._orders_by_id.values()):
            if order.status != "pending" or current_bar <= order.created_at_bar:
                continue

            fill_price: float | None = None
            if order.order_type == "market":
                fill_price = float(bar["open"])
            else:
                low = float(bar["low"])
                high = float(bar["high"])
                if low <= order.price <= high:
                    fill_price = float(order.price)

            if fill_price is None:
                continue

            updated_order = replace(
                order,
                status="filled",
                filled_quantity=float(order.quantity),
                avg_fill_price=fill_price,
                updated_at_bar=current_bar,
            )
            self._orders_by_id[order.order_id] = updated_order

            intent = self._intents_by_signal_id.get(order.signal_id)
            if intent is not None:
                updated_intent = replace(intent, status="filled", reason="")
                self._intents_by_signal_id[order.signal_id] = updated_intent
                self._intents_by_client_order_id[intent.client_order_id] = updated_intent

            fill = PaperFill(
                fill_id=stable_execution_id("fill", order.order_id, current_bar),
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                signal_id=order.signal_id,
                line_id=order.line_id,
                symbol=order.symbol,
                timeframe=order.timeframe,
                side=order.side,
                fill_price=fill_price,
                quantity=float(order.quantity),
                filled_at_bar=current_bar,
                filled_at_ts=timestamp,
            )
            self._recent_fills.append(fill)
            self._recent_fills = self._recent_fills[-100:]
            fills.append(fill)
        return fills


__all__ = [
    "PaperOrderManager",
    "make_client_order_id",
]
