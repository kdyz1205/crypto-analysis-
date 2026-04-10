from __future__ import annotations

from dataclasses import replace
from typing import Any

from .order_manager import PaperOrderManager
from .types import PaperFill, PaperPosition, stable_execution_id


class PaperPositionManager:
    def __init__(self) -> None:
        self._positions_by_id: dict[str, PaperPosition] = {}
        self._position_id_by_signal_id: dict[str, str] = {}
        self._closed_positions: list[PaperPosition] = []

    def reset(self) -> None:
        self._positions_by_id.clear()
        self._position_id_by_signal_id.clear()
        self._closed_positions.clear()

    def load_state(
        self,
        *,
        open_positions: list[PaperPosition],
        recent_closed_positions: list[PaperPosition],
    ) -> None:
        self.reset()
        for position in open_positions:
            self._positions_by_id[position.position_id] = position
            self._position_id_by_signal_id[position.signal_id] = position.position_id
        self._closed_positions.extend(recent_closed_positions[-100:])

    def get_open_positions(self) -> list[PaperPosition]:
        return [
            position for position in sorted(self._positions_by_id.values(), key=lambda item: (item.opened_at_bar, item.position_id))
            if position.status == "open"
        ]

    def get_recent_closed_positions(self) -> list[PaperPosition]:
        return list(self._closed_positions[-20:])

    def get_closed_trade_count(self) -> int:
        return len(self._closed_positions)

    def can_open_fill(
        self,
        fill: PaperFill,
        *,
        allow_multiple_same_direction_per_symbol: bool,
    ) -> tuple[bool, str]:
        existing_id = self._position_id_by_signal_id.get(fill.signal_id)
        if existing_id is not None:
            return True, ""

        if not allow_multiple_same_direction_per_symbol:
            for position in self.get_open_positions():
                if position.symbol == fill.symbol and position.direction == fill.side:
                    return False, "same_symbol_direction_open_blocked"

        return True, ""

    def open_from_fill(
        self,
        fill: PaperFill,
        order_manager: PaperOrderManager,
        *,
        current_bar: int,
        current_ts: Any,
        allow_multiple_same_direction_per_symbol: bool,
    ) -> PaperPosition | None:
        intent = order_manager.get_intent(fill.signal_id)
        if intent is None:
            return None

        can_open, _ = self.can_open_fill(
            fill,
            allow_multiple_same_direction_per_symbol=allow_multiple_same_direction_per_symbol,
        )
        if not can_open:
            return None

        existing_id = self._position_id_by_signal_id.get(fill.signal_id)
        if existing_id is not None:
            return self._positions_by_id.get(existing_id)

        position = PaperPosition(
            position_id=stable_execution_id("position", fill.signal_id),
            signal_id=fill.signal_id,
            line_id=fill.line_id,
            client_order_id=fill.client_order_id,
            symbol=fill.symbol,
            timeframe=fill.timeframe,
            direction=fill.side,
            quantity=float(fill.quantity),
            entry_price=float(fill.fill_price),
            mark_price=float(fill.fill_price),
            stop_price=float(intent.stop_price),
            tp_price=float(intent.tp_price),
            status="open",
            opened_at_bar=current_bar,
            opened_at_ts=current_ts,
        )
        self._positions_by_id[position.position_id] = position
        self._position_id_by_signal_id[fill.signal_id] = position.position_id
        return position

    def advance_positions_for_bar(self, current_bar: int, bar: dict[str, Any], timestamp: Any) -> list[PaperPosition]:
        closed_positions: list[PaperPosition] = []
        for position in list(self.get_open_positions()):
            exit_price, exit_reason = self._resolve_exit(position, bar)
            if exit_price is not None and exit_reason is not None:
                updated = self._close_position(position, exit_price, current_bar, timestamp, exit_reason)
                closed_positions.append(updated)
                continue

            close_price = float(bar["close"])
            updated = replace(
                position,
                mark_price=close_price,
                unrealized_pnl=self._calc_pnl(position.direction, position.entry_price, close_price, position.quantity),
            )
            self._positions_by_id[position.position_id] = updated
        return closed_positions

    def _resolve_exit(self, position: PaperPosition, bar: dict[str, Any]) -> tuple[float | None, str | None]:
        low = float(bar["low"])
        high = float(bar["high"])

        if position.direction == "long":
            stop_hit = low <= position.stop_price
            tp_hit = high >= position.tp_price
            if stop_hit:
                return float(position.stop_price), "stop_loss"
            if tp_hit:
                return float(position.tp_price), "take_profit"
        else:
            stop_hit = high >= position.stop_price
            tp_hit = low <= position.tp_price
            if stop_hit:
                return float(position.stop_price), "stop_loss"
            if tp_hit:
                return float(position.tp_price), "take_profit"

        return None, None

    def _close_position(
        self,
        position: PaperPosition,
        exit_price: float,
        current_bar: int,
        timestamp: Any,
        exit_reason: str,
    ) -> PaperPosition:
        realized_pnl = self._calc_pnl(position.direction, position.entry_price, exit_price, position.quantity)
        updated = replace(
            position,
            mark_price=float(exit_price),
            status="closed",
            closed_at_bar=current_bar,
            closed_at_ts=timestamp,
            exit_price=float(exit_price),
            exit_reason=exit_reason,
            realized_pnl=realized_pnl,
            unrealized_pnl=0.0,
        )
        self._positions_by_id[position.position_id] = updated
        self._closed_positions.append(updated)
        self._closed_positions = self._closed_positions[-100:]
        return updated

    @staticmethod
    def _calc_pnl(direction: str, entry_price: float, exit_price: float, quantity: float) -> float:
        if direction == "long":
            return (float(exit_price) - float(entry_price)) * float(quantity)
        return (float(entry_price) - float(exit_price)) * float(quantity)


__all__ = ["PaperPositionManager"]
