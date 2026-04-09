from __future__ import annotations

from typing import Mapping, Sequence

from ..execution.types import (
    KillSwitchState,
    PaperAccountSummary,
    PaperExecutionConfig,
    PaperOrder,
    PaperPosition,
    RiskDecision,
)
from ..strategy.types import StrategySignal


def cooldown_scope_key(symbol: str, timeframe: str, direction: str) -> str:
    return f"{symbol}:{timeframe}:{direction}"


def evaluate_signal_risk(
    signal: StrategySignal,
    account: PaperAccountSummary,
    open_positions: Sequence[PaperPosition],
    config: PaperExecutionConfig,
    *,
    current_bar: int,
    cooldowns: Mapping[str, int] | None = None,
    pending_orders: Sequence[PaperOrder] | None = None,
    kill_switch: KillSwitchState | None = None,
) -> RiskDecision:
    stop_distance = abs(float(signal.entry_price) - float(signal.stop_price))
    risk_amount = float(account.equity) * float(config.risk_per_trade)
    pending_orders = tuple(order for order in (pending_orders or ()) if order.status == "pending")

    if kill_switch is not None and kill_switch.blocked:
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason=kill_switch.reason or "kill_switch_blocked",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    if stop_distance <= 0:
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="invalid_stop_distance",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    if account.daily_realized_pnl <= -(config.max_daily_loss * account.starting_equity):
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="max_daily_loss_hit",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    if account.consecutive_losses >= config.max_consecutive_losses:
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="max_consecutive_losses_hit",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    reserved_slot_count = max(int(account.open_position_count), len(open_positions)) + len(pending_orders)
    if reserved_slot_count >= config.max_concurrent_positions:
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="max_concurrent_positions_hit",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    same_symbol_positions = [position for position in open_positions if position.symbol == signal.symbol and position.status == "open"]
    same_symbol_pending_orders = [order for order in pending_orders if order.symbol == signal.symbol]
    if (len(same_symbol_positions) + len(same_symbol_pending_orders)) >= config.max_positions_per_symbol:
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="max_positions_per_symbol_hit",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    if (not config.allow_multiple_same_direction_per_symbol) and any(
        position.symbol == signal.symbol and position.direction == signal.direction and position.status == "open"
        for position in open_positions
    ):
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="same_symbol_direction_blocked",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    if (not config.allow_multiple_same_direction_per_symbol) and any(
        order.symbol == signal.symbol and order.side == signal.direction and order.status == "pending"
        for order in pending_orders
    ):
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="same_symbol_direction_pending_blocked",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    cooldowns = cooldowns or {}
    scope_key = cooldown_scope_key(signal.symbol, signal.timeframe, signal.direction)
    cooldown_until = cooldowns.get(scope_key)
    if cooldown_until is not None and current_bar < cooldown_until:
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="cooldown_active",
            risk_amount=risk_amount,
            proposed_quantity=0.0,
            stop_distance=stop_distance,
            exposure_after_fill=float(account.total_exposure),
        )

    proposed_quantity = risk_amount / stop_distance
    pending_exposure = sum(float(order.quantity) * float(order.price) for order in pending_orders)
    reserved_exposure = float(account.total_exposure) + pending_exposure
    exposure_after_fill = reserved_exposure + (proposed_quantity * float(signal.entry_price))
    if exposure_after_fill > (account.equity * config.max_total_exposure):
        return RiskDecision(
            signal_id=signal.signal_id,
            approved=False,
            blocking_reason="max_total_exposure_hit",
            risk_amount=risk_amount,
            proposed_quantity=proposed_quantity,
            stop_distance=stop_distance,
            exposure_after_fill=exposure_after_fill,
        )

    return RiskDecision(
        signal_id=signal.signal_id,
        approved=True,
        blocking_reason="",
        risk_amount=risk_amount,
        proposed_quantity=proposed_quantity,
        stop_distance=stop_distance,
        exposure_after_fill=exposure_after_fill,
    )


__all__ = [
    "cooldown_scope_key",
    "evaluate_signal_risk",
]
