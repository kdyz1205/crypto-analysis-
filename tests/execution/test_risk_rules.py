from server.execution.types import (
    KillSwitchState,
    PaperAccountSummary,
    PaperExecutionConfig,
    PaperOrder,
    PaperPosition,
)
from server.risk.risk_rules import cooldown_scope_key, evaluate_signal_risk
from server.strategy.types import StrategySignal


def _signal(
    *,
    signal_id: str = "sig-1",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    direction: str = "short",
    trigger_mode: str = "pre_limit",
    entry_price: float = 100.0,
    stop_price: float = 105.0,
    tp_price: float = 90.0,
) -> StrategySignal:
    return StrategySignal(
        signal_id=signal_id,
        line_id="line-1",
        symbol=symbol,
        timeframe=timeframe,
        signal_type="PRE_LIMIT_SHORT" if direction == "short" else "PRE_LIMIT_LONG",
        direction=direction,
        trigger_mode=trigger_mode,
        timestamp=1,
        trigger_bar_index=1,
        score=0.8,
        priority_rank=1,
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
        risk_reward=2.0,
        confirming_touch_count=3,
        bars_since_last_confirming_touch=1,
        distance_to_line=0.1,
        line_side="resistance" if direction == "short" else "support",
        reason_code="test",
        factor_components={},
    )


def _account(**overrides) -> PaperAccountSummary:
    base = {
        "starting_equity": 10_000.0,
        "equity": 10_000.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "daily_realized_pnl": 0.0,
        "consecutive_losses": 0,
        "total_exposure": 0.0,
        "open_order_count": 0,
        "open_position_count": 0,
        "closed_trade_count": 0,
        "last_processed_bar_by_stream": {},
    }
    base.update(overrides)
    return PaperAccountSummary(**base)


def _position(symbol: str = "BTCUSDT", direction: str = "short") -> PaperPosition:
    return PaperPosition(
        position_id="pos-1",
        signal_id="sig-pos",
        line_id="line-pos",
        client_order_id="client-pos",
        symbol=symbol,
        timeframe="1h",
        direction=direction,
        quantity=1.0,
        entry_price=100.0,
        mark_price=100.0,
        stop_price=105.0,
        tp_price=90.0,
        status="open",
        opened_at_bar=1,
    )


def _pending_order(symbol: str = "BTCUSDT", side: str = "short", *, quantity: float = 6.0, price: float = 100.0) -> PaperOrder:
    return PaperOrder(
        order_id="order-1",
        line_id="line-1",
        client_order_id="client-1",
        signal_id="sig-order",
        symbol=symbol,
        timeframe="1h",
        side=side,
        order_type="limit",
        trigger_mode="pre_limit",
        price=price,
        quantity=quantity,
        filled_quantity=0.0,
        avg_fill_price=0.0,
        status="pending",
        created_at_bar=1,
        updated_at_bar=1,
    )


def test_risk_rejects_non_positive_stop_distance() -> None:
    decision = evaluate_signal_risk(
        _signal(entry_price=100.0, stop_price=100.0),
        _account(),
        [],
        PaperExecutionConfig(),
        current_bar=5,
    )
    assert decision.approved is False
    assert decision.blocking_reason == "invalid_stop_distance"


def test_risk_calculates_quantity_from_stop_distance() -> None:
    decision = evaluate_signal_risk(
        _signal(entry_price=100.0, stop_price=105.0),
        _account(),
        [],
        PaperExecutionConfig(risk_per_trade=0.003),
        current_bar=5,
    )
    assert decision.approved is True
    assert decision.risk_amount == 30.0
    assert decision.stop_distance == 5.0
    assert decision.proposed_quantity == 6.0
    assert decision.exposure_after_fill == 600.0


def test_risk_blocks_max_concurrent_positions() -> None:
    account = _account(open_position_count=3)
    decision = evaluate_signal_risk(_signal(), account, [_position("ETHUSDT")], PaperExecutionConfig(), current_bar=5)
    assert decision.approved is False
    assert decision.blocking_reason == "max_concurrent_positions_hit"


def test_risk_blocks_max_positions_per_symbol() -> None:
    account = _account(open_position_count=1)
    decision = evaluate_signal_risk(_signal(), account, [_position("BTCUSDT")], PaperExecutionConfig(), current_bar=5)
    assert decision.approved is False
    assert decision.blocking_reason == "max_positions_per_symbol_hit"


def test_risk_blocks_active_cooldown() -> None:
    signal = _signal(symbol="BTCUSDT", timeframe="4h", direction="short")
    cooldowns = {cooldown_scope_key("BTCUSDT", "4h", "short"): 12}
    decision = evaluate_signal_risk(signal, _account(), [], PaperExecutionConfig(), current_bar=10, cooldowns=cooldowns)
    assert decision.approved is False
    assert decision.blocking_reason == "cooldown_active"


def test_risk_blocks_daily_loss_and_consecutive_losses() -> None:
    config = PaperExecutionConfig(max_daily_loss=0.02, max_consecutive_losses=3)
    daily_loss_decision = evaluate_signal_risk(
        _signal(),
        _account(daily_realized_pnl=-250.0),
        [],
        config,
        current_bar=5,
    )
    consecutive_loss_decision = evaluate_signal_risk(
        _signal(signal_id="sig-2"),
        _account(consecutive_losses=3),
        [],
        config,
        current_bar=5,
    )
    assert daily_loss_decision.approved is False
    assert daily_loss_decision.blocking_reason == "max_daily_loss_hit"
    assert consecutive_loss_decision.approved is False
    assert consecutive_loss_decision.blocking_reason == "max_consecutive_losses_hit"


def test_risk_respects_kill_switch() -> None:
    decision = evaluate_signal_risk(
        _signal(),
        _account(),
        [],
        PaperExecutionConfig(),
        current_bar=5,
        kill_switch=KillSwitchState(blocked=True, reason="manual_block"),
    )
    assert decision.approved is False
    assert decision.blocking_reason == "manual_block"


def test_risk_blocks_same_symbol_when_pending_order_already_reserves_slot() -> None:
    decision = evaluate_signal_risk(
        _signal(),
        _account(),
        [],
        PaperExecutionConfig(),
        current_bar=5,
        pending_orders=[_pending_order("BTCUSDT", "short")],
    )
    assert decision.approved is False
    assert decision.blocking_reason == "max_positions_per_symbol_hit"


def test_risk_counts_pending_orders_in_total_exposure() -> None:
    config = PaperExecutionConfig(max_total_exposure=0.1)
    decision = evaluate_signal_risk(
        _signal(symbol="ETHUSDT"),
        _account(),
        [],
        config,
        current_bar=5,
        pending_orders=[_pending_order("BTCUSDT", "short", quantity=6.0, price=100.0)],
    )
    assert decision.approved is False
    assert decision.blocking_reason == "max_total_exposure_hit"
