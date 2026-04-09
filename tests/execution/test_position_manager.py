from server.execution.order_manager import PaperOrderManager
from server.execution.position_manager import PaperPositionManager
from server.execution.types import PaperExecutionConfig, RiskDecision
from server.strategy.types import StrategySignal


def _signal(
    *,
    signal_id: str = "sig-1",
    direction: str = "long",
    entry_price: float = 100.0,
    stop_price: float = 95.0,
    tp_price: float = 110.0,
) -> StrategySignal:
    return StrategySignal(
        signal_id=signal_id,
        line_id="line-1",
        symbol="BTCUSDT",
        timeframe="1h",
        signal_type="REJECTION_LONG" if direction == "long" else "REJECTION_SHORT",
        direction=direction,
        trigger_mode="rejection",
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
        line_side="support" if direction == "long" else "resistance",
        reason_code="test",
        factor_components={},
    )


def _decision(signal_id: str = "sig-1") -> RiskDecision:
    return RiskDecision(
        signal_id=signal_id,
        approved=True,
        blocking_reason="",
        risk_amount=20.0,
        proposed_quantity=2.0,
        stop_distance=5.0,
        exposure_after_fill=200.0,
    )


def _open_position(
    *,
    direction: str = "long",
    entry_price: float = 100.0,
    stop_price: float = 95.0,
    tp_price: float = 110.0,
):
    order_manager = PaperOrderManager()
    position_manager = PaperPositionManager()
    signal = _signal(direction=direction, entry_price=entry_price, stop_price=stop_price, tp_price=tp_price)
    intent = order_manager.create_order_intent_from_signal(signal, _decision(signal.signal_id), PaperExecutionConfig(), current_bar=1, current_ts=1)
    order_manager.submit_paper_order(intent)
    fill = order_manager.advance_orders_for_bar(
        current_bar=2,
        bar={"open": entry_price, "high": entry_price + 1, "low": entry_price - 1, "close": entry_price},
        timestamp=2,
    )[0]
    position = position_manager.open_from_fill(
        fill,
        order_manager,
        current_bar=2,
        current_ts=2,
        allow_multiple_same_direction_per_symbol=False,
    )
    return position_manager, position


def test_position_manager_opens_position_from_filled_order() -> None:
    position_manager, position = _open_position()
    assert position is not None
    assert position.status == "open"
    assert len(position_manager.get_open_positions()) == 1


def test_position_manager_closes_on_stop_loss() -> None:
    position_manager, _ = _open_position()
    closed = position_manager.advance_positions_for_bar(
        current_bar=2,
        bar={"open": 100.0, "high": 101.0, "low": 94.0, "close": 96.0},
        timestamp=2,
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"
    assert closed[0].exit_price == 95.0


def test_position_manager_closes_on_take_profit() -> None:
    position_manager, _ = _open_position()
    closed = position_manager.advance_positions_for_bar(
        current_bar=2,
        bar={"open": 100.0, "high": 111.0, "low": 99.0, "close": 109.0},
        timestamp=2,
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "take_profit"
    assert closed[0].exit_price == 110.0


def test_position_manager_same_bar_conflict_uses_worst_case() -> None:
    position_manager, _ = _open_position()
    closed = position_manager.advance_positions_for_bar(
        current_bar=2,
        bar={"open": 100.0, "high": 111.0, "low": 94.0, "close": 108.0},
        timestamp=2,
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop_loss"
    assert closed[0].exit_price == 95.0
