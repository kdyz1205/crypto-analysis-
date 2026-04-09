import server.execution.order_manager as order_manager_module
from server.execution.order_manager import PaperOrderManager
from server.execution.types import PaperExecutionConfig, RiskDecision
from server.strategy.types import StrategySignal


def _signal(
    *,
    signal_id: str = "sig-1",
    symbol: str = "BTCUSDT",
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
        timeframe="1h",
        signal_type="PRE_LIMIT_SHORT" if direction == "short" else "PRE_LIMIT_LONG",
        direction=direction,
        trigger_mode=trigger_mode,
        timestamp=1,
        trigger_bar_index=1,
        score=0.9,
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


def _decision(signal_id: str = "sig-1", approved: bool = True) -> RiskDecision:
    return RiskDecision(
        signal_id=signal_id,
        approved=approved,
        blocking_reason="" if approved else "blocked",
        risk_amount=30.0,
        proposed_quantity=6.0 if approved else 0.0,
        stop_distance=5.0,
        exposure_after_fill=600.0,
    )


def test_order_manager_dedupes_same_signal_id() -> None:
    manager = PaperOrderManager()
    signal = _signal()
    first = manager.create_order_intent_from_signal(signal, _decision(), PaperExecutionConfig(), current_bar=1, current_ts=1)
    second = manager.create_order_intent_from_signal(signal, _decision(), PaperExecutionConfig(), current_bar=1, current_ts=1)
    assert first.order_intent_id == second.order_intent_id
    assert len(manager.get_intents()) == 1


def test_order_manager_guards_duplicate_client_order_id(monkeypatch) -> None:
    monkeypatch.setattr(order_manager_module, "make_client_order_id", lambda signal: "paper-duplicate")
    manager = PaperOrderManager()
    first = manager.create_order_intent_from_signal(_signal(signal_id="sig-1"), _decision("sig-1"), PaperExecutionConfig(), current_bar=1, current_ts=1)
    second = manager.create_order_intent_from_signal(_signal(signal_id="sig-2"), _decision("sig-2"), PaperExecutionConfig(), current_bar=1, current_ts=1)
    assert first.order_intent_id == second.order_intent_id
    assert len(manager.get_intents()) == 1


def test_order_manager_expires_stale_pending_orders() -> None:
    manager = PaperOrderManager()
    intent = manager.create_order_intent_from_signal(_signal(), _decision(), PaperExecutionConfig(), current_bar=1, current_ts=1)
    order = manager.submit_paper_order(intent)
    assert order is not None
    expired = manager.expire_stale_orders(current_bar=5, cancel_after_bars=3)
    assert len(expired) == 1
    assert expired[0].status == "expired"
    assert manager.get_intent("sig-1").status == "expired"


def test_order_manager_fills_limit_order_when_next_bar_covers_price() -> None:
    manager = PaperOrderManager()
    intent = manager.create_order_intent_from_signal(_signal(entry_price=100.0), _decision(), PaperExecutionConfig(), current_bar=1, current_ts=1)
    manager.submit_paper_order(intent)
    fills = manager.advance_orders_for_bar(
        current_bar=2,
        bar={"open": 101.0, "high": 102.0, "low": 99.0, "close": 100.5},
        timestamp=2,
    )
    assert len(fills) == 1
    assert fills[0].fill_price == 100.0
    assert fills[0].signal_id == "sig-1"


def test_order_manager_cancel_updates_bar_when_provided() -> None:
    manager = PaperOrderManager()
    intent = manager.create_order_intent_from_signal(_signal(), _decision(), PaperExecutionConfig(), current_bar=1, current_ts=1)
    order = manager.submit_paper_order(intent)
    assert order is not None
    cancelled = manager.cancel_paper_order(order.order_id, "manual_cancel", current_bar=4)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.updated_at_bar == 4
