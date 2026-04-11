"""Tests that risk_rules uses timeframe-calibrated stop floors."""
# Import types first to avoid circular import
from server.execution.types import (
    KillSwitchState,
    PaperAccountSummary,
    PaperExecutionConfig,
    PaperOrder,
    PaperPosition,
)
from server.strategy.types import StrategySignal
from server.risk.risk_rules import evaluate_signal_risk


def _make_signal(timeframe="1h", entry=100.0, stop=99.99, tp=105.0, direction="long"):
    return StrategySignal(
        signal_id="test", line_id="line1", symbol="TESTUSDT",
        timeframe=timeframe, signal_type="ZONE_SUPPORT_LONG",
        direction=direction, trigger_mode="pre_limit",
        timestamp=0, trigger_bar_index=0, score=0.7,
        priority_rank=1, entry_price=entry, stop_price=stop,
        tp_price=tp, risk_reward=5.0, confirming_touch_count=3,
        bars_since_last_confirming_touch=5, distance_to_line=0.1,
        line_side="support", reason_code="test",
    )


def _make_account(equity=10000.0):
    return PaperAccountSummary(
        equity=equity, starting_equity=equity, total_exposure=0.0,
        open_position_count=0, daily_realized_pnl=0.0,
        consecutive_losses=0, last_processed_bar_by_stream={},
        realized_pnl=0.0, unrealized_pnl=0.0,
        open_order_count=0, closed_trade_count=0,
    )


def _make_config():
    return PaperExecutionConfig(
        risk_per_trade=0.01,
        max_concurrent_positions=3,
        max_positions_per_symbol=1,
        max_total_exposure=1.0,
    )


def test_tight_stop_gets_floored_by_timeframe():
    """A 0.01% stop on 4h should be floored to ~1.0% (calibrated minimum)."""
    signal = _make_signal(timeframe="4h", entry=100.0, stop=99.99)  # 0.01% stop
    decision = evaluate_signal_risk(signal, _make_account(), [], _make_config(), current_bar=0)
    assert decision.approved
    # With $10K equity, 1% risk = $100, floored stop ~1.0% = $1.00
    # proposed_quantity = $100 / $1.00 = 100 units (not 10,000 from 0.01% stop)
    assert decision.proposed_quantity < 200, f"qty {decision.proposed_quantity} too large for 4h"


def test_1m_has_tighter_floor_than_4h():
    """1m minimum stop should be smaller than 4h."""
    sig_1m = _make_signal(timeframe="1m", entry=100.0, stop=99.99)
    sig_4h = _make_signal(timeframe="4h", entry=100.0, stop=99.99)
    dec_1m = evaluate_signal_risk(sig_1m, _make_account(), [], _make_config(), current_bar=0)
    dec_4h = evaluate_signal_risk(sig_4h, _make_account(), [], _make_config(), current_bar=0)
    # 1m floor is 0.2%, 4h floor is 1.0%, so 1m should have larger qty
    assert dec_1m.proposed_quantity > dec_4h.proposed_quantity


def test_normal_stop_not_floored():
    """A reasonable 2% stop on 4h should NOT be floored."""
    signal = _make_signal(timeframe="4h", entry=100.0, stop=98.0)  # 2% stop
    decision = evaluate_signal_risk(signal, _make_account(), [], _make_config(), current_bar=0)
    assert decision.approved
    # $100 risk / $2.00 stop = 50 units
    assert abs(decision.proposed_quantity - 50.0) < 1.0
