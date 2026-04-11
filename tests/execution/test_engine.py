import pandas as pd

from server.execution.engine import PaperExecutionEngine
from server.execution.types import PaperExecutionConfig, RiskDecision
from server.strategy.replay import ReplayResult, ReplaySnapshot
from server.strategy.types import StrategySignal


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            {"timestamp": 2, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        ]
    )


def _signal(signal_id: str, *, symbol: str = "BTCUSDT", direction: str = "short", entry_price: float = 100.0) -> StrategySignal:
    return StrategySignal(
        signal_id=signal_id,
        line_id=f"line-{signal_id}",
        symbol=symbol,
        timeframe="4h",
        signal_type="PRE_LIMIT_SHORT" if direction == "short" else "PRE_LIMIT_LONG",
        direction=direction,
        trigger_mode="pre_limit",
        timestamp=1,
        trigger_bar_index=0,
        score=0.8,
        priority_rank=1,
        entry_price=entry_price,
        stop_price=105.0 if direction == "short" else 95.0,
        tp_price=90.0 if direction == "short" else 110.0,
        risk_reward=2.0,
        confirming_touch_count=3,
        bars_since_last_confirming_touch=1,
        distance_to_line=0.1,
        line_side="resistance" if direction == "short" else "support",
        reason_code="test",
        factor_components={},
    )


def _empty_snapshot(bar_index: int, timestamp: int) -> ReplaySnapshot:
    return ReplaySnapshot(
        bar_index=bar_index,
        timestamp=timestamp,
        pivots=(),
        candidate_lines=(),
        active_lines=(),
        line_states=(),
        signals=(),
        signal_states=(),
        invalidations=(),
    )


def test_engine_pending_reservation_blocks_second_same_symbol_signal() -> None:
    engine = PaperExecutionEngine(PaperExecutionConfig())
    replay = ReplayResult(
        symbol="BTCUSDT",
        timeframe="4h",
        snapshots=(
            ReplaySnapshot(
                bar_index=0,
                timestamp=1,
                pivots=(),
                candidate_lines=(),
                active_lines=(),
                line_states=(),
                signals=(
                    _signal("sig-a"),
                    _signal("sig-b"),
                ),
                signal_states=(),
                invalidations=(),
            ),
        ),
    )

    result = engine.step("BTCUSDT", "1h", _candles().iloc[:1].reset_index(drop=True), replay, bar_index=0)
    intents = {intent.signal_id: intent for intent in result["state"].intents}

    assert result["state"].account.open_order_count == 1
    assert intents["sig-a"].status == "submitted"
    assert intents["sig-b"].status == "blocked"
    assert intents["sig-b"].reason == "max_positions_per_symbol_hit"


def test_engine_rejects_orphan_fill_when_second_manual_order_cannot_open() -> None:
    engine = PaperExecutionEngine(PaperExecutionConfig(allow_multiple_same_direction_per_symbol=False))
    signal_a = _signal("sig-a", direction="long")
    signal_b = _signal("sig-b", direction="long")
    decision = RiskDecision(
        signal_id="sig-a",
        approved=True,
        blocking_reason="",
        risk_amount=20.0,
        proposed_quantity=2.0,
        stop_distance=5.0,
        exposure_after_fill=200.0,
    )

    intent_a = engine.order_manager.create_order_intent_from_signal(signal_a, decision, engine.config, current_bar=0, current_ts=1)
    intent_b = engine.order_manager.create_order_intent_from_signal(
        signal_b,
        RiskDecision(
            signal_id="sig-b",
            approved=True,
            blocking_reason="",
            risk_amount=20.0,
            proposed_quantity=2.0,
            stop_distance=5.0,
            exposure_after_fill=400.0,
        ),
        engine.config,
        current_bar=0,
        current_ts=1,
    )
    engine.order_manager.submit_paper_order(intent_a)
    engine.order_manager.submit_paper_order(intent_b)

    replay = ReplayResult(
        symbol="BTCUSDT",
        timeframe="4h",
        snapshots=(
            _empty_snapshot(0, 1),
            _empty_snapshot(1, 2),
        ),
    )

    result = engine.step("BTCUSDT", "1h", _candles(), replay, bar_index=1)
    intents = {intent.signal_id: intent for intent in result["state"].intents}

    assert len(result["state"].open_positions) == 1
    assert len(result["state"].recent_fills) == 1
    assert intents["sig-a"].status == "filled"
    assert intents["sig-b"].status == "rejected"
    assert intents["sig-b"].reason == "same_symbol_direction_open_blocked"


def test_engine_step_supports_snapshot_offset() -> None:
    engine = PaperExecutionEngine(PaperExecutionConfig())
    engine.last_processed_bar_by_stream["BTCUSDT:1h"] = 0
    replay = ReplayResult(
        symbol="BTCUSDT",
        timeframe="4h",
        snapshots=(
            ReplaySnapshot(
                bar_index=1,
                timestamp=2,
                pivots=(),
                candidate_lines=(),
                active_lines=(),
                line_states=(),
                signals=(_signal("sig-offset"),),
                signal_states=(),
                invalidations=(),
            ),
        ),
    )

    result = engine.step(
        "BTCUSDT",
        "1h",
        _candles(),
        replay,
        bar_index=1,
        snapshot_offset=1,
    )

    assert result["processedBars"] == [1]
    assert result["lastProcessedBar"] == 1
    assert result["state"].account.open_order_count == 1
