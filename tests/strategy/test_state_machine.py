from dataclasses import replace

import pandas as pd

from server.strategy.config import StrategyConfig
from server.strategy.signals import generate_pre_limit_signals, prioritize_signals, resolve_signal_conflicts
from server.strategy.state_machine import (
    advance_line_states,
    build_signal_state_snapshots,
    close_line_state,
)
from server.strategy.types import StrategySignal, Trendline


def _confirmed_resistance_line() -> Trendline:
    return Trendline(
        line_id="res-line",
        side="resistance",
        symbol="TEST",
        timeframe="1h",
        state="confirmed",
        anchor_pivot_ids=("a", "b"),
        confirming_touch_pivot_ids=("a", "b", "c", "d"),
        anchor_indices=(0, 10),
        anchor_prices=(1.10, 1.00),
        slope=-0.01,
        intercept=1.10,
        confirming_touch_indices=(0, 4, 8, 10),
        bar_touch_indices=(0, 4, 8, 10),
        confirming_touch_count=4,
        bar_touch_count=4,
        recent_bar_touch_count=1,
        residuals=(0.002, 0.002, 0.002, 0.002),
        score=82.0,
        score_components={"normalized_mean_residual": 0.1},
        projected_price_current=0.99,
        projected_price_next=0.98,
        latest_confirming_touch_index=10,
        latest_confirming_touch_price=1.00,
        bars_since_last_confirming_touch=1,
        recent_test_count=1,
        non_touch_cross_count=0,
        invalidation_reason=None,
    )


def _far_from_line_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": list(range(12)),
            "open": [0.90] * 12,
            "high": [0.94] * 12,
            "low": [0.86] * 12,
            "close": [0.91] * 12,
            "volume": [100] * 12,
        }
    )


def _armed_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": list(range(12)),
            "open": [0.95] * 12,
            "high": [0.98] * 12,
            "low": [0.92] * 12,
            "close": [0.96] * 11 + [0.985],
            "volume": [100] * 12,
        }
    )


def test_line_state_progresses_confirmed_to_armed_to_triggered() -> None:
    config = StrategyConfig()
    line = _confirmed_resistance_line()

    confirmed_snapshot = advance_line_states(
        _far_from_line_candles(),
        [line],
        [],
        config,
    )[0]
    assert confirmed_snapshot.state == "confirmed"

    armed_snapshot = advance_line_states(
        _armed_candles(),
        [line],
        [],
        config,
        previous_states={line.line_id: confirmed_snapshot.state},
    )[0]
    assert armed_snapshot.state == "armed"

    selected_signals = generate_pre_limit_signals(_armed_candles(), [line], config)
    triggered_snapshot = advance_line_states(
        _armed_candles(),
        [line],
        selected_signals,
        config,
        previous_states={line.line_id: armed_snapshot.state},
    )[0]
    assert triggered_snapshot.state == "triggered"
    assert triggered_snapshot.signal_ids == tuple(signal.signal_id for signal in selected_signals)


def test_line_state_marks_invalidated_and_expired() -> None:
    config = StrategyConfig()
    line = _confirmed_resistance_line()

    invalidated = advance_line_states(
        _armed_candles(),
        [replace(line, state="invalidated", invalidation_reason="break_distance")],
        [],
        config,
    )[0]
    expired = advance_line_states(
        _armed_candles(),
        [replace(line, state="expired", bars_since_last_confirming_touch=120)],
        [],
        config,
    )[0]

    assert invalidated.state == "invalidated"
    assert invalidated.transition_reason == "break_distance"
    assert expired.state == "expired"
    assert expired.transition_reason == "max_fresh_bars"


def test_close_line_state_only_allows_triggered_to_closed() -> None:
    triggered = advance_line_states(
        _armed_candles(),
        [_confirmed_resistance_line()],
        generate_pre_limit_signals(_armed_candles(), [_confirmed_resistance_line()], StrategyConfig()),
        StrategyConfig(),
    )[0]

    closed = close_line_state(triggered, reason="completed")
    assert closed.state == "closed"
    assert closed.previous_state == "triggered"


def test_signal_state_snapshots_mark_selected_and_suppressed() -> None:
    signals = [
        StrategySignal(
            signal_id="selected",
            line_id="line-a",
            symbol="TEST",
            timeframe="1h",
            signal_type="REJECTION_SHORT",
            direction="short",
            trigger_mode="rejection",
            timestamp=1,
            trigger_bar_index=10,
            score=0.82,
            priority_rank=None,
            entry_price=1.0,
            stop_price=1.1,
            tp_price=0.8,
            risk_reward=2.0,
            confirming_touch_count=4,
            bars_since_last_confirming_touch=1,
            distance_to_line=0.01,
            line_side="resistance",
            reason_code="rejection_short",
        ),
        StrategySignal(
            signal_id="suppressed",
            line_id="line-b",
            symbol="TEST",
            timeframe="1h",
            signal_type="PRE_LIMIT_SHORT",
            direction="short",
            trigger_mode="pre_limit",
            timestamp=2,
            trigger_bar_index=10,
            score=0.70,
            priority_rank=None,
            entry_price=1.0,
            stop_price=1.1,
            tp_price=0.8,
            risk_reward=2.0,
            confirming_touch_count=3,
            bars_since_last_confirming_touch=2,
            distance_to_line=0.02,
            line_side="resistance",
            reason_code="pre_limit_short",
        ),
    ]

    prioritized = prioritize_signals(signals, StrategyConfig())
    selected = resolve_signal_conflicts(prioritized)
    snapshots = build_signal_state_snapshots(prioritized, selected)

    assert snapshots[0].state == "active"
    assert snapshots[1].state == "suppressed"
    assert snapshots[1].reason == "same_symbol_conflict"
