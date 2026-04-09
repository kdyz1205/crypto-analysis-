import pandas as pd

from server.strategy.config import StrategyConfig
from server.strategy.signals import (
    generate_failed_breakout_signals,
    generate_pre_limit_signals,
    generate_rejection_signals,
    prioritize_signals,
    resolve_signal_conflicts,
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


def _pre_limit_candles() -> pd.DataFrame:
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


def _rejection_candles() -> pd.DataFrame:
    closes = [0.95] * 11 + [0.93]
    highs = [0.98] * 11 + [1.00]
    opens = [0.95] * 11 + [0.94]
    lows = [0.92] * 12
    return pd.DataFrame(
        {
            "timestamp": list(range(12)),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100] * 12,
        }
    )


def _failed_breakout_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": list(range(12)),
            "open": [0.95] * 10 + [0.99, 0.94],
            "high": [0.98] * 10 + [1.02, 0.97],
            "low": [0.92] * 10 + [0.94, 0.90],
            "close": [0.95] * 10 + [0.95, 0.92],
            "volume": [100] * 12,
        }
    )


def test_generate_pre_limit_signal() -> None:
    signals = generate_pre_limit_signals(_pre_limit_candles(), [_confirmed_resistance_line()], StrategyConfig())
    assert [signal.signal_type for signal in signals] == ["PRE_LIMIT_SHORT"]


def test_generate_rejection_signal() -> None:
    signals = generate_rejection_signals(_rejection_candles(), [_confirmed_resistance_line()], StrategyConfig())
    assert [signal.signal_type for signal in signals] == ["REJECTION_SHORT"]


def test_generate_failed_breakout_signal() -> None:
    signals = generate_failed_breakout_signals(_failed_breakout_candles(), [_confirmed_resistance_line()], StrategyConfig())
    assert [signal.signal_type for signal in signals] == ["FAILED_BREAKOUT_SHORT"]


def test_prioritize_and_resolve_same_symbol_conflicts() -> None:
    signals = [
        StrategySignal(
            signal_id="low",
            line_id="line-low",
            symbol="TEST",
            timeframe="1h",
            signal_type="PRE_LIMIT_SHORT",
            direction="short",
            trigger_mode="pre_limit",
            timestamp=1,
            trigger_bar_index=10,
            score=0.70,
            priority_rank=None,
            entry_price=1.0,
            stop_price=1.1,
            tp_price=0.8,
            risk_reward=2.0,
            confirming_touch_count=3,
            bars_since_last_confirming_touch=3,
            distance_to_line=0.02,
            line_side="resistance",
            reason_code="pre_limit_short",
        ),
        StrategySignal(
            signal_id="high",
            line_id="line-high",
            symbol="TEST",
            timeframe="4h",
            signal_type="REJECTION_SHORT",
            direction="short",
            trigger_mode="rejection",
            timestamp=2,
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
            signal_id="other",
            line_id="line-other",
            symbol="ALT",
            timeframe="1h",
            signal_type="FAILED_BREAKOUT_SHORT",
            direction="short",
            trigger_mode="failed_breakout",
            timestamp=3,
            trigger_bar_index=10,
            score=0.81,
            priority_rank=None,
            entry_price=1.0,
            stop_price=1.1,
            tp_price=0.8,
            risk_reward=2.0,
            confirming_touch_count=4,
            bars_since_last_confirming_touch=2,
            distance_to_line=0.01,
            line_side="resistance",
            reason_code="failed_breakout_short",
        ),
    ]

    prioritized = prioritize_signals(signals, StrategyConfig())
    assert prioritized[0].signal_id == "high"
    assert prioritized[0].priority_rank == 1

    resolved = resolve_signal_conflicts(prioritized)
    assert [signal.signal_id for signal in resolved] == ["high", "other"]
