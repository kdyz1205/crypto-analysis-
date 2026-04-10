from __future__ import annotations

import pandas as pd

from server.execution import PaperExecutionConfig
from server.strategy.backtest import run_strategy_backtest, summarize_backtest_results
from server.strategy.replay import ReplaySnapshot
from server.strategy.types import StrategySignal


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": 1, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1.0},
            {"timestamp": 2, "open": 100.5, "high": 100.8, "low": 99.8, "close": 100.2, "volume": 1.0},
            {"timestamp": 3, "open": 100.2, "high": 100.6, "low": 97.8, "close": 98.2, "volume": 1.0},
        ]
    )


def _signal() -> StrategySignal:
    return StrategySignal(
        signal_id="sig-1",
        line_id="line-1",
        symbol="BTCUSDT",
        timeframe="1h",
        signal_type="rejection_short",
        direction="short",
        trigger_mode="rejection",
        timestamp=1,
        trigger_bar_index=0,
        score=0.8,
        priority_rank=1,
        entry_price=100.0,
        stop_price=101.0,
        tp_price=98.0,
        risk_reward=2.0,
        confirming_touch_count=3,
        bars_since_last_confirming_touch=0,
        distance_to_line=0.1,
        line_side="resistance",
        reason_code="test",
        source="auto",
    )


def test_run_strategy_backtest_collects_closed_positions(monkeypatch) -> None:
    signal = _signal()

    def fake_snapshot(candles_df, *args, **kwargs):
        current_index = len(candles_df) - 1
        return ReplaySnapshot(
            bar_index=current_index,
            timestamp=int(candles_df.iloc[current_index]["timestamp"]),
            pivots=tuple(),
            candidate_lines=tuple(),
            active_lines=tuple(),
            line_states=tuple(),
            signals=(signal,) if current_index == 0 else tuple(),
            signal_states=tuple(),
            invalidations=tuple(),
        )

    monkeypatch.setattr("server.strategy.backtest.build_latest_snapshot", fake_snapshot)
    monkeypatch.setattr("server.strategy.backtest.augment_snapshot_with_manual_signals", lambda snapshot, *args, **kwargs: snapshot)

    result = run_strategy_backtest(
        _candles(),
        execution_config=PaperExecutionConfig(starting_equity=1000.0, risk_per_trade=0.01),
        symbol="BTCUSDT",
        timeframe="1h",
    )

    assert result.trade_count == 1
    assert len(result.closed_positions) == 1
    assert result.closed_positions[0].exit_reason == "take_profit"
    assert result.total_pnl > 0
    assert result.signal_stats["auto:short:rejection"].wins == 1


def test_summarize_backtest_results_aggregates_markets(monkeypatch) -> None:
    signal = _signal()

    def fake_snapshot(candles_df, *args, **kwargs):
        current_index = len(candles_df) - 1
        return ReplaySnapshot(
            bar_index=current_index,
            timestamp=int(candles_df.iloc[current_index]["timestamp"]),
            pivots=tuple(),
            candidate_lines=tuple(),
            active_lines=tuple(),
            line_states=tuple(),
            signals=(signal,) if current_index == 0 else tuple(),
            signal_states=tuple(),
            invalidations=tuple(),
        )

    monkeypatch.setattr("server.strategy.backtest.build_latest_snapshot", fake_snapshot)
    monkeypatch.setattr("server.strategy.backtest.augment_snapshot_with_manual_signals", lambda snapshot, *args, **kwargs: snapshot)

    result_a = run_strategy_backtest(
        _candles(),
        execution_config=PaperExecutionConfig(starting_equity=1000.0, risk_per_trade=0.01),
        symbol="BTCUSDT",
        timeframe="1h",
    )
    result_b = run_strategy_backtest(
        _candles(),
        execution_config=PaperExecutionConfig(starting_equity=1000.0, risk_per_trade=0.01),
        symbol="ETHUSDT",
        timeframe="4h",
    )

    summary = summarize_backtest_results([result_a, result_b])

    assert summary["result_count"] == 2
    assert summary["trade_count"] == 2
    assert summary["total_pnl"] > 0
    assert "auto:short:rejection" in summary["signal_stats"]
