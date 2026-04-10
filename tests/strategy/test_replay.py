import pandas as pd

import server.strategy.replay as replay_module
from server.strategy.config import StrategyConfig
from server.strategy.replay import build_latest_snapshot, build_tail_snapshots, replay_strategy
from server.strategy.types import Trendline, TrendlineDetectionResult


def _simple_candles(length: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": list(range(length)),
            "open": [1.0] * length,
            "high": [1.05] * length,
            "low": [0.95] * length,
            "close": [1.0] * length,
            "volume": [100] * length,
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


def test_replay_uses_only_prefix_data_without_lookahead(monkeypatch) -> None:
    pivot_lengths: list[int] = []
    trendline_lengths: list[int] = []

    def fake_detect_pivots(candles, config):
        pivot_lengths.append(len(candles))
        return []

    def fake_detect_trendlines(candles, pivots, config, **kwargs):
        trendline_lengths.append(len(candles))
        return TrendlineDetectionResult(candidate_lines=(), active_lines=())

    monkeypatch.setattr(replay_module, "detect_pivots", fake_detect_pivots)
    monkeypatch.setattr(replay_module, "detect_trendlines", fake_detect_trendlines)

    result = replay_strategy(_simple_candles(), StrategyConfig(), symbol="TEST", timeframe="1h")

    assert len(result.snapshots) == 5
    assert pivot_lengths == [1, 2, 3, 4, 5]
    assert trendline_lengths == [1, 2, 3, 4, 5]


def test_replay_failed_breakout_is_confirmed_on_next_bar(monkeypatch) -> None:
    line = _confirmed_resistance_line()

    monkeypatch.setattr(replay_module, "detect_pivots", lambda candles, config: [])
    monkeypatch.setattr(
        replay_module,
        "detect_trendlines",
        lambda candles, pivots, config, **kwargs: TrendlineDetectionResult(
            candidate_lines=(line,),
            active_lines=(line,),
        ),
    )

    result = replay_strategy(
        _failed_breakout_candles(),
        StrategyConfig(),
        symbol="TEST",
        timeframe="1h",
        enabled_trigger_modes=("failed_breakout",),
    )

    assert result.snapshots[10].signals == ()
    assert [signal.signal_type for signal in result.snapshots[11].signals] == ["FAILED_BREAKOUT_SHORT"]


def test_replay_snapshot_is_serializable_and_field_stable(monkeypatch) -> None:
    line = _confirmed_resistance_line()

    monkeypatch.setattr(replay_module, "detect_pivots", lambda candles, config: [])
    monkeypatch.setattr(
        replay_module,
        "detect_trendlines",
        lambda candles, pivots, config, **kwargs: TrendlineDetectionResult(
            candidate_lines=(line,),
            active_lines=(line,),
        ),
    )

    result = replay_strategy(
        _simple_candles(3),
        StrategyConfig(),
        symbol="TEST",
        timeframe="1h",
        enabled_trigger_modes=(),
    )
    snapshot = result.snapshots[-1].to_dict()

    assert {
        "bar_index",
        "timestamp",
        "pivots",
        "candidate_lines",
        "active_lines",
        "line_states",
        "signals",
        "signal_states",
        "invalidations",
    }.issubset(snapshot.keys())
    assert isinstance(snapshot["candidate_lines"], list)
    assert isinstance(snapshot["active_lines"], list)
    assert isinstance(snapshot["signals"], list)


def test_replay_exposes_invalidations(monkeypatch) -> None:
    invalidated_line = Trendline(
        line_id="invalidated-line",
        side="resistance",
        symbol="TEST",
        timeframe="1h",
        state="invalidated",
        anchor_pivot_ids=("a", "b"),
        confirming_touch_pivot_ids=("a", "b", "c"),
        anchor_indices=(0, 5),
        anchor_prices=(1.1, 1.0),
        slope=-0.02,
        intercept=1.1,
        confirming_touch_indices=(0, 2, 5),
        bar_touch_indices=(0, 2, 5),
        confirming_touch_count=3,
        bar_touch_count=3,
        recent_bar_touch_count=1,
        residuals=(0.001, 0.002, 0.001),
        score=61.0,
        score_components={"normalized_mean_residual": 0.1},
        projected_price_current=0.98,
        projected_price_next=0.96,
        latest_confirming_touch_index=5,
        latest_confirming_touch_price=1.0,
        bars_since_last_confirming_touch=1,
        recent_test_count=1,
        non_touch_cross_count=0,
        invalidation_reason="break_distance",
    )

    monkeypatch.setattr(replay_module, "detect_pivots", lambda candles, config: [])
    monkeypatch.setattr(
        replay_module,
        "detect_trendlines",
        lambda candles, pivots, config, **kwargs: TrendlineDetectionResult(
            candidate_lines=(invalidated_line,),
            active_lines=(),
        ),
    )

    result = replay_strategy(_simple_candles(2), StrategyConfig(), symbol="TEST", timeframe="1h")
    latest_snapshot = result.snapshots[-1]

    assert len(latest_snapshot.invalidations) == 1
    assert latest_snapshot.invalidations[0].state == "invalidated"


def test_build_latest_snapshot_matches_replay_latest_core_fields(monkeypatch) -> None:
    line = _confirmed_resistance_line()

    monkeypatch.setattr(replay_module, "detect_pivots", lambda candles, config: [])
    monkeypatch.setattr(
        replay_module,
        "detect_trendlines",
        lambda candles, pivots, config, **kwargs: TrendlineDetectionResult(
            candidate_lines=(line,),
            active_lines=(line,),
        ),
    )

    candles = _simple_candles(4)
    full_latest = replay_strategy(
        candles,
        StrategyConfig(),
        symbol="TEST",
        timeframe="1h",
        enabled_trigger_modes=(),
    ).snapshots[-1]
    latest_only = build_latest_snapshot(
        candles,
        StrategyConfig(),
        symbol="TEST",
        timeframe="1h",
        enabled_trigger_modes=(),
    )

    assert latest_only.bar_index == full_latest.bar_index
    assert latest_only.timestamp == full_latest.timestamp
    assert latest_only.pivots == full_latest.pivots
    assert latest_only.candidate_lines == full_latest.candidate_lines
    assert [state.line_id for state in latest_only.active_lines] == [state.line_id for state in full_latest.active_lines]
    assert [state.state for state in latest_only.active_lines] == [state.state for state in full_latest.active_lines]
    assert latest_only.signals == full_latest.signals
    assert latest_only.signal_states == full_latest.signal_states
    assert latest_only.invalidations == full_latest.invalidations
    assert [state.state for state in latest_only.line_states] == [state.state for state in full_latest.line_states]


def test_build_tail_snapshots_matches_replay_latest_core_fields(monkeypatch) -> None:
    line = _confirmed_resistance_line()

    monkeypatch.setattr(replay_module, "detect_pivots", lambda candles, config: [])
    monkeypatch.setattr(
        replay_module,
        "detect_trendlines",
        lambda candles, pivots, config, **kwargs: TrendlineDetectionResult(
            candidate_lines=(line,),
            active_lines=(line,),
        ),
    )

    candles = _simple_candles(6)
    full_tail = replay_strategy(
        candles,
        StrategyConfig(),
        symbol="TEST",
        timeframe="1h",
        enabled_trigger_modes=(),
    ).snapshots[-2:]
    fast_tail = build_tail_snapshots(
        candles,
        StrategyConfig(),
        symbol="TEST",
        timeframe="1h",
        tail=2,
        enabled_trigger_modes=(),
    )

    assert len(fast_tail) == 2
    for full_snapshot, fast_snapshot in zip(full_tail, fast_tail):
        assert fast_snapshot.bar_index == full_snapshot.bar_index
        assert fast_snapshot.pivots == full_snapshot.pivots
        assert fast_snapshot.candidate_lines == full_snapshot.candidate_lines
        assert [state.line_id for state in fast_snapshot.active_lines] == [state.line_id for state in full_snapshot.active_lines]
        assert [state.state for state in fast_snapshot.active_lines] == [state.state for state in full_snapshot.active_lines]
        assert fast_snapshot.signals == full_snapshot.signals
        assert fast_snapshot.signal_states == full_snapshot.signal_states
        assert fast_snapshot.invalidations == full_snapshot.invalidations
        assert [state.state for state in fast_snapshot.line_states] == [state.state for state in full_snapshot.line_states]
