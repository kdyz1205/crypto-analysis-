from dataclasses import replace

import pandas as pd

import server.strategy.scoring as scoring_module
import server.strategy.trendlines as trendlines_module
from server.strategy.config import StrategyConfig, calculate_atr
from server.strategy.scoring import score_lines
from server.strategy.trendlines import build_candidate_lines, detect_trendlines
from server.strategy.types import Pivot, Trendline, stable_id


def _descending_candles() -> pd.DataFrame:
    close = [0.82, 0.84, 1.16, 0.90, 0.89, 0.88, 0.87, 0.86, 1.04, 0.84, 0.83, 0.98, 0.81, 0.80, 0.92, 0.79, 0.89, 0.78, 0.77, 0.76]
    high = [value + 0.03 for value in close]
    low = [value - 0.03 for value in close]
    open_ = [value - 0.01 for value in close]

    special_highs = {2: 1.20, 8: 1.08, 11: 1.02, 14: 0.96, 16: 0.92}
    for index, high_value in special_highs.items():
        high[index] = high_value
        open_[index] = high_value - 0.05
        close[index] = high_value - 0.04
        low[index] = high_value - 0.08

    return pd.DataFrame(
        {
            "timestamp": list(range(20)),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100] * 20,
        }
    )


def _descending_pivots() -> list[Pivot]:
    pivot_points = {
        2: 1.20,
        8: 1.08,
        11: 1.02,
        14: 0.96,
    }
    return [
        Pivot(
            pivot_id=stable_id("pivot", "high", index),
            kind="high",
            index=index,
            timestamp=index,
            price=price,
            left_bars=3,
            right_bars=3,
            confirmed_at_index=index + 3,
        )
        for index, price in pivot_points.items()
    ]


def _raw_line(line_id: str, *, score: float = 0.0, slope: float = -0.01, projected_price_current: float = 1.0) -> Trendline:
    return Trendline(
        line_id=line_id,
        side="resistance",
        symbol="TEST",
        timeframe="1h",
        state="candidate",
        anchor_pivot_ids=("a", "b"),
        confirming_touch_pivot_ids=("a", "b", "c"),
        anchor_indices=(1, 7),
        anchor_prices=(1.08, 1.02),
        slope=slope,
        intercept=1.10,
        confirming_touch_indices=(1, 4, 7),
        bar_touch_indices=(1, 4, 7, 8),
        confirming_touch_count=3,
        bar_touch_count=4,
        recent_bar_touch_count=2,
        residuals=(0.001, 0.002, 0.0015),
        score=score,
        score_components={"normalized_mean_residual": 0.1},
        projected_price_current=projected_price_current,
        projected_price_next=projected_price_current - 0.01,
        latest_confirming_touch_index=7,
        latest_confirming_touch_price=1.02,
        bars_since_last_confirming_touch=2,
        recent_test_count=1,
        non_touch_cross_count=0,
        invalidation_reason=None,
    )


def _invalidation_candles(*, break_distance: bool = False) -> pd.DataFrame:
    close = [1.00, 1.06, 0.99, 0.98, 1.00, 0.95, 0.94, 0.96, 0.93, 0.92, 0.93, 0.915]
    if break_distance:
        close[-1] = 1.10
    high = [value + 0.10 for value in close]
    low = [value - 0.10 for value in close]
    open_ = [value - 0.02 for value in close]
    if break_distance:
        # 2026-04-23: force the body to STRADDLE the projected line at bar
        # 11. Without this, open_[-1] = close_[-1] - 0.02 = 1.08, which is
        # itself above the line (~0.90) and triggers "body_break" before
        # "break_distance" ever gets checked in _detect_invalidation. We
        # want this test to specifically exercise the break_distance
        # branch, so pull open_[-1] down below the line.
        open_[-1] = 0.88
    high[1] = 1.10
    high[4] = 1.04
    high[7] = 0.98
    return pd.DataFrame(
        {
            "timestamp": list(range(len(close))),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100] * len(close),
        }
    )


def _invalidation_pivots() -> list[Pivot]:
    pivot_points = {1: 1.10, 4: 1.04, 7: 0.98}
    return [
        Pivot(
            pivot_id=stable_id("pivot", "high", index, price),
            kind="high",
            index=index,
            timestamp=index,
            price=price,
            left_bars=3,
            right_bars=3,
            confirmed_at_index=index + 3,
        )
        for index, price in pivot_points.items()
    ]


def _late_touch_after_break_candles() -> pd.DataFrame:
    close = [1.00, 1.10, 1.02, 1.00, 1.04, 1.08, 1.09, 1.01, 0.98, 0.97, 0.96]
    high = [1.03, 1.12, 1.05, 1.02, 1.04, 1.10, 1.11, 0.99, 0.98, 0.97, 0.96]
    low = [0.97, 1.04, 0.98, 0.97, 1.00, 1.04, 1.05, 0.95, 0.93, 0.92, 0.91]
    open_ = [0.99, 1.06, 1.01, 0.99, 1.02, 1.05, 1.06, 0.98, 0.95, 0.94, 0.93]
    return pd.DataFrame(
        {
            "timestamp": list(range(len(close))),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100] * len(close),
        }
    )


def _late_touch_after_break_pivots() -> list[Pivot]:
    pivot_points = {1: 1.12, 4: 1.04, 7: 0.99}
    return [
        Pivot(
            pivot_id=stable_id("pivot", "high", index, price),
            kind="high",
            index=index,
            timestamp=index,
            price=price,
            left_bars=2,
            right_bars=2,
            confirmed_at_index=index + 2,
        )
        for index, price in pivot_points.items()
    ]


def test_candidate_pruning_and_dedup_merge_similar_lines() -> None:
    candles = _descending_candles()
    pivots = _descending_pivots()
    config = StrategyConfig(
        max_candidate_lines_per_side=2,
        max_active_lines_per_side=1,
        max_anchor_combinations_per_pivot=3,
        max_fresh_bars=20,
    )

    lines = build_candidate_lines(candles, pivots, config, symbol="TEST", timeframe="1h")
    resistance_lines = [line for line in lines if line.side == "resistance"]

    assert len(resistance_lines) == 1
    assert resistance_lines[0].anchor_pivot_ids[0] in {pivot.pivot_id for pivot in pivots}


def test_confirming_touch_and_bar_touch_stay_separate() -> None:
    candles = _descending_candles()
    pivots = _descending_pivots()
    config = StrategyConfig(max_fresh_bars=20, min_touch_spacing_bars=1)

    result = detect_trendlines(candles, pivots, config, symbol="TEST", timeframe="1h")
    line = max(
        (line for line in result.candidate_lines if line.side == "resistance"),
        key=lambda item: (item.confirming_touch_count, item.bar_touch_count),
    )

    assert 16 not in line.confirming_touch_indices
    assert 16 in line.bar_touch_indices
    assert set(line.confirming_touch_indices).isdisjoint(line.bar_touch_indices)


def test_pre_score_pruning_happens_before_scoring(monkeypatch) -> None:
    recorded: list[int] = []
    raw_lines = [_raw_line(f"line-{index}", projected_price_current=1.0 + (index * 0.02)) for index in range(5)]

    def fake_build_side_candidates(*args, side: str, **kwargs):
        return raw_lines if side == "resistance" else []

    def fake_score_lines(candles, lines, config):
        recorded.append(len(lines))
        return list(lines)

    monkeypatch.setattr(trendlines_module, "_build_side_candidates", fake_build_side_candidates)
    monkeypatch.setattr(scoring_module, "score_lines", fake_score_lines)

    build_candidate_lines(_descending_candles(), _descending_pivots(), StrategyConfig(max_candidate_lines_per_side=2), symbol="TEST", timeframe="1h")

    assert recorded == [2, 0]


def test_dedup_keeps_higher_scored_similar_line(monkeypatch) -> None:
    low_score = _raw_line("line-low", score=60.0, slope=-0.01, projected_price_current=1.00)
    high_score = _raw_line("line-high", score=85.0, slope=-0.0102, projected_price_current=1.001)

    def fake_build_side_candidates(*args, side: str, **kwargs):
        return [low_score, high_score] if side == "resistance" else []

    def fake_score_lines(candles, lines, config):
        return list(lines)

    monkeypatch.setattr(trendlines_module, "_build_side_candidates", fake_build_side_candidates)
    monkeypatch.setattr(scoring_module, "score_lines", fake_score_lines)

    lines = build_candidate_lines(_descending_candles(), _descending_pivots(), StrategyConfig(), symbol="TEST", timeframe="1h")

    resistance_lines = [line for line in lines if line.side == "resistance"]
    assert [line.line_id for line in resistance_lines] == ["line-high"]


def test_break_close_count_invalidation_is_detected() -> None:
    # 2026-04-23: explicit config overrides because defaults have tightened
    # since these tests were written:
    #   - min_touch_spacing_bars=1 (default 5) — fixture pivots are 3 apart
    #   - max_non_touch_crosses=5 (default 0) — invalidation bar legitimately
    #     crosses the line (that's WHY it's invalidated), but the pre-score
    #     filter would reject it under the strict default
    # The test's purpose is to verify invalidation reason-setting, not those
    # unrelated tightening filters.
    lines = build_candidate_lines(
        _invalidation_candles(),
        _invalidation_pivots(),
        StrategyConfig(max_fresh_bars=20, min_touch_spacing_bars=1, max_non_touch_crosses=5),
        symbol="TEST",
        timeframe="1h",
    )

    invalidated = next(line for line in lines if line.side == "resistance")
    assert invalidated.state == "invalidated"
    assert invalidated.invalidation_reason == "break_close_count"


def test_break_distance_invalidation_is_detected() -> None:
    # 2026-04-23: also override break_close_count (default tightened to 1).
    # Without this, bar 10 (close=0.93 vs line=0.92) fires break_close_count
    # before bar 11 (close=1.10) ever triggers break_distance.
    lines = build_candidate_lines(
        _invalidation_candles(break_distance=True),
        _invalidation_pivots(),
        StrategyConfig(
            max_fresh_bars=20, min_touch_spacing_bars=1,
            max_non_touch_crosses=5, break_close_count=3,
        ),
        symbol="TEST",
        timeframe="1h",
    )

    invalidated = next(line for line in lines if line.side == "resistance")
    assert invalidated.state == "invalidated"
    assert invalidated.invalidation_reason == "break_distance"


def test_late_pivot_does_not_count_as_confirming_touch_after_break() -> None:
    candles = _late_touch_after_break_candles()
    pivots = _late_touch_after_break_pivots()
    config = StrategyConfig(
        pivot_left=2,
        pivot_right=2,
        max_fresh_bars=20,
        min_touch_spacing_bars=1,
        break_close_count=2,
        break_atr_mult=0.05,
        break_pct=0.001,
    )
    # _evaluate_candidate_line now takes a _BarArrays bundle (numpy views
    # of OHLC+ATR) instead of df+atr. Construct it from the fixture.
    arrays = trendlines_module._extract_bar_arrays(
        candles, calculate_atr(candles, config.atr_period)
    )
    line = trendlines_module._evaluate_candidate_line(
        arrays,
        pivots,
        pivots[0],
        pivots[1],
        side="resistance",
        config=config,
        symbol="TEST",
        timeframe="1h",
        current_index=len(candles) - 1,
    )

    assert line is not None
    assert line.state == "invalidated"
    assert line.invalidation_index is not None and line.invalidation_index < 7
    assert 7 not in line.confirming_touch_indices
    assert set(line.confirming_touch_indices) == {1, 4}
    assert all(index not in line.bar_touch_indices for index in line.confirming_touch_indices)
