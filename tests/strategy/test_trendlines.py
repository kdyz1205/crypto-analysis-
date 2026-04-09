from dataclasses import replace

import pandas as pd

from server.strategy.config import StrategyConfig
from server.strategy.scoring import score_lines
from server.strategy.trendlines import build_candidate_lines, detect_trendlines
from server.strategy.types import Pivot, stable_id


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
    assert line.bar_touch_count > line.confirming_touch_count
