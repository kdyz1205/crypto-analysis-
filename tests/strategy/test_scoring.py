from dataclasses import replace
import math

import pandas as pd

from server.strategy.config import StrategyConfig
from server.strategy.factors import calculate_resistance_short_score, calculate_support_long_score
from server.strategy.scoring import score_line
from server.strategy.types import Trendline


def _candles_for_scoring() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": list(range(10)),
            "open": [1.00, 1.01, 1.02, 1.03, 1.04, 1.02, 1.01, 1.00, 0.99, 0.98],
            "high": [1.05, 1.06, 1.07, 1.08, 1.09, 1.07, 1.06, 1.05, 1.04, 1.03],
            "low": [0.95, 0.96, 0.97, 0.98, 0.99, 0.97, 0.96, 0.95, 0.94, 0.93],
            "close": [1.02, 1.03, 1.04, 1.05, 1.06, 1.01, 1.00, 0.99, 0.98, 0.97],
            "volume": [100] * 10,
        }
    )


def _base_line(side: str = "resistance") -> Trendline:
    slope = -0.01 if side == "resistance" else 0.01
    intercept = 1.10 if side == "resistance" else 0.90
    return Trendline(
        line_id=f"{side}-line",
        side=side,
        symbol="TEST",
        timeframe="1h",
        state="candidate",
        anchor_pivot_ids=("a", "b"),
        confirming_touch_pivot_ids=("a", "b", "c"),
        anchor_indices=(1, 7),
        anchor_prices=(1.09, 1.03) if side == "resistance" else (0.91, 0.97),
        slope=slope,
        intercept=intercept,
        confirming_touch_indices=(1, 4, 7),
        bar_touch_indices=(1, 4, 7, 8),
        confirming_touch_count=3,
        bar_touch_count=4,
        recent_bar_touch_count=1,
        residuals=(0.004, 0.005, 0.004),
        score=0.0,
        projected_price_current=(slope * 9) + intercept,
        projected_price_next=(slope * 10) + intercept,
        latest_confirming_touch_index=7,
        latest_confirming_touch_price=1.03 if side == "resistance" else 0.97,
        bars_since_last_confirming_touch=2,
        recent_test_count=1,
        non_touch_cross_count=0,
        invalidation_reason=None,
    )


def test_line_score_matches_formula_components() -> None:
    candles = _candles_for_scoring()
    config = StrategyConfig()
    line = score_line(candles, _base_line(), config)

    components = line.score_components
    expected = max(
        0.0,
        min(
            1.0,
            (0.25 * components["touch_score"])
            + (0.20 * components["fit_score"])
            + (0.15 * components["spacing_score"])
            + (0.15 * components["recency_score"])
            + (0.15 * components["slope_score"])
            + (0.10 * components["cleanliness_score"])
            - (0.20 * components["breakout_risk_penalty"]),
        ),
    ) * 100.0

    assert line.state == "confirmed"
    assert math.isclose(line.score, expected, rel_tol=1e-9)


def test_factor_scores_are_clamped_and_explicit() -> None:
    candles = _candles_for_scoring()
    config = StrategyConfig()

    short_score, short_components = calculate_resistance_short_score(candles, score_line(candles, _base_line("resistance"), config), config)
    long_score, long_components = calculate_support_long_score(candles, score_line(candles, _base_line("support"), config), config)

    assert 0.0 <= short_score <= 1.0
    assert 0.0 <= long_score <= 1.0
    assert "VolumeConfirmation" in short_components
    assert 0.0 <= short_components["VolumeConfirmation"] <= 1.0
    assert 0.0 <= short_components["TrendContext"] <= 1.0
    assert short_components["ConfluenceScore"] == 0.0
    assert "VolumeConfirmation" in long_components


def test_factor_uses_dynamic_projected_next_from_bar_index() -> None:
    candles = _candles_for_scoring()
    config = StrategyConfig()
    stale_projection_line = replace(_base_line("resistance"), projected_price_next=999.0)

    score_at_bar_5, components = calculate_resistance_short_score(
        candles,
        score_line(candles, stale_projection_line, config),
        config,
        bar_index=5,
    )

    assert 0.0 <= score_at_bar_5 <= 1.0
    assert components["DistanceCompression"] < 1.0


def test_score_line_marks_expired_when_too_old() -> None:
    candles = _candles_for_scoring()
    config = StrategyConfig(max_fresh_bars=1)

    expired_line = score_line(candles, replace(_base_line("resistance"), bars_since_last_confirming_touch=5), config)

    assert expired_line.state == "expired"


def test_score_line_marks_invalidated_when_reason_present() -> None:
    candles = _candles_for_scoring()
    config = StrategyConfig()

    invalidated_line = score_line(candles, replace(_base_line("resistance"), invalidation_reason="break_distance"), config)

    assert invalidated_line.state == "invalidated"
