from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import StrategyConfig, calculate_atr, clamp
from .types import Trendline, ensure_candles_df


def score_line(candles, line: Trendline, config: StrategyConfig | None = None) -> Trendline:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    atr = calculate_atr(df, cfg.atr_period)

    if line.confirming_touch_indices:
        normalized_residuals = []
        for touch_index, residual in zip(line.confirming_touch_indices, line.residuals, strict=False):
            max_error = cfg.max_line_error(float(atr.iloc[touch_index]), float(df.iloc[touch_index]["close"]))
            normalized_residuals.append(residual / max(max_error, cfg.tick_size))
        normalized_mean_residual = float(np.mean(normalized_residuals))
    else:
        normalized_mean_residual = 1.0

    touch_score = min(line.confirming_touch_count / 5.0, 1.0)
    fit_score = clamp(1.0 - normalized_mean_residual)

    if len(line.confirming_touch_indices) >= 2:
        gaps = np.diff(line.confirming_touch_indices)
        mean_gap = float(np.mean(gaps))
        spacing_score = clamp(mean_gap / cfg.target_touch_gap)
    else:
        spacing_score = 0.0

    recency_score = clamp(1.0 - (line.bars_since_last_confirming_touch / max(cfg.max_fresh_bars, 1)))
    slope_score = 1.0 if cfg.min_slope_abs <= abs(line.slope) <= cfg.max_slope_abs else 0.0
    cleanliness_score = clamp(1.0 - (line.non_touch_cross_count / max(cfg.cleanliness_cross_cap, 1)))
    breakout_risk_penalty = clamp(line.recent_test_count / max(cfg.breakout_risk_test_cap, 1))

    weighted_score = (
        (0.25 * touch_score)
        + (0.20 * fit_score)
        + (0.15 * spacing_score)
        + (0.15 * recency_score)
        + (0.15 * slope_score)
        + (0.10 * cleanliness_score)
        - (0.20 * breakout_risk_penalty)
    )
    final_score = clamp(weighted_score) * 100.0

    state = line.state
    if line.invalidation_reason:
        state = "invalidated"
    elif line.bars_since_last_confirming_touch > cfg.max_fresh_bars:
        state = "expired"
    elif line.confirming_touch_count >= cfg.min_touches and final_score >= cfg.confirm_threshold:
        state = "confirmed"
    else:
        state = "candidate"

    return replace(
        line,
        state=state,
        score=final_score,
        score_components={
            "touch_score": touch_score,
            "fit_score": fit_score,
            "spacing_score": spacing_score,
            "recency_score": recency_score,
            "slope_score": slope_score,
            "cleanliness_score": cleanliness_score,
            "breakout_risk_penalty": breakout_risk_penalty,
            "normalized_mean_residual": normalized_mean_residual,
        },
    )


def score_lines(candles, lines: list[Trendline], config: StrategyConfig | None = None) -> list[Trendline]:
    cfg = config or StrategyConfig()
    return [score_line(candles, line, cfg) for line in lines]


__all__ = [
    "score_line",
    "score_lines",
]
