from __future__ import annotations

from .config import StrategyConfig, calculate_atr, clamp
from .types import Trendline, ensure_candles_df, project_price


def calculate_resistance_short_score(candles, line: Trendline, config: StrategyConfig | None = None, *, bar_index: int | None = None) -> tuple[float, dict[str, float]]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    current_index = len(df) - 1 if bar_index is None else bar_index
    atr = calculate_atr(df, cfg.atr_period)

    atr_value = float(atr.iloc[current_index])
    close_price = float(df.iloc[current_index]["close"])
    line_value = project_price(line.slope, line.intercept, current_index)
    projected_next = project_price(line.slope, line.intercept, current_index + 1)

    touch_strength = min(line.confirming_touch_count / 5.0, 1.0)
    fit_tightness = clamp(1.0 - float(line.score_components.get("normalized_mean_residual", 1.0)))

    upper_wick_ratio = _upper_wick_ratio(df, current_index, cfg)
    wick_score = clamp(upper_wick_ratio / cfg.rejection_wick_ratio_cap)
    rejection_close_norm = max(0.12 * atr_value, 0.0015 * close_price)
    close_reclaim_score = clamp((line_value - close_price) / max(rejection_close_norm, cfg.tick_size))
    rejection_strength = clamp((0.6 * wick_score) + (0.4 * close_reclaim_score))

    arm_distance = cfg.arm_distance(atr_value, close_price)
    distance_compression = clamp(1.0 - (abs(close_price - projected_next) / max(arm_distance, cfg.tick_size)))
    freshness_score = clamp(1.0 - (line.bars_since_last_confirming_touch / max(cfg.max_fresh_bars, 1)))
    breakout_risk = clamp(line.recent_test_count / max(cfg.breakout_risk_test_cap, 1))

    score = (
        (0.26 * touch_strength)
        + (0.20 * fit_tightness)
        + (0.26 * rejection_strength)
        + (0.16 * distance_compression)
        + (0.12 * freshness_score)
        - (0.15 * breakout_risk)
    )
    score = clamp(score)

    return score, {
        "TouchStrength": touch_strength,
        "FitTightness": fit_tightness,
        "RejectionStrength": rejection_strength,
        "DistanceCompression": distance_compression,
        "FreshnessScore": freshness_score,
        "BreakoutRisk": breakout_risk,
        "VolumeFailure": 0.0,
        "TrendContext": 0.0,
        "ConfluenceScore": 0.0,
    }


def calculate_support_long_score(candles, line: Trendline, config: StrategyConfig | None = None, *, bar_index: int | None = None) -> tuple[float, dict[str, float]]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    current_index = len(df) - 1 if bar_index is None else bar_index
    atr = calculate_atr(df, cfg.atr_period)

    atr_value = float(atr.iloc[current_index])
    close_price = float(df.iloc[current_index]["close"])
    line_value = project_price(line.slope, line.intercept, current_index)
    projected_next = project_price(line.slope, line.intercept, current_index + 1)

    touch_strength = min(line.confirming_touch_count / 5.0, 1.0)
    fit_tightness = clamp(1.0 - float(line.score_components.get("normalized_mean_residual", 1.0)))

    lower_wick_ratio = _lower_wick_ratio(df, current_index, cfg)
    wick_score = clamp(lower_wick_ratio / cfg.rejection_wick_ratio_cap)
    rejection_close_norm = max(0.12 * atr_value, 0.0015 * close_price)
    close_reclaim_score = clamp((close_price - line_value) / max(rejection_close_norm, cfg.tick_size))
    rejection_strength = clamp((0.6 * wick_score) + (0.4 * close_reclaim_score))

    arm_distance = cfg.arm_distance(atr_value, close_price)
    distance_compression = clamp(1.0 - (abs(close_price - projected_next) / max(arm_distance, cfg.tick_size)))
    freshness_score = clamp(1.0 - (line.bars_since_last_confirming_touch / max(cfg.max_fresh_bars, 1)))
    breakdown_risk = clamp(line.recent_test_count / max(cfg.breakout_risk_test_cap, 1))

    score = (
        (0.26 * touch_strength)
        + (0.20 * fit_tightness)
        + (0.26 * rejection_strength)
        + (0.16 * distance_compression)
        + (0.12 * freshness_score)
        - (0.15 * breakdown_risk)
    )
    score = clamp(score)

    return score, {
        "TouchStrength": touch_strength,
        "FitTightness": fit_tightness,
        "RejectionStrength": rejection_strength,
        "DistanceCompression": distance_compression,
        "FreshnessScore": freshness_score,
        "BreakdownRisk": breakdown_risk,
        "VolumeFailure": 0.0,
        "TrendContext": 0.0,
        "ConfluenceScore": 0.0,
    }


def _upper_wick_ratio(df, bar_index: int, config: StrategyConfig) -> float:
    high = float(df.iloc[bar_index]["high"])
    open_price = float(df.iloc[bar_index]["open"])
    close_price = float(df.iloc[bar_index]["close"])
    upper_wick = high - max(open_price, close_price)
    body = max(abs(close_price - open_price), config.min_body_unit)
    return upper_wick / body


def _lower_wick_ratio(df, bar_index: int, config: StrategyConfig) -> float:
    low = float(df.iloc[bar_index]["low"])
    open_price = float(df.iloc[bar_index]["open"])
    close_price = float(df.iloc[bar_index]["close"])
    lower_wick = min(open_price, close_price) - low
    body = max(abs(close_price - open_price), config.min_body_unit)
    return lower_wick / body


__all__ = [
    "calculate_resistance_short_score",
    "calculate_support_long_score",
]
