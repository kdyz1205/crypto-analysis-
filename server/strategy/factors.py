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
    volume_score = _volume_confirmation(df, current_index, cfg.volume_lookback_bars, cfg.volume_surge_threshold)
    trend_score = _trend_context(df, current_index, "short", cfg.trend_ema_period)

    # Base score (original weights preserved for backward compatibility)
    base_score = (
        (0.26 * touch_strength)
        + (0.20 * fit_tightness)
        + (0.26 * rejection_strength)
        + (0.16 * distance_compression)
        + (0.12 * freshness_score)
        - (0.15 * breakout_risk)
    )
    # Volume and trend are additive bonuses (boost good setups, don't penalize neutral)
    bonus = (0.08 * volume_score) + (cfg.trend_weight * trend_score)
    score = clamp(base_score + bonus)

    return score, {
        "TouchStrength": touch_strength,
        "FitTightness": fit_tightness,
        "RejectionStrength": rejection_strength,
        "DistanceCompression": distance_compression,
        "FreshnessScore": freshness_score,
        "BreakoutRisk": breakout_risk,
        "VolumeConfirmation": volume_score,
        "TrendContext": trend_score,
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
    volume_score = _volume_confirmation(df, current_index, cfg.volume_lookback_bars, cfg.volume_surge_threshold)
    trend_score = _trend_context(df, current_index, "long", cfg.trend_ema_period)

    base_score = (
        (0.26 * touch_strength)
        + (0.20 * fit_tightness)
        + (0.26 * rejection_strength)
        + (0.16 * distance_compression)
        + (0.12 * freshness_score)
        - (0.15 * breakdown_risk)
    )
    bonus = (0.08 * volume_score) + (cfg.trend_weight * trend_score)
    score = clamp(base_score + bonus)

    return score, {
        "TouchStrength": touch_strength,
        "FitTightness": fit_tightness,
        "RejectionStrength": rejection_strength,
        "DistanceCompression": distance_compression,
        "FreshnessScore": freshness_score,
        "BreakdownRisk": breakdown_risk,
        "VolumeConfirmation": volume_score,
        "TrendContext": trend_score,
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


def _volume_confirmation(df, bar_index: int, lookback: int = 20, surge_threshold: float = 1.5) -> float:
    """Score 0-1 based on how much current bar volume exceeds the rolling average."""
    start = max(0, bar_index - lookback)
    if start >= bar_index:
        return 0.0
    avg_vol = float(df["volume"].iloc[start:bar_index].mean())
    if avg_vol <= 0:
        return 0.0
    current_vol = float(df["volume"].iloc[bar_index])
    ratio = current_vol / avg_vol
    if ratio <= 1.0:
        return 0.0
    band = max(surge_threshold - 1.0, 1e-9)
    return clamp(((ratio - 1.0) / band) ** 2 / 5.0)


def _trend_context(df, bar_index: int, direction: str, ema_period: int = 50) -> float:
    """Score 0-1 for trend alignment. Long scores higher in uptrends, short in downtrends."""
    if bar_index < ema_period:
        return 0.3

    close = df["close"].astype(float)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    current_close = float(close.iloc[bar_index])
    current_ema = float(ema.iloc[bar_index])
    prev_ema = float(ema.iloc[bar_index - 1])

    ema_slope = (current_ema - prev_ema) / max(abs(current_ema), 1e-10)
    price_vs_ema = (current_close - current_ema) / max(abs(current_ema), 1e-10)

    if direction == "long":
        slope_score = clamp(ema_slope * 1000)
        position_score = clamp(price_vs_ema * 50)
    else:
        slope_score = clamp(-ema_slope * 1000)
        position_score = clamp(-price_vs_ema * 50)

    return clamp(0.5 * slope_score + 0.5 * position_score)


__all__ = [
    "calculate_resistance_short_score",
    "calculate_support_long_score",
]
