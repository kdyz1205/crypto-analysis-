from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
import pandas as pd


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def calculate_atr(candles: pd.DataFrame, period: int) -> pd.Series:
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    close = candles["close"].astype(float)
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def _default_timeframe_priority() -> dict[str, int]:
    return {
        "1m": 1,
        "5m": 2,
        "15m": 3,
        "1h": 4,
        "4h": 5,
        "1d": 6,
    }


def _default_trigger_mode_priority() -> dict[str, int]:
    return {
        "pre_limit": 1,
        "rejection": 2,
        "failed_breakout": 3,
    }


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    pivot_left: int = 3
    pivot_right: int = 3
    lookback_bars: int = 500
    atr_period: int = 14
    tick_size: float = 0.0001

    tol_atr_mult: float = 0.10
    tol_pct: float = 0.0015
    tick_mult: int = 3
    close_touch_slack_atr_mult: float = 0.05
    close_touch_slack_pct: float = 0.0005

    line_error_atr_mult: float = 0.12
    line_error_pct: float = 0.002
    min_touches: int = 3
    min_touch_spacing_bars: int = 5  # default, overridden by timeframe_touch_spacing
    confirm_threshold: float = 60.0

    max_anchor_combinations_per_pivot: int = 8
    max_candidate_lines_per_side: int = 50
    max_active_lines_per_side: int = 5
    display_active_lines_per_side: int = 3
    display_touch_limit_per_line: int = 5
    display_bar_touch_limit_per_line: int = 2
    display_invalidation_merge_bars: int = 6
    line_merge_slope_eps: float = 0.0005
    merge_price_atr_mult: float = 0.10
    merge_price_pct: float = 0.001
    max_non_touch_crosses: int = 0

    target_touch_gap: int = 12
    min_slope_abs: float = 0.0001
    max_slope_abs: float = 0.1000  # relaxed for volatile crypto
    cleanliness_cross_cap: int = 3
    recent_test_window_bars: int = 30
    breakout_risk_test_cap: int = 4

    score_threshold: float = 0.62
    arm_atr_mult: float = 0.20
    arm_pct: float = 0.003
    entry_buffer_atr_mult: float = 0.03
    entry_buffer_pct: float = 0.0005
    entry_tick_mult: int = 1

    rejection_close_buffer_atr_mult: float = 0.08
    rejection_close_buffer_pct: float = 0.001
    wick_ratio_threshold: float = 1.2
    rejection_wick_ratio_cap: float = 3.0

    break_tol_atr_mult: float = 0.10
    break_tol_pct: float = 0.001
    failed_break_close_buffer_atr_mult: float = 0.08
    failed_break_close_buffer_pct: float = 0.001
    trigger_buffer_atr_mult: float = 0.02

    stop_atr_mult: float = 0.12
    stop_pct: float = 0.0015
    stop_tick_mult: int = 2
    rr_target: float = 2.0

    break_close_count: int = 1
    break_atr_mult: float = 0.15
    break_pct: float = 0.002
    max_fresh_bars: int = 80

    # Horizontal zone detection
    zone_min_touches: int = 2  # min pivot touches to form a zone
    zone_eps_atr_mult: float = 0.4  # clustering radius as ATR multiple
    zone_eps_pct: float = 0.008  # clustering radius as price percentage

    min_body_unit: float = 1e-8
    timeframe_priority: Mapping[str, int] = field(default_factory=_default_timeframe_priority)
    trigger_mode_priority: Mapping[str, int] = field(default_factory=_default_trigger_mode_priority)

    # Volume confirmation
    volume_surge_threshold: float = 1.5  # current vol must be >= 1.5x avg
    volume_lookback_bars: int = 20  # bars to average volume over

    # Trend context
    trend_ema_period: int = 50  # EMA period for trend direction
    trend_weight: float = 0.10  # weight of trend context in factor score

    # Signal quality gates
    min_rr_ratio: float = 1.8  # minimum RR to accept signal (was 3.0 — too strict, filtered valid setups)
    min_profit_space_atr_mult: float = 1.0  # min distance to opposing zone in ATR units

    def tolerance(self, atr_value: float, close_price: float) -> float:
        return max(
            self.tol_atr_mult * atr_value,
            self.tol_pct * close_price,
            self.tick_size * self.tick_mult,
        )

    def close_touch_slack(self, atr_value: float, close_price: float) -> float:
        return max(
            self.close_touch_slack_atr_mult * atr_value,
            self.close_touch_slack_pct * close_price,
        )

    def max_line_error(self, atr_value: float, close_price: float) -> float:
        return max(
            self.line_error_atr_mult * atr_value,
            self.line_error_pct * close_price,
        )

    def line_merge_price_eps(self, atr_value: float, close_price: float) -> float:
        return max(
            self.merge_price_atr_mult * atr_value,
            self.merge_price_pct * close_price,
        )

    def arm_distance(self, atr_value: float, close_price: float) -> float:
        return max(
            self.arm_atr_mult * atr_value,
            self.arm_pct * close_price,
        )

    def entry_buffer(self, atr_value: float, close_price: float) -> float:
        return max(
            self.entry_buffer_atr_mult * atr_value,
            self.entry_buffer_pct * close_price,
            self.tick_size * self.entry_tick_mult,
        )

    def rejection_close_buffer(self, atr_value: float, close_price: float) -> float:
        return max(
            self.rejection_close_buffer_atr_mult * atr_value,
            self.rejection_close_buffer_pct * close_price,
        )

    def break_tolerance(self, atr_value: float, close_price: float) -> float:
        return max(
            self.break_tol_atr_mult * atr_value,
            self.break_tol_pct * close_price,
        )

    def failed_break_close_buffer(self, atr_value: float, close_price: float) -> float:
        return max(
            self.failed_break_close_buffer_atr_mult * atr_value,
            self.failed_break_close_buffer_pct * close_price,
        )

    def trigger_buffer(self, atr_value: float) -> float:
        return max(
            self.trigger_buffer_atr_mult * atr_value,
            self.tick_size,
        )

    def stop_buffer(self, atr_value: float, close_price: float) -> float:
        return max(
            self.stop_atr_mult * atr_value,
            self.stop_pct * close_price,
            self.tick_size * self.stop_tick_mult,
        )

    def break_distance(self, atr_value: float, close_price: float) -> float:
        return max(
            self.break_atr_mult * atr_value,
            self.break_pct * close_price,
        )

    def dynamic_rr_target(
        self,
        *,
        score: float = 0.5,
        trend_aligned: bool = False,
        zone_strength: float = 0.0,
    ) -> float:
        """Compute RR target dynamically based on signal quality.

        - High score / strong zone / trend-aligned → larger target (up to 5.0)
        - Low score / weak zone / counter-trend → smaller target (floor at 1.5)
        """
        base = self.rr_target  # default 2.0

        # Score adjustment: strong signals get wider targets
        if score > 0.75:
            base += 1.0
        elif score > 0.60:
            base += 0.5

        # Zone strength adjustment
        if zone_strength > 70:
            base += 1.0
        elif zone_strength > 50:
            base += 0.5

        # Trend alignment bonus
        if trend_aligned:
            base += 0.5

        return max(1.5, min(base, 5.0))

    def timeframe_rank(self, timeframe: str) -> int:
        if timeframe in self.timeframe_priority:
            return self.timeframe_priority[timeframe]

        try:
            unit = timeframe[-1].lower()
            value = int(timeframe[:-1])
        except (IndexError, ValueError):
            return 0

        if unit == "m":
            minutes = value
        elif unit == "h":
            minutes = value * 60
        elif unit == "d":
            minutes = value * 1440
        else:
            return 0

        ordered = sorted(self.timeframe_priority.items(), key=lambda item: item[1])
        for label, rank in ordered:
            if self._timeframe_minutes(label) == minutes:
                return rank
        return max(self.timeframe_priority.values(), default=0) + 1

    def trigger_rank(self, trigger_mode: str) -> int:
        return int(self.trigger_mode_priority.get(trigger_mode, 0))

    def pivot_window_for_timeframe(self, timeframe: str) -> int:
        """Shorter TFs need smaller pivot windows; longer TFs need wider."""
        windows = {"1m": 2, "3m": 2, "5m": 3, "15m": 3, "1h": 4, "4h": 5, "1d": 5, "1w": 7}
        return windows.get(timeframe, self.pivot_left)

    def touch_spacing_for_timeframe(self, timeframe: str) -> int:
        """Shorter TFs need smaller spacing; longer TFs need wider spacing."""
        spacing = {
            "1m": 2, "3m": 3, "5m": 3,
            "15m": 5, "1h": 8, "4h": 12,
            "1d": 15, "1w": 20,
        }
        return spacing.get(timeframe, self.min_touch_spacing_bars)

    def _timeframe_minutes(self, timeframe: str) -> int:
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1])
        if unit == "m":
            return value
        if unit == "h":
            return value * 60
        if unit == "d":
            return value * 1440
        return 0


def apply_strategy_overrides(config: StrategyConfig, **changes) -> StrategyConfig:
    filtered = {key: value for key, value in changes.items() if value is not None}
    if not filtered:
        return config
    return replace(config, **filtered)


__all__ = [
    "StrategyConfig",
    "apply_strategy_overrides",
    "calculate_atr",
    "clamp",
]
