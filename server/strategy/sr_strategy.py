"""SR Strategy System — 4-layer architecture.

Layer 1: SR Engine (existing zones.py) → zones + stats
Layer 2: Context → trend, volatility, distance to zones
Layer 3: Trigger → fake break reclaim, breakout retest, zone rejection
Layer 4: Execution → entry/stop/tp/size/log

Two strategies:
A. Multi-TF Confluence + Fake Breakout Reclaim
B. True Breakout + Retest Confirmation
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import StrategyConfig, calculate_atr, clamp
from .zones import HorizontalZone, detect_horizontal_zones
from .pivots import detect_pivots
from .regime import detect_regime, MarketRegime


# ── Parameters (v1 fixed, optimize later) ────────────────────────────────

ZONE_SCORE_MIN = 0.40
RESPECT_MIN = 0.6
FRESHNESS_MIN = 0.3
DIST_TO_ZONE_MAX_ATR = 0.5
FAKE_BREAK_MAX_ATR = 0.6
CONFIRM_BARS = 2
RISK_PER_TRADE = 0.01
MIN_RR = 1.8
FORWARD_WINDOW = 10


# ── Layer 2: Context ─────────────────────────────────────────────────────

@dataclass
class ZoneContext:
    """Extended zone info for strategy consumption."""
    zone_id: str
    zone_type: Literal["support", "resistance", "flip"]
    z_low: float
    z_high: float
    z_center: float
    score: float
    touch_count: int
    respect_rate: float
    avg_reaction_atr: float
    avg_failure_atr: float
    freshness: float
    last_touch_age: int


@dataclass
class MarketContext:
    """Full market context for strategy decisions."""
    symbol: str
    timeframe: str
    current_price: float
    atr: float
    trend_state: Literal["up", "down", "range"]
    trend_strength: float
    zones: list[ZoneContext]
    nearest_support: ZoneContext | None
    nearest_resistance: ZoneContext | None
    distance_to_nearest_support_atr: float
    distance_to_nearest_resistance_atr: float


def build_market_context(
    df: pd.DataFrame,
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
) -> MarketContext:
    """Build complete market context from candle data."""
    cfg = config or StrategyConfig()
    current_index = len(df) - 1
    atr_series = calculate_atr(df, cfg.atr_period)
    atr_value = float(atr_series.iloc[current_index])
    close_price = float(df.iloc[current_index]["close"])

    # Detect regime
    regime = detect_regime(df, cfg)

    # Map regime to simple trend state
    if regime.trend_strength > 0.5:
        trend_state = "up" if regime.trend_direction == "up" else "down"
    else:
        trend_state = "range"

    # Detect zones
    pivots = tuple(detect_pivots(df, cfg))
    raw_zones = detect_horizontal_zones(df, pivots, cfg, symbol=symbol, timeframe=timeframe, max_zones_per_side=5)

    # Build extended zone contexts with respect_rate and reaction stats
    zone_contexts = []
    for z in raw_zones:
        respect, avg_react, avg_fail = _compute_zone_stats(df, z, atr_series, FORWARD_WINDOW)
        freshness = clamp(1.0 - (current_index - z.last_touch_index) / max(cfg.max_fresh_bars, 1))
        zone_contexts.append(ZoneContext(
            zone_id=z.zone_id,
            zone_type=z.side,
            z_low=z.price_low,
            z_high=z.price_high,
            z_center=z.price_center,
            score=z.strength / 100.0,
            touch_count=z.touches,
            respect_rate=respect,
            avg_reaction_atr=avg_react,
            avg_failure_atr=avg_fail,
            freshness=freshness,
            last_touch_age=current_index - z.last_touch_index,
        ))

    # Find nearest support/resistance
    supports = [z for z in zone_contexts if z.zone_type == "support" and z.z_center < close_price]
    resistances = [z for z in zone_contexts if z.zone_type == "resistance" and z.z_center > close_price]
    nearest_sup = min(supports, key=lambda z: close_price - z.z_center) if supports else None
    nearest_res = min(resistances, key=lambda z: z.z_center - close_price) if resistances else None

    dist_sup = (close_price - nearest_sup.z_center) / atr_value if nearest_sup and atr_value > 0 else 999
    dist_res = (nearest_res.z_center - close_price) / atr_value if nearest_res and atr_value > 0 else 999

    return MarketContext(
        symbol=symbol,
        timeframe=timeframe,
        current_price=close_price,
        atr=atr_value,
        trend_state=trend_state,
        trend_strength=regime.trend_strength,
        zones=zone_contexts,
        nearest_support=nearest_sup,
        nearest_resistance=nearest_res,
        distance_to_nearest_support_atr=dist_sup,
        distance_to_nearest_resistance_atr=dist_res,
    )


def _compute_zone_stats(df: pd.DataFrame, zone: HorizontalZone, atr: pd.Series, forward: int) -> tuple[float, float, float]:
    """Compute respect_rate, avg_reaction_atr, avg_failure_atr for a zone."""
    n = len(df)
    reactions = []
    failures = []

    for touch_idx in zone.touch_indices:
        if touch_idx + forward >= n:
            continue
        local_atr = float(atr.iloc[touch_idx]) if touch_idx < len(atr) else 1.0
        if local_atr <= 0:
            local_atr = 1.0

        touch_close = float(df.iloc[touch_idx]["close"])

        # Look forward to see if zone held
        future_closes = [float(df.iloc[min(touch_idx + j, n - 1)]["close"]) for j in range(1, forward + 1)]
        future_lows = [float(df.iloc[min(touch_idx + j, n - 1)]["low"]) for j in range(1, forward + 1)]
        future_highs = [float(df.iloc[min(touch_idx + j, n - 1)]["high"]) for j in range(1, forward + 1)]

        if zone.side == "support":
            # Did it bounce? Max upside after touch
            max_up = max(future_highs) - touch_close
            max_down = touch_close - min(future_lows)
            reaction_atr = max_up / local_atr
            failure_atr = max_down / local_atr
            held = max_up > max_down  # more upside than downside = held
        else:
            # Did it reject? Max downside after touch
            max_down = touch_close - min(future_lows)
            max_up = max(future_highs) - touch_close
            reaction_atr = max_down / local_atr
            failure_atr = max_up / local_atr
            held = max_down > max_up

        if held:
            reactions.append(reaction_atr)
        else:
            failures.append(failure_atr)

    total = len(reactions) + len(failures)
    respect_rate = len(reactions) / total if total > 0 else 0.5
    avg_react = float(np.mean(reactions)) if reactions else 0.0
    avg_fail = float(np.mean(failures)) if failures else 0.0

    return round(respect_rate, 3), round(avg_react, 3), round(avg_fail, 3)


# ── Layer 3: Triggers ────────────────────────────────────────────────────

@dataclass
class StrategyDecision:
    """Output of strategy decision — always produced, even for no-trade."""
    timestamp: Any
    symbol: str
    timeframe: str
    trend_state: str
    atr: float
    # Zone info
    zone_id: str = ""
    zone_type: str = ""
    z_low: float = 0.0
    z_high: float = 0.0
    zone_score: float = 0.0
    respect_rate: float = 0.0
    freshness: float = 0.0
    distance_to_zone_atr: float = 0.0
    # Behavior detected
    touched: bool = False
    fake_break_detected: bool = False
    breakout_detected: bool = False
    reclaim_detected: bool = False
    confirmation_passed: bool = False
    # Decision
    signal: Literal["LONG", "SHORT", "NONE"] = "NONE"
    reason: str = "no_signal"
    # Trade params
    entry: float = 0.0
    stop: float = 0.0
    tp: float = 0.0
    rr: float = 0.0
    position_size: float = 0.0
    # Execution
    order_status: Literal["SUBMITTED", "SKIPPED"] = "SKIPPED"
    skip_reason: str = ""


def evaluate_bar(
    df: pd.DataFrame,
    ctx: MarketContext,
    equity: float = 10000.0,
) -> StrategyDecision:
    """Main strategy loop — evaluate current bar for trade signals."""
    n = len(df)
    current_index = n - 1
    decision = StrategyDecision(
        timestamp=df.iloc[current_index].get("timestamp", current_index),
        symbol=ctx.symbol,
        timeframe=ctx.timeframe,
        trend_state=ctx.trend_state,
        atr=ctx.atr,
    )

    if n < 5 or ctx.atr <= 0:
        decision.skip_reason = "insufficient_data"
        return decision

    # Check each high-quality zone
    candidate_zones = [
        z for z in ctx.zones
        if z.score >= ZONE_SCORE_MIN
        and z.respect_rate >= RESPECT_MIN
        and z.freshness >= FRESHNESS_MIN
    ]

    if not candidate_zones:
        decision.reason = "no_qualified_zones"
        return decision

    for zone in candidate_zones:
        dist_atr = abs(ctx.current_price - zone.z_center) / ctx.atr
        decision.zone_id = zone.zone_id
        decision.zone_type = zone.zone_type
        decision.z_low = zone.z_low
        decision.z_high = zone.z_high
        decision.zone_score = zone.score
        decision.respect_rate = zone.respect_rate
        decision.freshness = zone.freshness
        decision.distance_to_zone_atr = dist_atr

        # Not in zone range
        if dist_atr > DIST_TO_ZONE_MAX_ATR:
            continue

        decision.touched = True

        # ── Strategy A: Fake Break Reclaim ────────────────────────
        fake_signal = _detect_fake_break_reclaim(df, zone, ctx)
        if fake_signal:
            decision.fake_break_detected = True
            decision.reclaim_detected = True
            decision.signal = fake_signal["direction"]
            decision.reason = "fake_break_reclaim"
            decision.entry = fake_signal["entry"]
            decision.stop = fake_signal["stop"]
            decision.tp = fake_signal["tp"]
            decision.rr = fake_signal["rr"]

            # Trend filter: don't trade counter-trend in strong trends
            if ctx.trend_state == "up" and decision.signal == "SHORT" and ctx.trend_strength > 0.6:
                decision.order_status = "SKIPPED"
                decision.skip_reason = "counter_trend"
                return decision
            if ctx.trend_state == "down" and decision.signal == "LONG" and ctx.trend_strength > 0.6:
                decision.order_status = "SKIPPED"
                decision.skip_reason = "counter_trend"
                return decision

            if decision.rr < MIN_RR:
                decision.order_status = "SKIPPED"
                decision.skip_reason = f"rr_too_low_{decision.rr:.1f}"
                return decision

            # Position size
            risk_usd = equity * RISK_PER_TRADE
            risk_per_unit = abs(decision.entry - decision.stop)
            decision.position_size = risk_usd / risk_per_unit if risk_per_unit > 0 else 0
            decision.order_status = "SUBMITTED"
            return decision

        # ── Strategy B: Breakout Retest ───────────────────────────
        retest_signal = _detect_breakout_retest(df, zone, ctx)
        if retest_signal:
            decision.breakout_detected = True
            decision.confirmation_passed = True
            decision.signal = retest_signal["direction"]
            decision.reason = "breakout_retest"
            decision.entry = retest_signal["entry"]
            decision.stop = retest_signal["stop"]
            decision.tp = retest_signal["tp"]
            decision.rr = retest_signal["rr"]

            if decision.rr < MIN_RR:
                decision.order_status = "SKIPPED"
                decision.skip_reason = f"rr_too_low_{decision.rr:.1f}"
                return decision

            risk_usd = equity * RISK_PER_TRADE
            risk_per_unit = abs(decision.entry - decision.stop)
            decision.position_size = risk_usd / risk_per_unit if risk_per_unit > 0 else 0
            decision.order_status = "SUBMITTED"
            return decision

    decision.reason = "no_trigger"
    return decision


def _detect_fake_break_reclaim(df: pd.DataFrame, zone: ZoneContext, ctx: MarketContext) -> dict | None:
    """Detect fake breakout and reclaim pattern."""
    n = len(df)
    if n < CONFIRM_BARS + 2:
        return None

    atr = ctx.atr
    current = df.iloc[-1]
    close = float(current["close"])

    if zone.zone_type == "support":
        # Look for fake breakdown: recent low pierced z_low, but close came back above
        for lookback in range(1, CONFIRM_BARS + 2):
            bar = df.iloc[-(lookback + 1)]
            bar_low = float(bar["low"])
            if bar_low < zone.z_low:
                pierce_depth = (zone.z_low - bar_low) / atr
                if pierce_depth <= FAKE_BREAK_MAX_ATR:
                    # Check reclaim: current close above z_low
                    if close > zone.z_low:
                        entry = close
                        stop = bar_low - 0.2 * atr
                        risk = entry - stop
                        tp = entry + MIN_RR * risk
                        rr = (tp - entry) / risk if risk > 0 else 0
                        return {"direction": "LONG", "entry": entry, "stop": stop, "tp": tp, "rr": rr}

    elif zone.zone_type == "resistance":
        for lookback in range(1, CONFIRM_BARS + 2):
            bar = df.iloc[-(lookback + 1)]
            bar_high = float(bar["high"])
            if bar_high > zone.z_high:
                pierce_depth = (bar_high - zone.z_high) / atr
                if pierce_depth <= FAKE_BREAK_MAX_ATR:
                    if close < zone.z_high:
                        entry = close
                        stop = bar_high + 0.2 * atr
                        risk = stop - entry
                        tp = entry - MIN_RR * risk
                        rr = (entry - tp) / risk if risk > 0 else 0
                        return {"direction": "SHORT", "entry": entry, "stop": stop, "tp": tp, "rr": rr}

    return None


def _detect_breakout_retest(df: pd.DataFrame, zone: ZoneContext, ctx: MarketContext) -> dict | None:
    """Detect breakout + retest confirmation."""
    n = len(df)
    if n < 10:
        return None

    atr = ctx.atr
    close = float(df.iloc[-1]["close"])
    low = float(df.iloc[-1]["low"])
    high = float(df.iloc[-1]["high"])

    if zone.zone_type == "resistance":
        # Price broke above resistance, now retesting from above
        # Check: was price below zone 5-15 bars ago, now above?
        was_below = any(float(df.iloc[-(i+1)]["close"]) < zone.z_low for i in range(5, min(15, n)))
        currently_above = close > zone.z_high
        retest_touch = low <= zone.z_high + 0.2 * atr

        if was_below and currently_above and retest_touch:
            # Confirmation: close is still above zone after retest
            prev_close = float(df.iloc[-2]["close"])
            if close > prev_close:  # price rising after retest
                entry = close
                stop = zone.z_low - 0.2 * atr
                risk = entry - stop
                tp = entry + MIN_RR * risk
                rr = (tp - entry) / risk if risk > 0 else 0
                return {"direction": "LONG", "entry": entry, "stop": stop, "tp": tp, "rr": rr}

    elif zone.zone_type == "support":
        # Price broke below support, now retesting from below
        was_above = any(float(df.iloc[-(i+1)]["close"]) > zone.z_high for i in range(5, min(15, n)))
        currently_below = close < zone.z_low
        retest_touch = high >= zone.z_low - 0.2 * atr

        if was_above and currently_below and retest_touch:
            prev_close = float(df.iloc[-2]["close"])
            if close < prev_close:  # price falling after retest
                entry = close
                stop = zone.z_high + 0.2 * atr
                risk = stop - entry
                tp = entry - MIN_RR * risk
                rr = (entry - tp) / risk if risk > 0 else 0
                return {"direction": "SHORT", "entry": entry, "stop": stop, "tp": tp, "rr": rr}

    return None


__all__ = [
    "MarketContext",
    "StrategyDecision",
    "ZoneContext",
    "build_market_context",
    "evaluate_bar",
]
