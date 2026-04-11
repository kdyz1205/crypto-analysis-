"""
Dynamic position sizing based on zone quality, timeframe, and Kelly criterion.

Core principle: Higher confidence setups get larger positions.
- Tight stops on small timeframes → larger position (same dollar risk)
- More touches + multi-TF confluence + volume → higher confidence → Kelly allows more
- ATR-scaled stops per timeframe
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import calculate_atr, clamp
from .types import StrategySignal, ensure_candles_df
from .zones import HorizontalZone


# ATR multipliers for stop distance by timeframe
# Calibrated from real backtest data (8 coins, 365 days)
# Median real stop sizes: 5m=1.03%, 15m=1.02%, 1h=1.09%, 4h=1.14%
TIMEFRAME_STOP_ATR_MULT = {
    "1m": 0.10,
    "3m": 0.12,
    "5m": 0.15,   # real: ~1.0% stop
    "15m": 0.18,  # real: ~1.0% stop
    "1h": 0.22,   # real: ~1.1% stop
    "4h": 0.30,   # real: ~1.1-1.4% stop
    "1d": 0.45,
    "1w": 0.60,
}

# Base leverage limits by timeframe
# Conservative: with 1%+ stops, high leverage = large dollar risk
TIMEFRAME_MAX_LEVERAGE = {
    "1m": 10,
    "3m": 8,
    "5m": 5,
    "15m": 5,
    "1h": 3,
    "4h": 2,
    "1d": 1,
    "1w": 1,
}


@dataclass(frozen=True, slots=True)
class PositionSizeResult:
    """Output of the position sizing engine."""
    base_risk_pct: float        # base risk as % of equity (before Kelly)
    kelly_fraction: float       # Kelly optimal fraction (0-1)
    adjusted_risk_pct: float    # actual risk % after Kelly + confidence scaling
    stop_distance_pct: float    # stop loss as % of entry price
    stop_distance_atr: float    # stop loss in ATR units
    leverage: float             # recommended leverage
    max_leverage: float         # hard cap for this timeframe
    confidence_score: float     # 0-1 composite confidence
    confidence_factors: dict    # breakdown of confidence components
    position_size_multiplier: float  # multiply base qty by this


def calculate_stop_distance(
    candles,
    timeframe: str,
    zone: HorizontalZone | None = None,
) -> tuple[float, float]:
    """Calculate optimal stop distance based on timeframe and ATR.

    Returns (stop_distance_price, stop_distance_pct).
    """
    df = ensure_candles_df(candles)
    if df.empty:
        return 0.0, 0.01

    atr = calculate_atr(df, 14)
    current_index = len(df) - 1
    atr_value = float(atr.iloc[current_index])
    close_price = float(df.iloc[current_index]["close"])

    atr_mult = TIMEFRAME_STOP_ATR_MULT.get(timeframe, 0.12)
    stop_from_atr = atr_value * atr_mult

    # If we have a zone, stop can be tighter: just outside the zone
    if zone is not None:
        zone_width = zone.price_high - zone.price_low
        stop_from_zone = zone_width * 0.3 + atr_value * 0.02  # 30% of zone width + tiny buffer
        # Use the tighter of ATR-based or zone-based
        stop_distance = min(stop_from_atr, stop_from_zone)
    else:
        stop_distance = stop_from_atr

    # Floor: never less than 0.05% of price
    min_stop = close_price * 0.0005
    stop_distance = max(stop_distance, min_stop)

    stop_pct = stop_distance / close_price if close_price > 0 else 0.01
    return stop_distance, stop_pct


def calculate_confidence(
    zone: HorizontalZone,
    signal: StrategySignal | None = None,
    *,
    has_volume_surge: bool = False,
    multi_tf_confluence: float = 0.0,
    trend_aligned: bool = True,
) -> tuple[float, dict]:
    """Calculate confidence score for a zone-based trade.

    Higher confidence = can risk more (via Kelly).
    Returns (score 0-1, factor breakdown).
    """
    factors = {}

    # 1. Touch count (3=base, 4-5=good, 6+=excellent)
    touch_score = clamp((zone.touches - 2) / 4.0)
    factors["touches"] = round(touch_score, 3)

    # 2. Zone strength (already 0-100)
    strength_score = clamp(zone.strength / 100.0)
    factors["zone_strength"] = round(strength_score, 3)

    # 3. Volume confirmation
    volume_score = 1.0 if has_volume_surge else 0.0
    factors["volume"] = volume_score

    # 4. Multi-TF confluence (0-1 from confluence.py)
    confluence_score = clamp(multi_tf_confluence)
    factors["confluence"] = round(confluence_score, 3)

    # 5. Trend alignment
    trend_score = 1.0 if trend_aligned else 0.3
    factors["trend"] = trend_score

    # 6. Signal quality (if available)
    signal_score = clamp(signal.score) if signal else 0.5
    factors["signal"] = round(signal_score, 3)

    # 7. Recency
    recency = clamp(1.0 - (zone.last_touch_index / max(zone.last_touch_index + 50, 1)))
    factors["recency"] = round(recency, 3)

    # Weighted composite
    confidence = (
        0.25 * touch_score
        + 0.15 * strength_score
        + 0.10 * volume_score
        + 0.20 * confluence_score
        + 0.15 * trend_score
        + 0.10 * signal_score
        + 0.05 * recency
    )

    return clamp(confidence), factors


def kelly_fraction(win_rate: float, avg_rr: float) -> float:
    """Calculate Kelly criterion optimal fraction.

    f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = avg win/avg loss ratio

    We use half-Kelly for safety.
    """
    if avg_rr <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    p = win_rate
    q = 1.0 - p
    b = avg_rr

    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return 0.0

    # Half-Kelly for safety
    return clamp(full_kelly * 0.5, 0.0, 0.25)  # cap at 25% of equity


def calculate_position_size(
    candles,
    zone: HorizontalZone,
    signal: StrategySignal | None = None,
    *,
    timeframe: str = "1h",
    equity: float = 10000.0,
    base_risk_pct: float = 0.01,  # 1% base risk
    win_rate: float = 0.0,        # MUST be provided per-timeframe from backtest data
    avg_rr: float = 0.0,          # MUST be provided per-timeframe from backtest data
    has_volume_surge: bool = False,
    multi_tf_confluence: float = 0.0,
    trend_aligned: bool = True,
) -> PositionSizeResult:
    """Full position sizing calculation.

    Combines:
    - ATR-scaled stop distance by timeframe
    - Zone-based tight stops
    - Confidence scoring for leverage adjustment
    - Kelly criterion for optimal risk fraction
    """
    # 1. Calculate stop distance
    stop_dist, stop_pct = calculate_stop_distance(candles, timeframe, zone)
    df = ensure_candles_df(candles)
    atr = calculate_atr(df, 14)
    atr_value = float(atr.iloc[-1]) if len(df) > 0 else 1.0
    stop_atr = stop_dist / max(atr_value, 1e-10)

    # 2. Calculate confidence
    confidence, conf_factors = calculate_confidence(
        zone, signal,
        has_volume_surge=has_volume_surge,
        multi_tf_confluence=multi_tf_confluence,
        trend_aligned=trend_aligned,
    )

    # 3. Kelly fraction
    kf = kelly_fraction(win_rate, avg_rr)

    # 4. Adjusted risk: scale base risk by confidence and Kelly
    # Low confidence (< 0.3): reduce risk
    # High confidence (> 0.7): increase up to Kelly fraction
    if confidence < 0.3:
        risk_multiplier = 0.5
    elif confidence < 0.5:
        risk_multiplier = 1.0
    elif confidence < 0.7:
        risk_multiplier = 1.5
    else:
        risk_multiplier = 2.0  # high confidence: double base risk

    adjusted_risk = min(base_risk_pct * risk_multiplier, kf if kf > 0 else base_risk_pct)

    # 5. Leverage
    max_lev = TIMEFRAME_MAX_LEVERAGE.get(timeframe, 5)
    # Leverage based on confidence, capped by timeframe limit
    # Only high confidence (>0.6) gets meaningful leverage
    if confidence >= 0.7:
        leverage = min(max_lev, max_lev * 0.8)
    elif confidence >= 0.5:
        leverage = min(max_lev * 0.5, max_lev)
    elif confidence >= 0.3:
        leverage = min(2.0, max_lev)
    else:
        leverage = 1.0
    leverage = max(1.0, leverage)

    # 6. Position size multiplier
    multiplier = adjusted_risk / max(base_risk_pct, 1e-10)

    return PositionSizeResult(
        base_risk_pct=base_risk_pct,
        kelly_fraction=kf,
        adjusted_risk_pct=adjusted_risk,
        stop_distance_pct=stop_pct,
        stop_distance_atr=stop_atr,
        leverage=round(leverage, 1),
        max_leverage=max_lev,
        confidence_score=confidence,
        confidence_factors=conf_factors,
        position_size_multiplier=round(multiplier, 2),
    )


# Per-timeframe calibration from real backtest (8 coins, 365 days, 76 trades)
# Only use timeframes that showed positive expected value
BACKTEST_CALIBRATION = {
    # tf: (win_rate, avg_rr, ev_per_trade, half_kelly, median_stop_pct)
    "1h":  (0.125, 8.36, 0.170, 0.0102, 0.0108),  # marginal
    "4h":  (0.229, 5.62, 0.516, 0.0460, 0.0114),  # solid
    # 5m and 15m: NOT PROFITABLE in backtest — do not use for sizing
    # "5m":  (0.0, 3.52, -1.0, 0.0, 0.0103),
    # "15m": (0.0, 3.54, -1.0, 0.0, 0.0102),
}


def get_calibrated_params(timeframe: str) -> tuple[float, float, float]:
    """Return (win_rate, avg_rr, half_kelly) for a timeframe.
    Returns (0, 0, 0) for uncalibrated timeframes — they must NOT trade.
    """
    if timeframe in BACKTEST_CALIBRATION:
        wr, rr, _, hk, _ = BACKTEST_CALIBRATION[timeframe]
        return wr, rr, hk
    # Uncalibrated = no evidence of profitability = do not trade
    return 0.0, 0.0, 0.0


def is_timeframe_verified(timeframe: str) -> bool:
    """True only if this timeframe has positive EV in backtest calibration."""
    return timeframe in BACKTEST_CALIBRATION


__all__ = [
    "BACKTEST_CALIBRATION",
    "TIMEFRAME_STOP_ATR_MULT",
    "PositionSizeResult",
    "calculate_confidence",
    "calculate_position_size",
    "calculate_stop_distance",
    "get_calibrated_params",
    "is_timeframe_verified",
    "kelly_fraction",
]
