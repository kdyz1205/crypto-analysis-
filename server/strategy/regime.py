"""Market regime detection — trend vs range, volatility state, directional strength.

Outputs a MarketRegime dataclass that downstream layers use to adjust behavior:
- Signal gating: suppress mean-reversion signals in strong trends, suppress breakout signals in ranges
- Scoring adjustments: weight zone strength differently by regime
- RR targets: wider targets in trends, tighter in ranges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import StrategyConfig, calculate_atr, clamp


RegimeType = Literal["trending", "ranging", "compressed", "breakout"]
VolatilityState = Literal["expanding", "normal", "compressed"]
TrendDirection = Literal["up", "down", "neutral"]


@dataclass(frozen=True, slots=True)
class MarketRegime:
    """Complete market state assessment for one symbol+timeframe."""

    trend_direction: TrendDirection
    trend_strength: float          # 0-1, how directional the market is
    regime: RegimeType             # overall classification
    volatility_state: VolatilityState
    adx: float                     # raw ADX value (0-100)
    range_score: float             # 0-1, how range-bound the market is
    volatility_ratio: float        # current ATR / historical ATR median
    structure_score: float         # 0-1, how clean the HH/HL or LL/LH pattern is

    def favors_trend_following(self) -> bool:
        return self.trend_strength > 0.5 and self.regime == "trending"

    def favors_mean_reversion(self) -> bool:
        return self.range_score > 0.5 and self.regime == "ranging"

    def is_compressed(self) -> bool:
        return self.volatility_state == "compressed"


def detect_regime(
    candles,
    config: StrategyConfig | None = None,
    *,
    adx_period: int = 14,
    structure_lookback: int = 40,
    vol_lookback: int = 50,
) -> MarketRegime:
    """Detect market regime from OHLCV candles.

    Uses three independent signals:
    1. ADX for directional strength
    2. HH/HL structure for trend clarity
    3. ATR ratio for volatility state
    """
    cfg = config or StrategyConfig()
    df = _ensure_df(candles)

    if len(df) < max(adx_period * 2, structure_lookback, vol_lookback):
        return _default_regime()

    # ── 1. ADX — directional strength ──────────────────────────────────
    adx_value, plus_di, minus_di = _calculate_adx(df, adx_period)
    adx_strength = clamp(adx_value / 50.0)  # 50+ = very strong trend

    # DI direction
    if plus_di > minus_di * 1.15:
        di_direction: TrendDirection = "up"
    elif minus_di > plus_di * 1.15:
        di_direction = "down"
    else:
        di_direction = "neutral"

    # ── 2. Structure — HH/HL or LH/LL pattern ─────────────────────────
    structure_score, struct_direction = _score_structure(df, structure_lookback)

    # ── 3. Volatility state ────────────────────────────────────────────
    atr = calculate_atr(df, cfg.atr_period)
    vol_ratio, vol_state = _classify_volatility(atr, vol_lookback)

    # ── 4. Range score — price bounded in narrow band ──────────────────
    range_score = _score_range(df, atr, lookback=structure_lookback)

    # ── Combine into trend direction ───────────────────────────────────
    # Prefer structure direction, fall back to DI
    if structure_score > 0.4:
        trend_direction = struct_direction
    else:
        trend_direction = di_direction

    # ── Trend strength — blend of ADX, structure, and EMA alignment ────
    ema_alignment = _ema_alignment_score(df, trend_direction)
    trend_strength = clamp(
        0.40 * adx_strength
        + 0.35 * structure_score
        + 0.25 * ema_alignment
    )

    # ── Regime classification ──────────────────────────────────────────
    regime = _classify_regime(trend_strength, range_score, vol_state, adx_value)

    return MarketRegime(
        trend_direction=trend_direction,
        trend_strength=round(trend_strength, 4),
        regime=regime,
        volatility_state=vol_state,
        adx=round(adx_value, 2),
        range_score=round(range_score, 4),
        volatility_ratio=round(vol_ratio, 4),
        structure_score=round(structure_score, 4),
    )


# ── ADX calculation ──────────────────────────────────────────────────────

def _calculate_adx(df: pd.DataFrame, period: int) -> tuple[float, float, float]:
    """Calculate ADX, +DI, -DI."""
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    n = len(high)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder smoothing
    atr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    if n <= period:
        return 20.0, 50.0, 50.0

    atr_smooth[period] = np.mean(tr[1 : period + 1])
    plus_dm_smooth[period] = np.mean(plus_dm[1 : period + 1])
    minus_dm_smooth[period] = np.mean(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        atr_smooth[i] = (atr_smooth[i - 1] * (period - 1) + tr[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i - 1] * (period - 1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i - 1] * (period - 1) + minus_dm[i]) / period

    # DI values
    atr_last = atr_smooth[-1]
    if atr_last <= 0:
        return 20.0, 50.0, 50.0

    plus_di = 100.0 * plus_dm_smooth[-1] / atr_last
    minus_di = 100.0 * minus_dm_smooth[-1] / atr_last

    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_dm_smooth[i] + minus_dm_smooth[i]
        if di_sum > 0 and atr_smooth[i] > 0:
            pdi = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
            mdi = 100.0 * minus_dm_smooth[i] / atr_smooth[i]
            dx[i] = 100.0 * abs(pdi - mdi) / max(pdi + mdi, 1e-10)

    # ADX = smoothed DX
    adx_values = np.zeros(n)
    start = period * 2
    if start >= n:
        return 20.0, plus_di, minus_di

    adx_values[start] = np.mean(dx[period:start])
    for i in range(start + 1, n):
        adx_values[i] = (adx_values[i - 1] * (period - 1) + dx[i]) / period

    return float(adx_values[-1]), float(plus_di), float(minus_di)


# ── Structure scoring (HH/HL or LH/LL) ──────────────────────────────────

def _score_structure(df: pd.DataFrame, lookback: int) -> tuple[float, TrendDirection]:
    """Score how clean the swing structure is. Returns (score 0-1, direction)."""
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    n = len(close)
    start = max(0, n - lookback)

    # Find swing highs and lows (simple 3-bar pivots)
    swing_highs = []
    swing_lows = []
    for i in range(start + 2, n - 2):
        if high[i] >= high[i - 1] and high[i] >= high[i - 2] and high[i] >= high[i + 1] and high[i] >= high[i + 2]:
            swing_highs.append((i, float(high[i])))
        if low[i] <= low[i - 1] and low[i] <= low[i - 2] and low[i] <= low[i + 1] and low[i] <= low[i + 2]:
            swing_lows.append((i, float(low[i])))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 0.0, "neutral"

    # Count HH/HL vs LH/LL
    hh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i][1] > swing_highs[i - 1][1])
    lh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i][1] < swing_highs[i - 1][1])
    hl_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i][1] > swing_lows[i - 1][1])
    ll_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i][1] < swing_lows[i - 1][1])

    total_high_pairs = max(len(swing_highs) - 1, 1)
    total_low_pairs = max(len(swing_lows) - 1, 1)

    up_score = (hh_count / total_high_pairs + hl_count / total_low_pairs) / 2.0
    down_score = (lh_count / total_high_pairs + ll_count / total_low_pairs) / 2.0

    if up_score > down_score:
        return clamp(up_score), "up"
    elif down_score > up_score:
        return clamp(down_score), "down"
    return 0.0, "neutral"


# ── Volatility classification ────────────────────────────────────────────

def _classify_volatility(atr: pd.Series, lookback: int) -> tuple[float, VolatilityState]:
    """Classify volatility as expanding/normal/compressed based on ATR ratio."""
    if len(atr) < lookback:
        return 1.0, "normal"

    current_atr = float(atr.iloc[-1])
    median_atr = float(atr.iloc[-lookback:].median())

    if median_atr <= 0:
        return 1.0, "normal"

    ratio = current_atr / median_atr

    if ratio < 0.6:
        return ratio, "compressed"
    elif ratio > 1.5:
        return ratio, "expanding"
    return ratio, "normal"


# ── Range score ──────────────────────────────────────────────────────────

def _score_range(df: pd.DataFrame, atr: pd.Series, lookback: int) -> float:
    """Score how range-bound the market is. High = price staying in narrow band."""
    if len(df) < lookback:
        return 0.0

    close = df["close"].astype(float).values
    recent = close[-lookback:]
    price_range = float(np.max(recent) - np.min(recent))
    avg_atr = float(atr.iloc[-lookback:].mean())

    if avg_atr <= 0:
        return 0.0

    # Range width relative to ATR — narrow range = high score
    # If the range is < 3 ATR over the lookback, it's very range-bound
    normalized_range = price_range / avg_atr
    range_score = clamp(1.0 - (normalized_range - 2.0) / 6.0)

    # Also measure how often price crosses the midpoint (mean-reversion behavior)
    midpoint = float(np.mean(recent))
    crosses = sum(
        1 for i in range(1, len(recent))
        if (recent[i] - midpoint) * (recent[i - 1] - midpoint) < 0
    )
    cross_rate = crosses / max(lookback - 1, 1)
    cross_score = clamp(cross_rate * 5.0)  # ~20% cross rate = max score

    return clamp(0.6 * range_score + 0.4 * cross_score)


# ── EMA alignment ────────────────────────────────────────────────────────

def _ema_alignment_score(df: pd.DataFrame, direction: TrendDirection) -> float:
    """Score EMA stack alignment with trend direction."""
    if len(df) < 55 or direction == "neutral":
        return 0.0

    close = df["close"].astype(float)
    ema_fast = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
    ema_mid = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema_slow = float(close.ewm(span=55, adjust=False).mean().iloc[-1])

    if direction == "up":
        # Perfect alignment: fast > mid > slow
        if ema_fast > ema_mid > ema_slow:
            return 1.0
        elif ema_fast > ema_slow:
            return 0.5
        return 0.0
    else:
        # Perfect alignment: fast < mid < slow
        if ema_fast < ema_mid < ema_slow:
            return 1.0
        elif ema_fast < ema_slow:
            return 0.5
        return 0.0


# ── Regime classification ────────────────────────────────────────────────

def _classify_regime(
    trend_strength: float,
    range_score: float,
    vol_state: VolatilityState,
    adx: float,
) -> RegimeType:
    """Classify into one of four regimes."""
    if vol_state == "compressed" and range_score > 0.5:
        return "compressed"  # Squeeze — expect breakout
    if trend_strength > 0.55 and adx > 25:
        return "trending"
    if vol_state == "expanding" and trend_strength > 0.4:
        return "breakout"
    if range_score > 0.45:
        return "ranging"
    if trend_strength > 0.35:
        return "trending"
    return "ranging"


# ── Helpers ──────────────────────────────────────────────────────────────

def _ensure_df(candles) -> pd.DataFrame:
    if isinstance(candles, pd.DataFrame):
        return candles
    raise TypeError(f"Expected DataFrame, got {type(candles)}")


def _default_regime() -> MarketRegime:
    return MarketRegime(
        trend_direction="neutral",
        trend_strength=0.0,
        regime="ranging",
        volatility_state="normal",
        adx=20.0,
        range_score=0.5,
        volatility_ratio=1.0,
        structure_score=0.0,
    )


__all__ = [
    "MarketRegime",
    "RegimeType",
    "TrendDirection",
    "VolatilityState",
    "detect_regime",
]
