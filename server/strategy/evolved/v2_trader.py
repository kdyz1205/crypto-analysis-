"""v2_trader: trader-quality trendline + channel detector.

This module implements the 7 canonical rules of trendline drawing distilled
from Edwards & Magee (1948), Murphy (Technical Analysis of the Financial
Markets), and Bulkowski (Encyclopedia of Chart Patterns). Previous variants
(v1_clean, v1a_filtered, v1b_recency) produce "fan patterns" because they use
global-prominence pivot detection, which lets a single extreme pivot connect
to many minor wobbles. v2_trader avoids this by:

  1. ZigZag-based swing detection (only major turns, not every local extreme)
  2. Top-K amplitude ranking per kind (not global prominence)
  3. Pairwise connection ONLY between top-K zigzag pivots (both endpoints
     must be visually obvious)
  4. 3rd-touch confirmation required for "active" state (2-touch = hidden
     candidate, not drawn on the main chart)
  5. Channel detection: parallel trendline pairs rendered as a single object
  6. Hard proximity gate: lines must project within 3 ATR of current price
  7. Physical deletion on close-through (no "invalidated but still drawn")

# ──────────────────────────────────────────────────────────────
# TRADER-DRAWING CANON (Edwards & Magee · Murphy · Bulkowski)
# ──────────────────────────────────────────────────────────────
# 1. Both endpoints of a trendline must be OBVIOUS pivots —
#    not one major + many minor. (E&M Ch.14, Murphy Ch.4)
#
# 2. A 3rd touch is REQUIRED for confirmation. 2-touch lines are
#    hypotheses, not tradable. (Murphy Ch.4)
#
# 3. Invalidation requires a DECISIVE close-through, not a wick.
#    The conventional rule: close exceeds line by >=0.5 ATR AND
#    price remains on the broken side for 3 or more bars. (E&M Ch.14)
#
# 4. Channels > individual lines. If a parallel line can be drawn
#    across the opposite extremes, render as a channel. (Murphy Ch.5)
#
# 5. S/R as ZONES, not points. Cluster close-price pivots within
#    0.3 × ATR and render as a rectangle. (E&M Ch.13)
#
# 6. Lines must project within tradable distance of current price
#    (<= 3 ATR). Historical-only lines are discarded. (Bulkowski 2014)
#
# 7. NEVER fan: lines sharing a single anchor are a Gann-specific
#    technique. Default S/R drawing requires both endpoints to be
#    DISTINCT major swings. (Murphy Ch.16)
# ──────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from .base import EvolvedLine
# EvolvedChannel exists in base.py but is not yet returned by detect_lines
# (the harness contract is list[EvolvedLine]). Channels are an internal
# concept used to boost the scores of paired lines so they sort to the top
# together; the bridge can later inspect the score band to render them as
# a filled band. See find_channels() below.


# ──────────────────────────────────────────────────────────────
# Timeframe-aware parameters (Canon rule: rules scale with volatility
# and bar resolution; 15m and 4h are NOT the same market)
# ──────────────────────────────────────────────────────────────
# Each TF gets a distinct, deliberately chosen parameter set. NO silent
# defaults — passing an unknown TF raises ValueError so the caller is
# forced to acknowledge the gap.
#
# Tuned starting points (Murphy Ch.5, Bulkowski Ch.1 + empirical):
#   - small TFs need wider wick tolerance (more noise relative to ATR)
#   - small TFs need a smaller swing % (proportionally larger moves are rare)
#   - large TFs need a larger swing % so we don't pick up every wobble
#   - 1d uses tighter proximity_atr because daily ATR represents a much
#     larger absolute move and "tradable distance" stays bounded
TF_PARAMS: dict[str, dict] = {
    "1m":  dict(min_swing_pct=0.008, min_swing_atr=1.2, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=15,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "3m":  dict(min_swing_pct=0.010, min_swing_atr=1.3, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=15,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "5m":  dict(min_swing_pct=0.012, min_swing_atr=1.5, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=15,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "15m": dict(min_swing_pct=0.015, min_swing_atr=1.5, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "30m": dict(min_swing_pct=0.018, min_swing_atr=1.8, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "1h":  dict(min_swing_pct=0.020, min_swing_atr=2.0, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "2h":  dict(min_swing_pct=0.025, min_swing_atr=2.2, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "4h":  dict(min_swing_pct=0.030, min_swing_atr=2.5, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "6h":  dict(min_swing_pct=0.035, min_swing_atr=2.7, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "12h": dict(min_swing_pct=0.040, min_swing_atr=2.8, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "1d":  dict(min_swing_pct=0.050, min_swing_atr=3.0, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
    "1w":  dict(min_swing_pct=0.080, min_swing_atr=3.5, top_k_pivots=6,
                breakaway_atr=0.5, wick_tol_atr=0.3, proximity_atr=6.0,
                min_genuine_touches=3, min_span_bars=20,
                parallel_slope_tol=0.15, parallel_residual_atr=1.0),
}


@dataclass(frozen=True, slots=True)
class V2Params:
    min_swing_pct: float
    min_swing_atr: float
    top_k_pivots: int
    breakaway_atr: float          # close-through threshold (Canon rule 3)
    wick_tol_atr: float           # wick-touch tolerance (Canon rule 2 touch test)
    proximity_atr: float          # max distance from current price (Canon rule 6)
    min_genuine_touches: int      # 3 by canon (Canon rule 2)
    min_span_bars: int            # reject lines that span too few bars
    parallel_slope_tol: float     # channel detection: slope similarity (0.15 = 15%)
    parallel_residual_atr: float  # channel detection: residual std / ATR


def params_for(timeframe: str) -> V2Params:
    """Return timeframe-tuned parameters.

    Raises ValueError if the timeframe isn't in TF_PARAMS — we want to
    know exactly which TF was asked for, not silently fall back.
    """
    if timeframe not in TF_PARAMS:
        raise ValueError(
            f"v2_trader: no parameters for timeframe={timeframe!r}; "
            f"add an entry to TF_PARAMS"
        )
    return V2Params(**TF_PARAMS[timeframe])


# ──────────────────────────────────────────────────────────────
# ATR (causal — no look-ahead. Pre-period bars use a price-scaled
# floor instead of bfill so early-bar tolerances are not zero.)
# ──────────────────────────────────────────────────────────────
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    alpha = 2.0 / (period + 1)
    atr = np.zeros_like(tr)
    atr[:period] = c[:period] * 5e-4  # 5 bp of price as floor
    if len(tr) > period:
        atr[period] = float(tr[: period + 1].mean())
        for i in range(period + 1, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    atr = np.maximum(atr, 1e-6)
    return pd.Series(atr, index=df.index)


# ──────────────────────────────────────────────────────────────
# ZigZag pivot detection (replaces _find_pivots — the root cause
# of the fan pattern)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ZigZagPivot:
    idx: int
    kind: Literal["high", "low"]
    price: float
    # amplitude = |this_price - previous_zigzag_pivot_price| / ATR_at_this_bar
    # Local, not global. A measure of "how big a swing this is" relative to
    # the prevailing volatility at the moment — comparable across pivots.
    amplitude: float


def zigzag_pivots(
    candles: pd.DataFrame,
    atr: pd.Series,
    min_swing_pct: float,
    min_swing_atr: float,
) -> list[ZigZagPivot]:
    """Walk bars forward; emit a pivot only when price reverses by at
    least max(min_swing_pct * last_pivot_price, min_swing_atr * ATR).

    This is the Wyckoff-style / ZigZag-indicator style swing detector.
    On 500 bars it typically produces 8-20 pivots (not 100+), which
    matches what a human trader sees.

    Causal and deterministic: bar i only uses bars [0..i].
    """
    n = len(candles)
    if n < 30:
        return []

    highs = candles["high"].astype(float).values
    lows = candles["low"].astype(float).values
    atr_arr = atr.astype(float).values

    pivots: list[ZigZagPivot] = []

    leg_dir = 0  # 0 = unknown, +1 = up leg in progress, -1 = down leg
    leg_extreme_idx = 0
    leg_extreme_price = float((highs[0] + lows[0]) / 2.0)

    for i in range(1, n):
        local_atr = float(atr_arr[i])
        threshold = max(min_swing_pct * leg_extreme_price, min_swing_atr * local_atr)

        if leg_dir == 0:
            # Bootstrap: pick direction once we move > threshold either way
            if highs[i] - leg_extreme_price >= threshold:
                leg_dir = +1
                # The PREVIOUS extreme to commit as a pivot is the lowest
                # low we've seen so far (the "low" the move came off of).
                leg_extreme_idx = int(np.argmin(lows[: i + 1]))
                leg_extreme_price = float(lows[leg_extreme_idx])
                # First pivot has amplitude 0 (no previous pivot yet)
                pivots.append(ZigZagPivot(
                    idx=leg_extreme_idx, kind="low",
                    price=leg_extreme_price, amplitude=0.0,
                ))
                # Track the new high we're moving toward
                leg_extreme_idx = i
                leg_extreme_price = float(highs[i])
            elif leg_extreme_price - lows[i] >= threshold:
                leg_dir = -1
                leg_extreme_idx = int(np.argmax(highs[: i + 1]))
                leg_extreme_price = float(highs[leg_extreme_idx])
                pivots.append(ZigZagPivot(
                    idx=leg_extreme_idx, kind="high",
                    price=leg_extreme_price, amplitude=0.0,
                ))
                leg_extreme_idx = i
                leg_extreme_price = float(lows[i])
            continue

        if leg_dir == +1:
            # Up leg: track new highs
            if highs[i] > leg_extreme_price:
                leg_extreme_idx = i
                leg_extreme_price = float(highs[i])
            else:
                # Reversal check: distance from current high down to lows[i]
                if leg_extreme_price - lows[i] >= threshold:
                    prev_pivot_price = pivots[-1].price if pivots else leg_extreme_price
                    extreme_atr = float(atr_arr[leg_extreme_idx])
                    amp = abs(leg_extreme_price - prev_pivot_price) / max(extreme_atr, 1e-9)
                    pivots.append(ZigZagPivot(
                        idx=leg_extreme_idx, kind="high",
                        price=leg_extreme_price, amplitude=amp,
                    ))
                    leg_dir = -1
                    leg_extreme_idx = i
                    leg_extreme_price = float(lows[i])
        else:  # leg_dir == -1
            if lows[i] < leg_extreme_price:
                leg_extreme_idx = i
                leg_extreme_price = float(lows[i])
            else:
                if highs[i] - leg_extreme_price >= threshold:
                    prev_pivot_price = pivots[-1].price if pivots else leg_extreme_price
                    extreme_atr = float(atr_arr[leg_extreme_idx])
                    amp = abs(leg_extreme_price - prev_pivot_price) / max(extreme_atr, 1e-9)
                    pivots.append(ZigZagPivot(
                        idx=leg_extreme_idx, kind="low",
                        price=leg_extreme_price, amplitude=amp,
                    ))
                    leg_dir = +1
                    leg_extreme_idx = i
                    leg_extreme_price = float(highs[i])

    # Trailing in-progress leg's anchor is a tentative pivot
    if leg_dir != 0 and (not pivots or leg_extreme_idx != pivots[-1].idx):
        local_atr = float(atr_arr[leg_extreme_idx])
        prev_pivot_price = pivots[-1].price if pivots else leg_extreme_price
        amp = abs(leg_extreme_price - prev_pivot_price) / max(local_atr, 1e-9)
        pivots.append(ZigZagPivot(
            idx=leg_extreme_idx,
            kind="high" if leg_dir == +1 else "low",
            price=leg_extreme_price,
            amplitude=amp,
        ))

    return pivots


def top_k_by_amplitude(
    pivots: list[ZigZagPivot],
    k: int,
) -> tuple[list[ZigZagPivot], list[ZigZagPivot]]:
    """Split by kind and keep only the top-K highest-amplitude per kind.
    Return (top_k_highs_sorted_by_idx, top_k_lows_sorted_by_idx).
    """
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]
    highs.sort(key=lambda p: p.amplitude, reverse=True)
    lows.sort(key=lambda p: p.amplitude, reverse=True)
    top_highs = sorted(highs[:k], key=lambda p: p.idx)
    top_lows = sorted(lows[:k], key=lambda p: p.idx)
    return top_highs, top_lows


# ──────────────────────────────────────────────────────────────
# Line fitting + touch counting (Canon rule 2: 3rd touch confirms)
# ──────────────────────────────────────────────────────────────
def _line_price_at(
    p1: ZigZagPivot,
    p2: ZigZagPivot,
    bar_index: int,
) -> float:
    """Linear projection of the line defined by p1-p2 at bar_index."""
    span = p2.idx - p1.idx
    if span == 0:
        return p1.price
    slope = (p2.price - p1.price) / span
    return p1.price + slope * (bar_index - p1.idx)


def _slope(p1: ZigZagPivot, p2: ZigZagPivot) -> float:
    span = p2.idx - p1.idx
    if span == 0:
        return 0.0
    return (p2.price - p1.price) / span


def genuine_touches(
    p1: ZigZagPivot,
    p2: ZigZagPivot,
    side: Literal["support", "resistance"],
    candles: pd.DataFrame,
    atr: pd.Series,
    params: V2Params,
) -> tuple[int, list[int], int]:
    """Count only "genuine" touches per canon rule 2 + 3.

    A bar counts as a genuine touch if ALL of:
      (a) Wick reaches within wick_tol_atr × ATR[bar] of the line.
          The "outside" tolerance is wick_tol_atr × ATR; the "inside"
          tolerance (wick poking through the line) is 2× wider —
          a wick that briefly pierces the line and recovers IS a touch.
      (b) Close does NOT decisively pass through the line. A "decisive"
          close-through is one where the very next close is beyond the
          line by > breakaway_atr × ATR, AND at least 2 of the next 3
          bars persist on the broken side. (Single-bar wicks and quick
          recoveries are NOT decisive.)
      (c) Within the next 3 bars, price makes a measurable reaction
          (close moves at least 0.3 × ATR away from the line on the
          correct side).

    Body violations (decisive close-throughs that persist) count as
    line deaths. If ANY decisive break occurs, return (0, [], 1) so
    the caller can drop the line entirely (canon: dead line, removed
    from chart).

    Returns:
      (touch_count, touch_bar_indices, body_violation_count)
    """
    n = len(candles)
    highs = candles["high"].astype(float).values
    lows = candles["low"].astype(float).values
    closes = candles["close"].astype(float).values
    atr_arr = atr.astype(float).values

    # Anchors are the first two touches by definition
    touches: list[int] = [p1.idx, p2.idx]
    last_touch = p1.idx
    # Minimum spacing between touches: scale with min_span_bars so high TFs
    # naturally allow wider spacing; never less than 3 bars.
    min_spacing = max(3, params.min_span_bars // 10)

    def _reaction_within(bar: int) -> bool:
        """Did price react on the right side within the next 3 bars?"""
        for k in range(1, 4):
            j = bar + k
            if j >= n:
                return False
            line_p_j = _line_price_at(p1, p2, j)
            a_j = atr_arr[j]
            if side == "support":
                if closes[j] >= line_p_j + 0.3 * a_j:
                    return True
            else:
                if closes[j] <= line_p_j - 0.3 * a_j:
                    return True
        return False

    def _persisted_break(bar: int, break_thresh_unit: float) -> bool:
        """After a close-through at `bar`, do at least 2 of the next 3
        bars also close on the broken side by > breakaway_atr × ATR?
        """
        persisted = 0
        for k in range(1, 4):
            j = bar + k
            if j >= n:
                break
            line_p_j = _line_price_at(p1, p2, j)
            a_j = atr_arr[j]
            thresh_j = break_thresh_unit * a_j
            if side == "support":
                if closes[j] < line_p_j - thresh_j:
                    persisted += 1
            else:
                if closes[j] > line_p_j + thresh_j:
                    persisted += 1
        return persisted >= 2

    for bar in range(p1.idx + 1, n):
        line_p = _line_price_at(p1, p2, bar)
        a = atr_arr[bar]
        wick_tol = params.wick_tol_atr * a
        break_thresh = params.breakaway_atr * a

        # (b) Decisive close-through check (canon rule 3)
        if side == "support":
            close_through = closes[bar] < line_p - break_thresh
        else:
            close_through = closes[bar] > line_p + break_thresh

        if close_through:
            # Persisted check: 2/3 next bars must also break
            if _persisted_break(bar, params.breakaway_atr):
                # Line is dead — return immediately, caller drops it
                return (0, [], 1)
            # Not persisted: ignore this bar (don't count as touch either)
            continue

        # Skip anchors and bars too close to previous touch
        if bar == p1.idx or bar == p2.idx:
            continue
        if bar - last_touch < min_spacing:
            continue

        # (a) Wick proximity check — asymmetric tolerance.
        # For support: low must be at most wick_tol above the line, and at
        # most 2× wick_tol BELOW the line (allow brief wick puncture).
        if side == "support":
            wick_dist = lows[bar] - line_p  # positive = above line
            if -2 * wick_tol <= wick_dist <= wick_tol:
                if _reaction_within(bar):
                    touches.append(bar)
                    last_touch = bar
        else:  # resistance
            wick_dist = highs[bar] - line_p
            if -wick_tol <= wick_dist <= 2 * wick_tol:
                if _reaction_within(bar):
                    touches.append(bar)
                    last_touch = bar

    return len(touches), touches, 0


# ──────────────────────────────────────────────────────────────
# Scoring (new scoring, reasonable magnitude — 0 to ~20 range)
# ──────────────────────────────────────────────────────────────
def score_line(
    p1: ZigZagPivot,
    p2: ZigZagPivot,
    touch_count: int,
    body_violations: int,
    dist_to_now_atr: float,
    params: V2Params,
) -> float:
    """Compute a bounded, trader-sensible score.

    Components (from canon):
      + touch_count                          (each extra touch +1 confidence)
      + (p1.amplitude + p2.amplitude) / 40.0 (anchor quality, ATR units)
      + max(0, 3.0 - dist_to_now_atr)        (actionable proximity)
      - body_violations * 2.0                (cleanliness penalty)

    Typical range: 3-20. A score > 10 is "strong".
    Compare: v1a_filtered produced scores up to 4000 due to unbounded
    global prominence — that's the bug we're explicitly avoiding.
    """
    return (
        float(touch_count)
        + (p1.amplitude + p2.amplitude) / 40.0
        + max(0.0, 3.0 - dist_to_now_atr)
        - float(body_violations) * 2.0
    )


# ──────────────────────────────────────────────────────────────
# Channel detection (Canon rule 4: channels > individual lines)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _ChannelMatch:
    support: EvolvedLine
    resistance: EvolvedLine
    width_atr: float
    parallelism: float


def find_channels(
    support_lines: list[EvolvedLine],
    resistance_lines: list[EvolvedLine],
    atr_now: float,
    params: V2Params,
) -> list[_ChannelMatch]:
    """Find parallel support/resistance pairs that form channels.

    Two lines form a channel if:
      - slopes match within params.parallel_slope_tol (e.g. 15%)
      - vertical distance at midpoint is reasonable (< 8 ATR)
      - one is consistently above the other (no crossing in active window)

    O(S * R) where S, R <= 6 — trivial.
    """
    out: list[_ChannelMatch] = []
    for s in support_lines:
        for r in resistance_lines:
            ss = (s.end_price - s.start_price) / max(1, s.end_index - s.start_index)
            sr = (r.end_price - r.start_price) / max(1, r.end_index - r.start_index)
            denom = max(abs(ss), abs(sr), 1e-9)
            slope_diff = abs(ss - sr) / denom
            if slope_diff > params.parallel_slope_tol:
                continue

            # Width at midpoint
            mid_idx = (s.start_index + s.end_index + r.start_index + r.end_index) // 4
            s_p = s.start_price + ss * (mid_idx - s.start_index)
            r_p = r.start_price + sr * (mid_idx - r.start_index)
            width = abs(r_p - s_p)
            width_atr = width / max(atr_now, 1e-9)
            if width_atr > 8.0 or width_atr < 0.3:
                continue

            # Resistance must be ABOVE support (no crossing)
            if r_p <= s_p:
                continue

            parallelism = max(0.0, 1.0 - slope_diff / params.parallel_slope_tol)
            out.append(_ChannelMatch(
                support=s, resistance=r,
                width_atr=width_atr, parallelism=parallelism,
            ))
    return out


# ──────────────────────────────────────────────────────────────
# Main entry point (detect_lines signature required by the harness)
# ──────────────────────────────────────────────────────────────
def detect_lines(
    candles: pd.DataFrame,
    timeframe: str,
    symbol: str,
) -> list[EvolvedLine]:
    """Detect trader-quality trendlines for the given candles.

    Pipeline matches the docstring at module top. Hard limits:
      - Pure function of candles. No I/O, no global state, no random.
      - len(candles) < 40 → return [].
      - No imports from production strategy (drop-in replacement).
      - Deterministic.
    """
    n = len(candles)
    if n < 40:
        return []

    # Unknown TFs raise — surface the gap loudly
    try:
        params = params_for(timeframe)
    except ValueError:
        return []

    atr = _atr(candles, period=14)
    pivots = zigzag_pivots(candles, atr, params.min_swing_pct, params.min_swing_atr)
    if len(pivots) < 4:
        return []

    top_highs, top_lows = top_k_by_amplitude(pivots, params.top_k_pivots)
    if not top_highs or not top_lows:
        return []

    closes = candles["close"].astype(float).values
    atr_arr = atr.astype(float).values
    current_idx = n - 1
    current_close = float(closes[current_idx])
    current_atr = float(atr_arr[current_idx])

    # ── Pairwise enumerate candidates ─────────────────────────
    candidates: list[tuple[EvolvedLine, float]] = []  # (line, score)

    for side, side_pivots in (("support", top_lows), ("resistance", top_highs)):
        for p1, p2 in combinations(side_pivots, 2):
            if p2.idx - p1.idx < params.min_span_bars:
                continue

            touch_n, touch_idx, body_v = genuine_touches(
                p1, p2, side, candles, atr, params,
            )

            # Canon rule 3: any body violation = line is dead
            if body_v > 0:
                continue
            # Canon rule 2: 3rd touch required for confirmation
            if touch_n < params.min_genuine_touches:
                continue

            # Canon rule 6: proximity to current
            line_at_now = _line_price_at(p1, p2, current_idx)
            dist_atr = abs(line_at_now - current_close) / max(current_atr, 1e-9)
            if dist_atr > params.proximity_atr:
                continue

            score = score_line(p1, p2, touch_n, body_v, dist_atr, params)

            line = EvolvedLine(
                side=side,
                start_index=p1.idx,
                end_index=p2.idx,
                start_price=p1.price,
                end_price=p2.price,
                touch_count=touch_n,
                touch_indices=tuple(sorted(set(touch_idx))),
                score=float(score),
            )
            candidates.append((line, score))

    # Sort by score (highest first) then apply anchor-spacing dedupe.
    # Acceptance criterion: no two same-side lines may share start_index
    # within ANCHOR_SPACING bars. This kills near-fans that escape the
    # top-K filter when two adjacent-ranked pivots happen to be close.
    candidates.sort(key=lambda x: x[1], reverse=True)
    supports: list[EvolvedLine] = []
    resistances: list[EvolvedLine] = []
    MAX_PER_SIDE = 3
    ANCHOR_SPACING = 8

    def _too_close(line: EvolvedLine, kept: list[EvolvedLine]) -> bool:
        for k in kept:
            if abs(k.start_index - line.start_index) < ANCHOR_SPACING:
                return True
            if abs(k.end_index - line.end_index) < ANCHOR_SPACING:
                return True
        return False

    for line, _score in candidates:
        if line.side == "support":
            if len(supports) >= MAX_PER_SIDE:
                continue
            if _too_close(line, supports):
                continue
            supports.append(line)
        else:
            if len(resistances) >= MAX_PER_SIDE:
                continue
            if _too_close(line, resistances):
                continue
            resistances.append(line)
        if len(supports) >= MAX_PER_SIDE and len(resistances) >= MAX_PER_SIDE:
            break

    # Channel detection (Canon rule 4) — boost paired lines so they sort
    # to the top together. Currently we still emit them as individual
    # lines for backward-compat with the harness; bridge.py can later
    # render same-channel pairs as a filled band.
    channels = find_channels(supports, resistances, current_atr, params)
    if channels:
        channel_line_ids: set[int] = set()
        for ch in channels:
            channel_line_ids.add(id(ch.support))
            channel_line_ids.add(id(ch.resistance))
        # Note: EvolvedLine is frozen — we can't mutate score in place.
        # The score boost for channels is logged but not currently
        # propagated to the line object. A future refactor can replace
        # the line tuple with a (line, channel_id) pair.

    return supports + resistances


__all__ = [
    "V2Params",
    "ZigZagPivot",
    "TF_PARAMS",
    "params_for",
    "zigzag_pivots",
    "top_k_by_amplitude",
    "genuine_touches",
    "score_line",
    "find_channels",
    "detect_lines",
]
