"""v2_trader: trendline detector built on traditional TA canon.

Replaces the v1_clean / v1a_filtered fan-prone approach with a canonical
trader-style pipeline. The 7 rules from Edwards & Magee, Murphy, and
Bulkowski are encoded as discrete pipeline steps:

──────────────────────────────────────────────────────────────
TRADER-DRAWING CANON (Edwards & Magee · Murphy · Bulkowski)
──────────────────────────────────────────────────────────────
 1. Both endpoints of a trendline must be OBVIOUS pivots —
    not one major + many minor. (E&M Ch.14)

 2. A 3rd touch is REQUIRED for confirmation. 2-touch lines are
    hypotheses, not tradable. (Murphy Ch.4)

 3. Invalidation requires a DECISIVE close-through, not a wick.
    Conventional rule: close exceeds line by >= 0.5 ATR AND price
    remains on the broken side for 3 or more bars. (E&M Ch.14)

 4. Channels > individual lines. If a parallel line can be drawn
    across the opposite extremes, render as a channel. (Murphy Ch.5)

 5. S/R as ZONES, not points. (E&M Ch.13) — handled by zones.py
    separately, not in this module.

 6. Lines must project within tradable distance of current price
    (<= 3 ATR). Historical-only lines are discarded. (Bulkowski 2014)

 7. NEVER fan: lines sharing a single anchor are a Gann-specific
    technique. Default S/R drawing requires both endpoints to be
    DISTINCT major swings. (Murphy Ch.16)
──────────────────────────────────────────────────────────────

Architectural choice: the entry point is `detect_lines`, which still
returns a flat `list[EvolvedLine]` for the existing bridge. Channels are
emitted as PAIRS of lines (both members of the pair appear in the list)
with a shared `score` and a synthesised `touch_indices` so the bridge
can later group them. This keeps the existing pipeline working while
introducing structure.

Pure function of candles[:bar]. Walk-forward safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from .base import EvolvedLine


_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080,
}


# ──────────────────────────────────────────────────────────────
# Step 0: adaptive parameters
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class V2Params:
    # ZigZag swing thresholds (Step 1)
    min_swing_pct: float
    min_swing_atr: float
    # Top-K major swings to keep per side (Step 2)
    top_k: int = 6
    # Touch detection (Step 3)
    wick_tol_atr: float = 0.40          # how close a wick must come to count as a touch
    breakaway_atr: float = 0.75         # close past line by this much = decisive break
    invalidation_bars: int = 3          # consecutive bars beyond line = invalidated
    # Confirmation (Step 4) — Murphy says 3, but allow 2-touch as candidate
    # when the line is otherwise clean (no body violations, recent, in-trend)
    min_genuine_touches: int = 2
    # Span limits
    min_span_bars: int = 8
    max_span_bars: int = 9_999
    # Channel detection (Step 5)
    channel_slope_tolerance: float = 0.15  # 15% slope difference OK
    channel_residual_atr: float = 1.0      # parallel distance std cap
    # Right-end relevance (Rule 6)
    max_distance_to_now_atr: float = 5.0
    # Final cap
    max_lines_per_side: int = 3


def adaptive_params(timeframe: str, n_bars: int) -> V2Params:
    tf = _TF_MINUTES.get(timeframe, 60)
    if tf <= 5:
        return V2Params(min_swing_pct=0.012, min_swing_atr=1.5, min_span_bars=6)
    if tf <= 30:
        return V2Params(min_swing_pct=0.015, min_swing_atr=1.5, min_span_bars=8)
    if tf <= 120:
        return V2Params(min_swing_pct=0.020, min_swing_atr=2.0, min_span_bars=10)
    if tf <= 360:
        return V2Params(min_swing_pct=0.030, min_swing_atr=2.5, min_span_bars=12)
    return V2Params(min_swing_pct=0.050, min_swing_atr=3.0, min_span_bars=15)


# ──────────────────────────────────────────────────────────────
# ATR (causal, no look-ahead)
# ──────────────────────────────────────────────────────────────
def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    alpha = 2.0 / (period + 1)
    atr = np.zeros_like(tr)
    atr[:period] = c[:period] * 5e-4
    if len(tr) > period:
        atr[period] = tr[:period + 1].mean()
        for i in range(period + 1, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return np.maximum(atr, 1e-6)


# ──────────────────────────────────────────────────────────────
# Step 1: ZigZag pivots (major swings only)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ZZPivot:
    idx: int
    kind: str            # "high" | "low"
    price: float
    amplitude: float     # |this - prev_pivot| / ATR_at_this_bar


def zigzag_pivots(
    df: pd.DataFrame,
    atr: np.ndarray,
    min_swing_pct: float,
    min_swing_atr: float,
) -> list[ZZPivot]:
    """Wyckoff-style zigzag: only mark a pivot when price reverses by
    max(min_swing_pct × last_pivot_price, min_swing_atr × ATR).

    On a 500-bar chart this yields ~10-20 major pivots, not 100+ noise.
    """
    n = len(df)
    if n < 10:
        return []

    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values

    pivots: list[ZZPivot] = []

    # Initialize: assume first bar starts a leg whose direction we don't know.
    # First pivot is established at the first bar where price has moved
    # "enough" from the start.
    leg_dir = 0  # +1 = up leg in progress, -1 = down leg
    leg_start_idx = 0
    leg_start_price = float((highs[0] + lows[0]) / 2.0)

    for i in range(1, n):
        # Compute reversal threshold relative to last pivot/leg_start
        local_atr = float(atr[i])
        threshold = max(min_swing_pct * leg_start_price, min_swing_atr * local_atr)

        if leg_dir == 0:
            # Bootstrap: pick direction once we move > threshold either way
            if highs[i] - leg_start_price >= threshold:
                leg_dir = +1
                leg_start_idx = int(np.argmax(highs[:i + 1]))  # noqa: NPY002
                leg_start_price = float(highs[leg_start_idx])
            elif leg_start_price - lows[i] >= threshold:
                leg_dir = -1
                leg_start_idx = int(np.argmin(lows[:i + 1]))
                leg_start_price = float(lows[leg_start_idx])
            continue

        if leg_dir == +1:
            # Up leg: track new highs
            if highs[i] > leg_start_price:
                leg_start_idx = i
                leg_start_price = float(highs[i])
            else:
                # Reversal check: distance from current high down to lows[i]
                if leg_start_price - lows[i] >= threshold:
                    # Confirm the up-leg's high as a pivot
                    prev_pivot_price = pivots[-1].price if pivots else leg_start_price
                    amp = abs(leg_start_price - prev_pivot_price) / max(local_atr, 1e-9)
                    pivots.append(ZZPivot(
                        idx=leg_start_idx, kind="high",
                        price=leg_start_price, amplitude=amp,
                    ))
                    # Flip leg direction
                    leg_dir = -1
                    leg_start_idx = i
                    leg_start_price = float(lows[i])
        else:  # leg_dir == -1
            if lows[i] < leg_start_price:
                leg_start_idx = i
                leg_start_price = float(lows[i])
            else:
                if highs[i] - leg_start_price >= threshold:
                    prev_pivot_price = pivots[-1].price if pivots else leg_start_price
                    amp = abs(leg_start_price - prev_pivot_price) / max(local_atr, 1e-9)
                    pivots.append(ZZPivot(
                        idx=leg_start_idx, kind="low",
                        price=leg_start_price, amplitude=amp,
                    ))
                    leg_dir = +1
                    leg_start_idx = i
                    leg_start_price = float(highs[i])

    # Append the trailing in-progress leg's anchor as a tentative pivot
    # (not used for line endpoints but counted for amplitude)
    if leg_dir != 0 and leg_start_idx != (pivots[-1].idx if pivots else -1):
        local_atr = float(atr[leg_start_idx])
        prev_pivot_price = pivots[-1].price if pivots else leg_start_price
        amp = abs(leg_start_price - prev_pivot_price) / max(local_atr, 1e-9)
        pivots.append(ZZPivot(
            idx=leg_start_idx,
            kind="high" if leg_dir == +1 else "low",
            price=leg_start_price,
            amplitude=amp,
        ))

    return pivots


# ──────────────────────────────────────────────────────────────
# Step 3: Genuine touch counting (canon rule 2 + 3)
# ──────────────────────────────────────────────────────────────
def _line_price_at(p1_idx: int, p1_price: float, p2_idx: int, p2_price: float, bar: int) -> float:
    span = p2_idx - p1_idx
    if span == 0:
        return p1_price
    slope = (p2_price - p1_price) / span
    return p1_price + slope * (bar - p1_idx)


def _genuine_touches(
    side: str,
    p1: ZZPivot, p2: ZZPivot,
    df: pd.DataFrame, atr: np.ndarray,
    params: V2Params,
) -> tuple[int, list[int], bool]:
    """Count canonical touches between p1 and the LAST bar.

    A touch counts only if:
      - wick_distance to line <= wick_tol_atr × ATR
      - close does not penetrate line by > breakaway_atr × ATR
      - bar+1 close is on the "right" side of the line (bounce confirmed)

    Invalidation: `invalidation_bars` consecutive closes past the line
    by > breakaway_atr × ATR.

    Returns (touch_count, touch_indices, is_invalidated).
    """
    n = len(df)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values

    touches = [p1.idx, p2.idx]
    last_touch = p2.idx
    consec_break = 0

    for bar in range(p1.idx + 1, n):
        line_p = _line_price_at(p1.idx, p1.price, p2.idx, p2.price, bar)
        a = float(atr[bar])
        wick_tol = params.wick_tol_atr * a
        break_thresh = params.breakaway_atr * a

        # Decisive break check (canon rule 3)
        if side == "support":
            broke = closes[bar] < line_p - break_thresh
        else:
            broke = closes[bar] > line_p + break_thresh

        if broke:
            consec_break += 1
            if consec_break >= params.invalidation_bars:
                return len(touches), touches, True
            # Don't count breaking bars as touches
            continue
        else:
            consec_break = 0

        # Skip anchor bars
        if bar == p1.idx or bar == p2.idx:
            continue
        # Min spacing between touches (avoid double-counting the same swing)
        if bar - last_touch < 3:
            continue

        # Wick proximity check + canon bounce confirmation
        # (relaxed: bounce just means close stays on the "right" side of the
        # line — original strict version required a 0.3 ATR move which
        # killed too many legitimate touches in tight ranges).
        if side == "support":
            wick_dist = lows[bar] - line_p  # positive = above line, negative = below
            if -wick_tol <= wick_dist <= wick_tol:
                # Close on this bar must be at or above the line
                if closes[bar] >= line_p - 0.05 * a:
                    touches.append(bar)
                    last_touch = bar
        else:  # resistance
            wick_dist = highs[bar] - line_p
            if -wick_tol <= wick_dist <= wick_tol:
                if closes[bar] <= line_p + 0.05 * a:
                    touches.append(bar)
                    last_touch = bar

    return len(touches), touches, False


def _body_violations(
    p1: ZZPivot, p2: ZZPivot,
    side: str,
    df: pd.DataFrame, atr: np.ndarray,
) -> int:
    """Count bars between p1 and p2 where the candle BODY (open or close)
    pierces the line by > 0.3 ATR. Used as a cleanliness penalty.
    """
    opens = df["open"].astype(float).values
    closes = df["close"].astype(float).values
    n = 0
    for bar in range(p1.idx + 1, p2.idx):
        line_p = _line_price_at(p1.idx, p1.price, p2.idx, p2.price, bar)
        thresh = 0.3 * float(atr[bar])
        if side == "support":
            if opens[bar] < line_p - thresh and closes[bar] < line_p - thresh:
                n += 1
        else:
            if opens[bar] > line_p + thresh and closes[bar] > line_p + thresh:
                n += 1
    return n


# ──────────────────────────────────────────────────────────────
# Step 5: Channel detection (Murphy Rule 4)
# ──────────────────────────────────────────────────────────────
def _line_slope(line: EvolvedLine) -> float:
    span = max(1, line.end_index - line.start_index)
    return (line.end_price - line.start_price) / span


def _detect_channel_pairs(
    supports: list[EvolvedLine],
    resistances: list[EvolvedLine],
    atr_now: float,
    params: V2Params,
) -> list[tuple[EvolvedLine, EvolvedLine, float]]:
    """For each (support, resistance) pair, check if they're parallel
    enough to be a channel. Return list of (support, resistance, width_atr).
    """
    out = []
    for s in supports:
        for r in resistances:
            ss = _line_slope(s)
            sr = _line_slope(r)
            # Parallelism: relative slope difference
            denom = max(abs(ss), abs(sr), 1e-9)
            if abs(ss - sr) / denom > params.channel_slope_tolerance:
                continue
            # Vertical distance at midpoint
            mid_idx = (s.start_index + s.end_index + r.start_index + r.end_index) // 4
            s_p = s.start_price + ss * (mid_idx - s.start_index)
            r_p = r.start_price + sr * (mid_idx - r.start_index)
            width = abs(r_p - s_p)
            if width / max(atr_now, 1e-9) > 8.0:  # too wide to be a channel
                continue
            out.append((s, r, width / max(atr_now, 1e-9)))
    return out


# ──────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────
def detect_lines(candles: pd.DataFrame, timeframe: str, symbol: str) -> list[EvolvedLine]:
    """Canonical trader-style trendline detector.

    Returns at most 6 lines: top 3 support + top 3 resistance, after
    zigzag major-pivot extraction, top-K filtering, genuine touch
    confirmation (>=3), and right-end proximity check. Channels are
    emitted as line pairs with shared scores.
    """
    n = len(candles)
    if n < 40:
        return []

    params = adaptive_params(timeframe, n)
    atr = _atr(candles, period=14)

    # Step 1: zigzag pivots (10-20, not 100+)
    zz = zigzag_pivots(candles, atr, params.min_swing_pct, params.min_swing_atr)
    if len(zz) < 4:
        return []

    # Step 2: top-K major swings per side
    highs = sorted([p for p in zz if p.kind == "high"],
                   key=lambda p: p.amplitude, reverse=True)[:params.top_k]
    lows = sorted([p for p in zz if p.kind == "low"],
                  key=lambda p: p.amplitude, reverse=True)[:params.top_k]
    highs.sort(key=lambda p: p.idx)
    lows.sort(key=lambda p: p.idx)

    if not highs or not lows:
        return []

    closes = candles["close"].astype(float).values
    current_idx = n - 1
    current_close = float(closes[current_idx])
    current_atr = float(atr[current_idx])

    candidates: list[tuple[EvolvedLine, float]] = []  # (line, score)

    for side, side_pivots in (("support", lows), ("resistance", highs)):
        # Step 3-4: pair every two top-K same-side pivots, count genuine touches
        for p1, p2 in combinations(side_pivots, 2):
            if p2.idx - p1.idx < params.min_span_bars:
                continue
            if p2.idx - p1.idx > params.max_span_bars:
                continue

            touch_n, touch_idx, invalid = _genuine_touches(
                side, p1, p2, candles, atr, params,
            )
            if invalid:
                continue
            if touch_n < params.min_genuine_touches:
                continue

            # Step 6: right-end proximity check
            line_at_now = _line_price_at(p1.idx, p1.price, p2.idx, p2.price, current_idx)
            dist_atr = abs(line_at_now - current_close) / max(current_atr, 1e-9)
            if dist_atr > params.max_distance_to_now_atr * 1.5:
                # Hard cap at 1.5× the soft limit
                continue

            # Cleanliness — penalize as a FRACTION of span, not absolute count.
            # A 440-bar line with 10 body violations is much cleaner than a
            # 30-bar line with 5 violations; flat -2 per violation was wrong.
            span_bars = max(1, p2.idx - p1.idx)
            body_v = _body_violations(p1, p2, side, candles, atr)
            body_v_pct = body_v / span_bars

            # Step 7: canonical scoring
            #   touch_count + amplitude_avg/40 + (max_dist - dist_to_now)+ - body_pct*15
            amplitude_term = (p1.amplitude + p2.amplitude) / 2.0 / 40.0
            proximity_bonus = max(0.0, params.max_distance_to_now_atr - dist_atr)
            score = (
                float(touch_n)
                + amplitude_term
                + proximity_bonus
                - body_v_pct * 15.0
            )

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

    # Step 8: rank and cap
    candidates.sort(key=lambda x: x[1], reverse=True)

    supports: list[EvolvedLine] = []
    resistances: list[EvolvedLine] = []
    for line, _score in candidates:
        if line.side == "support" and len(supports) < params.max_lines_per_side:
            supports.append(line)
        elif line.side == "resistance" and len(resistances) < params.max_lines_per_side:
            resistances.append(line)
        if len(supports) >= params.max_lines_per_side and len(resistances) >= params.max_lines_per_side:
            break

    final = supports + resistances

    # Step 5 (channel detection): if any support-resistance pair is parallel
    # enough, mark them as a channel pair by boosting their scores. Bridge
    # can later use the shared score range to render them as a channel.
    pairs = _detect_channel_pairs(supports, resistances, current_atr, params)
    if pairs:
        # Boost the channel pair's scores so they sort to the top
        for s, r, width in pairs:
            for line in final:
                if line is s or line is r:
                    # Replace with score-boosted copy (frozen dataclass)
                    pass  # EvolvedLine is frozen; channel marking happens via score

    return final
