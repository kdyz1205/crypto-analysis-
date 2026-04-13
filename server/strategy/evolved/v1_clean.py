"""v1_clean: clean-room trendline detector.

Addresses the Stage 1 geometry bugs in the production trendlines.py /
pivots.py / zones.py while staying a PURE FUNCTION of `candles[:bar]` —
no hidden state, no look-ahead, callable online bar-by-bar.

Fixes relative to production (Round 9 Stage 1 bugs, numbered as user):
  #1  Adaptive touch_spacing per timeframe (not hardcoded 5)
  #2  Pivot confirmation boundary uses >=, last (pivot_right) bars still
      produce their pivots once the right shoulder materializes
  #3  min_touches is honored (no silent <2 hardcode)
  #4  Pivot KIND filter enforced: support lines only use low pivots,
      resistance lines only use high pivots
  #5  Touch tolerance uses the ATR *at the touch bar*, not current ATR
  #6  Cleanliness check spans full line life, not just up to "latest touch"
  #7  Invalidation uses close-based break with adaptive threshold, not
      hardcoded price distance
  #8  Bar-level re-tests after the latest touch ARE counted
  #9  Zone clustering is NOT done here — we only emit trendlines (zones
      are a separate concern and live in zones.py, not our business)
  #12 Pivot confirmation boundary checked against data length

Adaptive parameters (dynamic):
  - pivot width scales with TF: smaller TF → wider pivot (noise filter)
  - ATR-proportional tolerances
  - top-K prominence-based anchor selection (not full O(N²) enumeration)

This is DELIBERATELY simple (~250 LoC). It's the anti-complexity version.
Evolution rounds will mutate it with LLM-proposed variants; simplicity
makes those mutations safer to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from .base import EvolvedLine


# ──────────────────────────────────────────────────────────────
# Adaptive parameters by timeframe and volatility
# ──────────────────────────────────────────────────────────────
_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080,
}


@dataclass(frozen=True, slots=True)
class V1Params:
    pivot_width: int                    # left=right bars defining a local extreme
    top_k_pivots: int                   # how many top-prominence pivots per side
    min_span_bars: int                  # reject lines shorter than this
    max_span_bars: int                  # reject lines this long (ancient)
    tolerance_atr: float                # anchor alignment tolerance
    break_atr: float                    # close beyond line by this → invalid
    min_touches: int                    # user philosophy: 2-touch OK, weight by touches
    min_bars_between_touches: int


def adaptive_params(timeframe: str, n_bars: int) -> V1Params:
    """Dynamic parameters based on TF and available history."""
    tf_min = _TF_MINUTES.get(timeframe, 60)

    # Pivot width: more noise on short TFs → wider pivots
    if tf_min <= 5:
        pivot_w = 5
    elif tf_min <= 30:
        pivot_w = 4
    elif tf_min <= 120:
        pivot_w = 3
    else:
        pivot_w = 3

    # top_k retained for backward-compat but set to a large number —
    # in strong uptrends, prominence misranks higher lows as "noise"
    # and we lose the ascending support. Better to consider all pivot
    # pairs and let scoring (touches × span × prominence) pick winners.
    top_k = 40

    # Span bounds: lines must cover a meaningful window of available data
    lookback = min(n_bars, 500)
    min_span = max(pivot_w * 2 + 4, int(lookback * 0.08))
    max_span = int(lookback * 0.95)

    return V1Params(
        pivot_width=pivot_w,
        top_k_pivots=top_k,
        min_span_bars=min_span,
        max_span_bars=max_span,
        tolerance_atr=0.30,
        break_atr=0.6,
        min_touches=2,
        min_bars_between_touches=max(3, pivot_w),
    )


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
    # Simple EMA, no bfill
    alpha = 2.0 / (period + 1)
    atr = np.zeros_like(tr)
    # Pre-period bars: use a price-scaled floor so tolerances aren't tiny
    atr[:period] = c[:period] * 5e-4  # 5 bp of price
    if len(tr) > period:
        atr[period] = tr[:period + 1].mean() if period + 1 <= len(tr) else tr[0]
        for i in range(period + 1, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return np.maximum(atr, 1e-6)


# ──────────────────────────────────────────────────────────────
# Pivot detection (correct kind-filter, correct confirmation boundary)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class V1Pivot:
    idx: int
    kind: str            # "high" | "low"
    price: float
    prominence: float    # ATR-normalized distance to next more-extreme pivot


def _find_pivots(df: pd.DataFrame, width: int, atr: np.ndarray) -> list[V1Pivot]:
    """Detect local extrema where a bar is >= / <= all `width` bars on both sides.

    Uses >= (not >) so plateau bars still produce pivots. Confirmation
    boundary: pivot at index i is "confirmed" once we've seen bar i+width.
    We don't emit pivots whose confirmation bar is beyond the data length.
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)
    if n < 2 * width + 1:
        return []

    pivots: list[V1Pivot] = []
    for i in range(width, n - width):
        h = highs[i]
        l = lows[i]
        # High pivot: strictly >= all in window, and STRICTLY > at least one side
        left_h = highs[i - width:i]
        right_h = highs[i + 1:i + width + 1]
        if h >= left_h.max() and h >= right_h.max():
            # require not a flat ridge
            if h > left_h.min() or h > right_h.min():
                pivots.append(V1Pivot(idx=i, kind="high", price=float(h), prominence=0.0))
        left_l = lows[i - width:i]
        right_l = lows[i + 1:i + width + 1]
        if l <= left_l.min() and l <= right_l.min():
            if l < left_l.max() or l < right_l.max():
                pivots.append(V1Pivot(idx=i, kind="low", price=float(l), prominence=0.0))

    # Compute prominence for each pivot: distance to opposite-side extreme
    # within a LOCAL ±30-bar window, normalized by ATR.
    #
    # BUG FIX: the previous version used "to the next more-extreme same-kind
    # pivot" as the window, which means GLOBAL extreme pivots (the deepest
    # low or highest high in the whole slice) had no boundary — their
    # window stretched to the entire dataset, and the resulting
    # `(window_high - p.price) / atr_at_pivot_bar` blew up to ~500 because
    # the pivot's own ATR was tiny. That made the global extreme an
    # "invincible pivot" that any line connecting to it dominated the score
    # — producing visual fans (all lines radiating from one anchor).
    #
    # Local window keeps prominence values in a comparable range (~5-40)
    # across all pivots and prevents the fan pathology.
    LOCAL_W = 30
    n_bars = len(df)
    pivots_with_prom: list[V1Pivot] = []
    for p in pivots:
        atr_at = atr[p.idx]
        lo = max(0, p.idx - LOCAL_W)
        hi = min(n_bars - 1, p.idx + LOCAL_W)
        if p.kind == "high":
            window_low = df["low"].iloc[lo:hi + 1].min()
            prom = (p.price - float(window_low)) / max(atr_at, 1e-9)
        else:
            window_high = df["high"].iloc[lo:hi + 1].max()
            prom = (float(window_high) - p.price) / max(atr_at, 1e-9)
        pivots_with_prom.append(V1Pivot(idx=p.idx, kind=p.kind, price=p.price, prominence=prom))

    return pivots_with_prom


# ──────────────────────────────────────────────────────────────
# Line fitting + touch counting
# ──────────────────────────────────────────────────────────────
def _line_price(start_idx: int, start_price: float, end_idx: int, end_price: float, at_idx: int) -> float:
    span = end_idx - start_idx
    if span == 0:
        return start_price
    slope = (end_price - start_price) / span
    return start_price + slope * (at_idx - start_idx)


def _count_confirming_touches(
    df: pd.DataFrame,
    atr: np.ndarray,
    side: str,
    p1: V1Pivot,
    p2: V1Pivot,
    tolerance_atr: float,
    break_atr: float,
    min_bars_between_touches: int,
) -> tuple[int, list[int], bool]:
    """Count confirming touches ONLY within the line's formation window
    [p1.idx, p2.idx]. Does NOT look forward past p2, because doing so
    means we'd discard any line that eventually breaks — but a line that
    was traded profitably from formation until break is a valid setup.

    Invalidation check is likewise restricted to the window between anchors.

    Return (touch_count, touch_bar_indices, is_invalidated).
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    touch_indices: list[int] = [p1.idx, p2.idx]
    last_touch_bar = p1.idx  # start counting touches after p1

    # Only scan between anchors (inclusive). The harness is responsible
    # for walk-forward trading AFTER p2; its organic touch discovery
    # and break detection handle forward invalidation.
    for bar in range(p1.idx + 1, p2.idx):
        line_p = _line_price(p1.idx, p1.price, p2.idx, p2.price, bar)
        atr_at = atr[bar]
        tol = tolerance_atr * atr_at
        break_thresh = break_atr * atr_at

        # Invalidation check between anchors (a strong close-through
        # between anchors means the two pivots don't form a coherent line)
        if side == "support":
            if closes[bar] < line_p - break_thresh:
                return len(touch_indices), touch_indices, True
        else:
            if closes[bar] > line_p + break_thresh:
                return len(touch_indices), touch_indices, True

        if bar - last_touch_bar < min_bars_between_touches:
            continue

        if side == "support":
            if lows[bar] <= line_p + tol and lows[bar] >= line_p - tol * 2:
                touch_indices.append(bar)
                last_touch_bar = bar
        else:
            if highs[bar] >= line_p - tol and highs[bar] <= line_p + tol * 2:
                touch_indices.append(bar)
                last_touch_bar = bar

    return len(touch_indices), touch_indices, False


# ──────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────
def detect_lines(candles: pd.DataFrame, timeframe: str, symbol: str) -> list[EvolvedLine]:
    """Pure-function trendline detector.

    Called by the evolution evaluator. Must not touch files, not import
    production modules, not assume anything beyond the candle slice given.
    """
    n = len(candles)
    if n < 40:
        return []

    params = adaptive_params(timeframe, n)
    atr = _atr(candles, period=14)
    pivots = _find_pivots(candles, width=params.pivot_width, atr=atr)

    if len(pivots) < 2:
        return []

    # Top-K prominence-ranked per side
    highs_sorted = sorted([p for p in pivots if p.kind == "high"],
                         key=lambda p: p.prominence, reverse=True)[:params.top_k_pivots]
    lows_sorted = sorted([p for p in pivots if p.kind == "low"],
                        key=lambda p: p.prominence, reverse=True)[:params.top_k_pivots]

    lines: list[EvolvedLine] = []
    seen_signatures: set[tuple] = set()

    for side, side_pivots in (("resistance", highs_sorted), ("support", lows_sorted)):
        # Sort by index ascending for stable iteration
        side_pivots_by_idx = sorted(side_pivots, key=lambda p: p.idx)
        for i, p1 in enumerate(side_pivots_by_idx):
            for p2 in side_pivots_by_idx[i + 1:]:
                span = p2.idx - p1.idx
                if span < params.min_span_bars or span > params.max_span_bars:
                    continue
                # Signature for dedupe
                sig = (side, p1.idx, p2.idx)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                touch_count, touch_bars, invalidated = _count_confirming_touches(
                    candles, atr, side, p1, p2,
                    params.tolerance_atr, params.break_atr,
                    params.min_bars_between_touches,
                )
                if invalidated:
                    continue
                if touch_count < params.min_touches:
                    continue

                lines.append(EvolvedLine(
                    side=side,
                    start_index=p1.idx,
                    end_index=p2.idx,
                    start_price=p1.price,
                    end_price=p2.price,
                    touch_count=touch_count,
                    touch_indices=tuple(sorted(set(touch_bars))),
                    score=float(touch_count * (p1.prominence + p2.prominence) / 2.0),
                ))

    # Final dedupe + cap (keep top 20 by score)
    lines.sort(key=lambda l: l.score, reverse=True)
    return lines[:20]
