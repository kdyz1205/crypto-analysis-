"""v1b_recency: v1a + stronger recency bias, span sweet-spot, 15m skip, proximity check.

Evolution from v1a based on further trace analysis:
  1. 15m TF has negative avg_R (-0.19) — skip it entirely.
     (From reflection: 15m 444 trades @ -83R. The noise is eating the edge.)
  2. Span sweet spot is 0.4-0.6 (avg_R +0.40). Bias score toward this range
     with a Gaussian-style penalty outside it.
  3. Recency: lines whose RIGHT anchor (p2) is within the last 25% of data
     get a 2x score boost. Lines older than 60% get a 0.5x penalty.
     (Rationale: we trade setups that approach NOW, not ancient history.)
  4. Proximity check: the projected line price at the CURRENT bar must be
     within 3 ATR of the current close. Lines projecting to price levels
     nowhere near current price are unreachable — they're just geometry.

Everything else inherits from v1a_filtered.
"""

from __future__ import annotations

import math
import pandas as pd

from .base import EvolvedLine
from .v1_clean import _atr, _find_pivots, _count_confirming_touches, adaptive_params


def _vol_regime(atr_at: float, close_at: float) -> str:
    if close_at <= 0:
        return "normal"
    pct = atr_at / close_at
    if pct < 0.005:
        return "low"
    if pct > 0.02:
        return "high"
    return "normal"


def _span_score_bias(span_pct: float) -> float:
    """Gaussian-ish weight peaking at 0.5 of lookback."""
    # Peak at 0.5, stddev 0.15 → multiplier 0.3..1.0
    if span_pct < 0.08 or span_pct > 0.80:
        return 0.0
    return math.exp(-((span_pct - 0.5) ** 2) / (2 * 0.18 ** 2))


def _recency_bonus(end_idx: int, n: int) -> float:
    """Right-anchor position as a fraction of available data → bonus.
    Late lines (>75% through) get 2×, middle get 1×, old (<40%) get 0.5×.
    """
    if n <= 1:
        return 1.0
    pct = end_idx / (n - 1)
    if pct >= 0.75:
        return 2.0
    if pct >= 0.4:
        return 1.0 + (pct - 0.4) * 2.86  # ramp 1.0 → 2.0 across 0.4-0.75
    return 0.5 + pct * 1.25  # ramp 0.5 → 1.0 across 0.0-0.4


def detect_lines(candles: pd.DataFrame, timeframe: str, symbol: str) -> list[EvolvedLine]:
    n = len(candles)
    if n < 40:
        return []

    # Filter 1: skip 15m entirely (reflection-driven)
    if timeframe in ("1m", "3m", "5m", "15m"):
        return []

    params = adaptive_params(timeframe, n)
    atr = _atr(candles, period=14)
    pivots = _find_pivots(candles, width=params.pivot_width, atr=atr)
    if len(pivots) < 2:
        return []

    highs = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.idx)
    lows = sorted([p for p in pivots if p.kind == "low"], key=lambda p: p.idx)

    closes = candles["close"].astype(float).values
    current_close = float(closes[-1])
    current_atr = float(atr[-1])
    lookback = min(n, 500)

    lines: list[EvolvedLine] = []
    seen: set[tuple] = set()

    for side, side_pivots in (("resistance", highs), ("support", lows)):
        for i, p1 in enumerate(side_pivots):
            for p2 in side_pivots[i + 1:]:
                span = p2.idx - p1.idx
                if span < params.min_span_bars or span > params.max_span_bars:
                    continue

                span_pct = span / lookback
                span_weight = _span_score_bias(span_pct)
                if span_weight == 0:
                    continue

                if p1.price <= 0:
                    continue
                slope_pct = (p2.price - p1.price) / p1.price / span

                # Same-trend filter
                if side == "support" and slope_pct < -0.0005:
                    continue
                if side == "resistance" and slope_pct > 0.0005:
                    continue

                sig = (side, p1.idx, p2.idx)
                if sig in seen:
                    continue
                seen.add(sig)

                # Vol regime filter
                end_bar = p2.idx
                vol = _vol_regime(atr[end_bar], closes[end_bar])
                if vol == "low":
                    continue

                touch_count, touch_bars, invalidated = _count_confirming_touches(
                    candles, atr, side, p1, p2,
                    params.tolerance_atr, params.break_atr,
                    params.min_bars_between_touches,
                )
                if invalidated:
                    continue
                if touch_count < params.min_touches:
                    continue

                # Filter 4: proximity check — line projected to current bar
                # must be within 3 ATR of current price
                slope = (p2.price - p1.price) / max(1, span)
                line_at_now = p1.price + slope * (n - 1 - p1.idx)
                distance = abs(line_at_now - current_close)
                if distance > 3.0 * current_atr:
                    continue

                # Score: touches × prominence × span_weight × recency_bonus
                anchor_prom = (p1.prominence + p2.prominence) / 2.0
                recency = _recency_bonus(p2.idx, n)
                score = touch_count * anchor_prom * span_weight * recency

                lines.append(EvolvedLine(
                    side=side,
                    start_index=p1.idx,
                    end_index=p2.idx,
                    start_price=p1.price,
                    end_price=p2.price,
                    touch_count=touch_count,
                    touch_indices=tuple(sorted(set(touch_bars))),
                    score=float(score),
                ))

    lines.sort(key=lambda l: l.score, reverse=True)
    # Cap to top 12 (tighter than v1a's 20 — "少而精" philosophy)
    return lines[:12]
