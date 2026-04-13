"""v1a_filtered: v1_clean + reflection-driven filters.

Filters derived from v1 backtest reflection (data/evolution/rounds/round_00):
  1. Same-trend only: drop downsloping support lines and upsloping resistance.
     (Reflection: counter-trend lines have ~same WR but lower avg_R.)
  2. Span pct in [0.1, 0.6]: avoid both too-short and too-long lines.
     (Sweet spot 0.2-0.6 had avg_R 0.27-0.40; 0.6+ drops to 0.10.)
  3. Skip LOW volatility regimes at detection time.
     (Reflection: low vol avg_R = -0.45 — systematic loser.)
  4. Recency bonus in score: prefer lines whose right anchor is recent.

All other logic is identical to v1_clean — this keeps the A/B clean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EvolvedLine
from .v1_clean import (
    _atr,
    _find_pivots,
    _count_confirming_touches,
    adaptive_params,
)


def _vol_regime(atr_at_bar: float, close_at_bar: float) -> str:
    if close_at_bar <= 0:
        return "normal"
    pct = atr_at_bar / close_at_bar
    if pct < 0.005:
        return "low"
    if pct > 0.02:
        return "high"
    return "normal"


def detect_lines(candles: pd.DataFrame, timeframe: str, symbol: str) -> list[EvolvedLine]:
    n = len(candles)
    if n < 40:
        return []

    params = adaptive_params(timeframe, n)
    atr = _atr(candles, period=14)
    pivots = _find_pivots(candles, width=params.pivot_width, atr=atr)
    if len(pivots) < 2:
        return []

    # Use the SAME pivot pool as v1_clean for clean A/B comparison:
    # top_k prominence-ranked, then sorted by idx for stable iteration.
    highs = sorted(
        sorted([p for p in pivots if p.kind == "high"],
               key=lambda p: p.prominence, reverse=True)[:params.top_k_pivots],
        key=lambda p: p.idx,
    )
    lows = sorted(
        sorted([p for p in pivots if p.kind == "low"],
               key=lambda p: p.prominence, reverse=True)[:params.top_k_pivots],
        key=lambda p: p.idx,
    )

    closes = candles["close"].astype(float).values
    lookback = min(n, 500)

    lines: list[EvolvedLine] = []
    seen: set[tuple] = set()

    for side, side_pivots in (("resistance", highs), ("support", lows)):
        for i, p1 in enumerate(side_pivots):
            for p2 in side_pivots[i + 1:]:
                span = p2.idx - p1.idx
                if span < params.min_span_bars or span > params.max_span_bars:
                    continue

                # Filter 2: span_pct of available
                span_pct = span / lookback
                if not (0.08 <= span_pct <= 0.65):
                    continue

                # Filter 1: same-trend only
                # slope per bar (as fraction of price)
                if p1.price <= 0:
                    continue
                slope_pct = (p2.price - p1.price) / p1.price / span
                if side == "support" and slope_pct < -0.0005:
                    continue  # downsloping support = counter-trend
                if side == "resistance" and slope_pct > 0.0005:
                    continue  # upsloping resistance = counter-trend

                sig = (side, p1.idx, p2.idx)
                if sig in seen:
                    continue
                seen.add(sig)

                # Filter 3: vol regime at line's END (recent) bar
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

                # Score: touches × prominence × recency_bonus
                recency_bonus = 1.0 + 0.5 * (p2.idx / max(n - 1, 1))  # 1.0..1.5
                score = touch_count * (p1.prominence + p2.prominence) / 2.0 * recency_bonus

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
    return lines[:20]
