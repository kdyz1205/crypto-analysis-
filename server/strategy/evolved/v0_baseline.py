"""v0_baseline: wraps the current production trendlines.py as the evolution baseline.

Every later variant competes against this. If no variant beats it, we keep v0.
"""

from __future__ import annotations

import pandas as pd

from .base import EvolvedLine
from ..config import StrategyConfig
from ..pivots import detect_pivots
from ..trendlines import detect_trendlines


def detect_lines(
    candles: pd.DataFrame,
    timeframe: str,
    symbol: str,
) -> list[EvolvedLine]:
    """Run the production trendline detector and project its output into
    the minimal EvolvedLine shape.

    Returns both active_lines (confirmed) and top candidate_lines.
    """
    if candles is None or len(candles) < 20:
        return []

    cfg = StrategyConfig()
    pivots = detect_pivots(candles, cfg)
    result = detect_trendlines(candles, pivots, cfg, symbol=symbol, timeframe=timeframe)

    out: list[EvolvedLine] = []
    seen_ids: set[str] = set()

    for line in list(result.active_lines) + list(result.candidate_lines):
        if line.line_id in seen_ids:
            continue
        seen_ids.add(line.line_id)
        if line.state == "invalidated" or line.state == "expired":
            continue
        a_idx, b_idx = line.anchor_indices
        a_price, b_price = line.anchor_prices
        if b_idx <= a_idx:
            continue
        touch_idx = tuple(int(i) for i in (line.confirming_touch_indices or ()))
        # Ensure anchors are included in touch sequence
        combined = sorted(set((int(a_idx), int(b_idx), *touch_idx)))
        out.append(
            EvolvedLine(
                side=line.side,
                start_index=int(a_idx),
                end_index=int(b_idx),
                start_price=float(a_price),
                end_price=float(b_price),
                touch_count=int(line.confirming_touch_count),
                touch_indices=tuple(combined),
                score=float(getattr(line, "score", 0.0)),
            )
        )
    return out
