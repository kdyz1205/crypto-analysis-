"""Pivot detection with reaction-strength filtering.

A "pivot" is not just a local extremum; it's a local extremum followed by a
measurable reaction. We want the points where the market visibly respected the
level — that's what human eyes lock onto.

Reaction strength is measured in ATR units so it normalizes across regimes.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


@dataclass
class Pivot:
    idx: int          # bar index
    price: float      # the pivot price
    kind: str         # 'high' or 'low'
    strength: float   # reaction in ATR units (post-pivot move away from level)
    atr_at_pivot: float


def find_pivots(
    df: pd.DataFrame,
    k: int = 5,
    min_reaction_atr: float = 0.5,
    atr_period: int = 14,
) -> list[Pivot]:
    """Fractal pivots with forward-reaction filter.

    A bar i is a swing low if df.low[i] == min(low[i-k..i+k]) AND the post-pivot
    move upward within the next `k` bars is ≥ `min_reaction_atr` × ATR. Symmetric
    for highs. This drops the noise pivots that had no market reaction.
    """
    atr = compute_atr(df, atr_period).values
    lows = df["low"].values
    highs = df["high"].values
    n = len(df)
    pivots: list[Pivot] = []

    for i in range(k, n - k):
        cur_atr = atr[i]
        if not np.isfinite(cur_atr) or cur_atr == 0:
            continue

        # Swing low
        window_low = lows[i - k:i + k + 1].min()
        if lows[i] == window_low:
            # Reaction: max high in [i+1, i+k] minus pivot low
            fwd_high = highs[i + 1:i + k + 1].max()
            reaction = (fwd_high - lows[i]) / cur_atr
            if reaction >= min_reaction_atr:
                pivots.append(Pivot(i, float(lows[i]), "low", float(reaction), float(cur_atr)))

        # Swing high
        window_high = highs[i - k:i + k + 1].max()
        if highs[i] == window_high:
            fwd_low = lows[i + 1:i + k + 1].min()
            reaction = (highs[i] - fwd_low) / cur_atr
            if reaction >= min_reaction_atr:
                pivots.append(Pivot(i, float(highs[i]), "high", float(reaction), float(cur_atr)))

    return pivots
