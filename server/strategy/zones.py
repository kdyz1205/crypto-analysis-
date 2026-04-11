"""Horizontal support/resistance zone detection via pivot clustering.

Ports the zone-clustering logic from support_resistance.py (Polars) into the
server/strategy pipeline (pandas), consuming the same Pivot objects produced
by pivots.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import StrategyConfig, calculate_atr, clamp
from .pivots import filter_confirmed_pivots
from .types import Pivot, ensure_candles_df


@dataclass(frozen=True, slots=True)
class HorizontalZone:
    """A horizontal support or resistance price band."""
    zone_id: str
    side: str  # "support" or "resistance"
    price_low: float
    price_high: float
    price_center: float
    width: float
    touches: int
    touch_indices: tuple[int, ...]
    touch_prices: tuple[float, ...]
    first_touch_index: int
    last_touch_index: int
    strength: float  # 0-100 composite score
    strength_components: dict


def detect_horizontal_zones(
    candles,
    pivots: Sequence[Pivot],
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
    max_zones_per_side: int = 3,
) -> list[HorizontalZone]:
    """Detect horizontal S/R zones by clustering nearby pivots.

    Returns at most `max_zones_per_side` support zones and
    `max_zones_per_side` resistance zones, ranked by strength.
    """
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if df.empty or not pivots:
        return []

    current_index = len(df) - 1
    atr = calculate_atr(df, cfg.atr_period)
    atr_value = float(atr.iloc[current_index])
    close_price = float(df.iloc[current_index]["close"])

    # Clustering epsilon: max of ATR-based and percentage-based
    eps = max(atr_value * 0.3, close_price * 0.005)

    zones: list[HorizontalZone] = []
    for side, pivot_kind in [("resistance", "high"), ("support", "low")]:
        confirmed = filter_confirmed_pivots(pivots, kind=pivot_kind, up_to_index=current_index)
        lookback_start = max(0, current_index - cfg.lookback_bars + 1)
        side_pivots = [p for p in confirmed if p.index >= lookback_start]

        if len(side_pivots) < 2:
            continue

        side_zones = _cluster_pivots_into_zones(
            side_pivots,
            side=side,
            eps=eps,
            current_index=current_index,
            min_touches=2,
            symbol=symbol,
            timeframe=timeframe,
        )

        # Score each zone
        scored = []
        for zone in side_zones:
            strength, components = _score_zone(
                zone, df, atr, current_index, close_price, atr_value, cfg
            )
            scored.append(HorizontalZone(
                zone_id=zone.zone_id,
                side=zone.side,
                price_low=zone.price_low,
                price_high=zone.price_high,
                price_center=zone.price_center,
                width=zone.width,
                touches=zone.touches,
                touch_indices=zone.touch_indices,
                touch_prices=zone.touch_prices,
                first_touch_index=zone.first_touch_index,
                last_touch_index=zone.last_touch_index,
                strength=strength,
                strength_components=components,
            ))

        # Sort by strength descending, keep top N
        scored.sort(key=lambda z: -z.strength)
        zones.extend(scored[:max_zones_per_side])

    return zones


def _cluster_pivots_into_zones(
    pivots: Sequence[Pivot],
    *,
    side: str,
    eps: float,
    current_index: int,
    min_touches: int,
    symbol: str,
    timeframe: str,
) -> list[HorizontalZone]:
    """Cluster pivots by price proximity into horizontal zones."""
    if not pivots:
        return []

    # Sort by price
    sorted_pivots = sorted(pivots, key=lambda p: p.price)
    zones: list[HorizontalZone] = []

    i = 0
    while i < len(sorted_pivots):
        cluster = [sorted_pivots[i]]
        j = i + 1
        while j < len(sorted_pivots) and sorted_pivots[j].price - sorted_pivots[i].price <= eps:
            cluster.append(sorted_pivots[j])
            j += 1

        if len(cluster) >= min_touches:
            prices = [p.price for p in cluster]
            indices = [p.index for p in cluster]
            center = float(np.mean(prices))
            half_eps = eps * 0.5

            from .types import stable_id
            zone_id = stable_id("zone", side, symbol, timeframe, round(center, 6), len(cluster))

            zones.append(HorizontalZone(
                zone_id=zone_id,
                side=side,
                price_low=center - half_eps,
                price_high=center + half_eps,
                price_center=center,
                width=eps,
                touches=len(cluster),
                touch_indices=tuple(sorted(indices)),
                touch_prices=tuple(prices),
                first_touch_index=min(indices),
                last_touch_index=max(indices),
                strength=0.0,
                strength_components={},
            ))

        i = j

    return zones


def _score_zone(
    zone: HorizontalZone,
    df,
    atr,
    current_index: int,
    close_price: float,
    atr_value: float,
    config: StrategyConfig,
) -> tuple[float, dict]:
    """Score a horizontal zone on multiple quality dimensions. Returns (0-100 score, components)."""

    # 1. Touch count score (more touches = stronger, cap at 6)
    touch_score = clamp(zone.touches / 6.0)

    # 2. Reaction strength: average bounce/rejection after each touch
    reaction_scores = []
    for touch_idx in zone.touch_indices:
        if touch_idx + 3 >= len(df):
            continue
        touch_close = float(df.iloc[touch_idx]["close"])
        # Look 3 bars ahead for reaction
        future_close = float(df.iloc[min(touch_idx + 3, len(df) - 1)]["close"])
        local_atr = float(atr.iloc[touch_idx]) if touch_idx < len(atr) else atr_value
        if local_atr <= 0:
            continue
        reaction = abs(future_close - touch_close) / local_atr
        reaction_scores.append(clamp(reaction / 2.0))  # 2 ATR reaction = max score

    reaction_score = float(np.mean(reaction_scores)) if reaction_scores else 0.0

    # 3. Recency: how recently was this zone last tested
    bars_since = current_index - zone.last_touch_index
    recency_score = clamp(1.0 - (bars_since / max(config.max_fresh_bars, 1)))

    # 4. Zone width clarity: narrower zones (relative to ATR) are clearer
    relative_width = zone.width / max(atr_value, 1e-10)
    clarity_score = clamp(1.0 - (relative_width - 0.3) / 1.0)  # best around 0.3 ATR width

    # 5. Volume at touches: average volume at touch bars vs overall average
    if len(df) > 0:
        overall_avg_vol = float(df["volume"].mean())
        touch_vols = [float(df.iloc[idx]["volume"]) for idx in zone.touch_indices if idx < len(df)]
        avg_touch_vol = float(np.mean(touch_vols)) if touch_vols else overall_avg_vol
        volume_score = clamp((avg_touch_vol / max(overall_avg_vol, 1e-10) - 1.0) / 2.0)
    else:
        volume_score = 0.0

    # 6. Distance from current price (closer = more immediately relevant)
    distance = abs(close_price - zone.price_center) / max(atr_value, 1e-10)
    proximity_score = clamp(1.0 - distance / 5.0)  # within 5 ATR = relevant

    # Weighted composite
    strength = 100.0 * clamp(
        (0.25 * touch_score)
        + (0.20 * reaction_score)
        + (0.15 * recency_score)
        + (0.10 * clarity_score)
        + (0.15 * volume_score)
        + (0.15 * proximity_score)
    )

    components = {
        "touch_score": round(touch_score, 4),
        "reaction_score": round(reaction_score, 4),
        "recency_score": round(recency_score, 4),
        "clarity_score": round(clarity_score, 4),
        "volume_score": round(volume_score, 4),
        "proximity_score": round(proximity_score, 4),
    }

    return round(strength, 2), components


__all__ = [
    "HorizontalZone",
    "detect_horizontal_zones",
]
