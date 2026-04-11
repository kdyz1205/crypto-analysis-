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

    # Clustering epsilon: adaptive by timeframe
    # Shorter timeframes use tighter clustering to avoid oversized zones
    tf_eps_scale = _timeframe_eps_scale(timeframe)
    eps = max(
        atr_value * cfg.zone_eps_atr_mult * tf_eps_scale,
        close_price * cfg.zone_eps_pct * tf_eps_scale,
    )

    zones: list[HorizontalZone] = []
    for side, pivot_kind in [("resistance", "high"), ("support", "low")]:
        confirmed = filter_confirmed_pivots(pivots, kind=pivot_kind, up_to_index=current_index)
        lookback_start = max(0, current_index - cfg.lookback_bars + 1)
        side_pivots = [p for p in confirmed if p.index >= lookback_start]

        if len(side_pivots) < cfg.zone_min_touches:
            continue

        side_zones = _cluster_pivots_into_zones(
            side_pivots,
            side=side,
            eps=eps,
            current_index=current_index,
            min_touches=cfg.zone_min_touches,
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

        # Filter out broken zones — price has moved far past them
        scored = [z for z in scored if not _is_zone_broken(z, df, atr_value, close_price)]

        # Sort by strength descending, keep top N
        scored.sort(key=lambda z: -z.strength)
        zones.extend(scored[:max_zones_per_side])

    # Remove S/R conflicts — same price can't be both support and resistance
    zones = _resolve_sr_conflicts(zones, atr_value)

    # Detect flip zones — broken S/R that flipped role
    zones = _detect_flip_zones(zones, df, close_price, atr_value)

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
        # Chain clustering: each pivot must be within eps of the PREVIOUS one (not the first)
        while j < len(sorted_pivots) and sorted_pivots[j].price - cluster[-1].price <= eps:
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

    # 2. Reaction strength: directional bounce after each touch
    # Support: price should go UP after touch. Resistance: price should go DOWN.
    reaction_scores = []
    for touch_idx in zone.touch_indices:
        if touch_idx + 3 >= len(df):
            continue
        touch_close = float(df.iloc[touch_idx]["close"])
        future_close = float(df.iloc[min(touch_idx + 3, len(df) - 1)]["close"])
        local_atr = float(atr.iloc[touch_idx]) if touch_idx < len(atr) else atr_value
        if local_atr <= 0:
            continue
        # Direction-aware: support expects up, resistance expects down
        if zone.side == "support":
            reaction = (future_close - touch_close) / local_atr  # positive = bounced up (good)
        else:
            reaction = (touch_close - future_close) / local_atr  # positive = rejected down (good)
        reaction_scores.append(clamp(reaction / 2.0))  # 2 ATR directional reaction = max

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

    # 7. Trend context: support is more reliable in uptrends, resistance in downtrends
    trend_score = _trend_alignment_score(df, zone.side, current_index, config)

    # 8. Volume failure: touches on LOW volume reduce reliability
    volume_failure_score = _volume_failure_score(df, zone.touch_indices, current_index)

    # Weighted composite
    strength = 100.0 * clamp(
        (0.20 * touch_score)
        + (0.15 * reaction_score)
        + (0.10 * recency_score)
        + (0.08 * clarity_score)
        + (0.12 * volume_score)
        + (0.10 * proximity_score)
        + (0.15 * trend_score)
        + (0.10 * volume_failure_score)
    )

    components = {
        "touch_score": round(touch_score, 4),
        "reaction_score": round(reaction_score, 4),
        "recency_score": round(recency_score, 4),
        "clarity_score": round(clarity_score, 4),
        "volume_score": round(volume_score, 4),
        "proximity_score": round(proximity_score, 4),
        "trend_score": round(trend_score, 4),
        "volume_failure_score": round(volume_failure_score, 4),
    }

    return round(strength, 2), components


def _trend_alignment_score(df, side: str, current_index: int, config: StrategyConfig) -> float:
    """Score 0-1: support zones score higher in uptrends, resistance in downtrends."""
    ema_period = config.trend_ema_period
    if current_index < ema_period:
        return 0.5  # neutral when insufficient data

    close = df["close"].astype(float)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    current_close = float(close.iloc[current_index])
    current_ema = float(ema.iloc[current_index])
    prev_ema = float(ema.iloc[current_index - 1])

    ema_slope = (current_ema - prev_ema) / max(abs(current_ema), 1e-10)
    price_vs_ema = (current_close - current_ema) / max(abs(current_ema), 1e-10)

    if side == "support":
        # Uptrend = support is reliable
        trend_signal = clamp(ema_slope * 1000) * 0.5 + clamp(price_vs_ema * 50) * 0.5
    else:
        # Downtrend = resistance is reliable
        trend_signal = clamp(-ema_slope * 1000) * 0.5 + clamp(-price_vs_ema * 50) * 0.5

    return clamp(trend_signal)


def _volume_failure_score(df, touch_indices: tuple[int, ...], current_index: int) -> float:
    """Score 0-1: penalize zones where touches happened on LOW volume.
    High volume touches = 1.0 (reliable). Low volume touches = 0.0 (unreliable).
    """
    if len(df) == 0 or not touch_indices:
        return 0.5

    overall_avg = float(df["volume"].astype(float).mean())
    if overall_avg <= 0:
        return 0.5

    ratios = []
    for idx in touch_indices:
        if idx >= len(df):
            continue
        vol = float(df.iloc[idx]["volume"])
        ratios.append(vol / overall_avg)

    if not ratios:
        return 0.5

    avg_ratio = float(np.mean(ratios))
    # ratio < 0.5 = weak volume at touches → score 0
    # ratio > 1.5 = strong volume at touches → score 1
    return clamp((avg_ratio - 0.5) / 1.0)


def _is_zone_broken(zone: HorizontalZone, df, atr_value: float, close_price: float) -> bool:
    """A zone is 'broken' if price has moved decisively past it and stayed there.

    Support broken = price well below zone and not coming back
    Resistance broken = price well above zone and not coming back
    """
    distance = close_price - zone.price_center
    threshold = atr_value * 2.0  # must be 2 ATR past the zone

    if zone.side == "resistance":
        # Price is far ABOVE resistance — resistance is broken
        if distance > threshold:
            # Verify: last 5 candle closes are all above the zone
            recent_closes = df["close"].astype(float).iloc[-5:]
            if all(c > zone.price_high for c in recent_closes):
                return True
    else:
        # Price is far BELOW support — support is broken
        if distance < -threshold:
            recent_closes = df["close"].astype(float).iloc[-5:]
            if all(c < zone.price_low for c in recent_closes):
                return True
    return False


def _resolve_sr_conflicts(zones: list[HorizontalZone], atr_value: float) -> list[HorizontalZone]:
    """Remove zones where support and resistance overlap at the same price.
    Keep the one with higher strength."""
    if not zones:
        return zones

    conflict_threshold = atr_value * 0.5  # within 0.5 ATR = conflict
    to_remove = set()

    for i, z1 in enumerate(zones):
        for j, z2 in enumerate(zones):
            if i >= j or z1.side == z2.side:
                continue
            if abs(z1.price_center - z2.price_center) < conflict_threshold:
                # Keep the stronger one
                loser = i if z1.strength < z2.strength else j
                to_remove.add(loser)

    return [z for i, z in enumerate(zones) if i not in to_remove]


def _detect_flip_zones(zones: list[HorizontalZone], df, close_price: float, atr_value: float) -> list[HorizontalZone]:
    """Detect zones where price has crossed through — support becomes resistance and vice versa.
    If current price is significantly below a support zone, that zone has flipped to resistance.
    If current price is significantly above a resistance zone, it has flipped to support.
    """
    flipped = []
    threshold = atr_value * 1.5  # must be 1.5 ATR past the zone to count as flipped

    for zone in zones:
        if zone.side == "support" and close_price < zone.price_low - threshold:
            # Price fell well below support → this is now resistance
            flipped.append(HorizontalZone(
                zone_id=zone.zone_id + "_flip",
                side="resistance",  # flipped!
                price_low=zone.price_low, price_high=zone.price_high,
                price_center=zone.price_center, width=zone.width,
                touches=zone.touches, touch_indices=zone.touch_indices,
                touch_prices=zone.touch_prices,
                first_touch_index=zone.first_touch_index,
                last_touch_index=zone.last_touch_index,
                strength=zone.strength * 0.8,  # slightly weaker than original
                strength_components={**zone.strength_components, "flipped": True},
            ))
        elif zone.side == "resistance" and close_price > zone.price_high + threshold:
            # Price rose well above resistance → this is now support
            flipped.append(HorizontalZone(
                zone_id=zone.zone_id + "_flip",
                side="support",
                price_low=zone.price_low, price_high=zone.price_high,
                price_center=zone.price_center, width=zone.width,
                touches=zone.touches, touch_indices=zone.touch_indices,
                touch_prices=zone.touch_prices,
                first_touch_index=zone.first_touch_index,
                last_touch_index=zone.last_touch_index,
                strength=zone.strength * 0.8,
                strength_components={**zone.strength_components, "flipped": True},
            ))

    return zones + flipped


def _timeframe_eps_scale(timeframe: str) -> float:
    """Shorter timeframes get tighter clustering radius to avoid oversized zones."""
    scales = {
        "1m": 0.4, "3m": 0.5, "5m": 0.6,
        "15m": 0.7, "1h": 0.85, "4h": 1.0,
        "1d": 1.2, "1w": 1.5,
    }
    return scales.get(timeframe, 1.0)


__all__ = [
    "HorizontalZone",
    "detect_horizontal_zones",
]
