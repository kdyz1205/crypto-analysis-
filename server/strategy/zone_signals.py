"""Generate trading signals from horizontal S/R zones.

When price approaches a high-quality horizontal zone, generate entry signals
with stop loss just beyond the zone and take profit at the next opposing zone.
"""

from __future__ import annotations

from typing import Sequence

from .config import StrategyConfig, calculate_atr, clamp
from .types import StrategySignal, ensure_candles_df, stable_id
from .zones import HorizontalZone


def generate_zone_signals(
    candles,
    zones: Sequence[HorizontalZone],
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
) -> list[StrategySignal]:
    """Generate pre-limit signals when price is near a high-quality S/R zone.

    For support zones: generate long signal when price is within arm distance above the zone.
    For resistance zones: generate short signal when price is within arm distance below the zone.
    """
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if df.empty or not zones:
        return []

    current_index = len(df) - 1
    atr = calculate_atr(df, cfg.atr_period)
    atr_value = float(atr.iloc[current_index])
    close_price = float(df.iloc[current_index]["close"])
    timestamp = df.iloc[current_index]["timestamp"]

    # Arm distance: how close price must be to zone to trigger
    arm_dist = max(atr_value * 0.5, close_price * 0.005)

    support_zones = sorted([z for z in zones if z.side == "support"], key=lambda z: -z.strength)
    resist_zones = sorted([z for z in zones if z.side == "resistance"], key=lambda z: -z.strength)

    signals: list[StrategySignal] = []

    for zone in support_zones:
        # Price must be near or touching the zone from above
        distance_to_zone = close_price - zone.price_high
        if distance_to_zone < -arm_dist or distance_to_zone > arm_dist * 2:
            continue  # too far or already below

        entry_price = zone.price_high + atr_value * 0.02  # slightly above zone top
        stop_price = zone.price_low - atr_value * 0.15  # below zone bottom
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            continue

        # Find nearest resistance above for TP
        tp_price = _find_opposing_target(entry_price, "long", resist_zones, atr_value, cfg.rr_target, risk)
        reward = abs(tp_price - entry_price)
        rr = reward / risk if risk > 0 else 0

        if rr < cfg.min_rr_ratio:
            continue

        score = _zone_signal_score(zone, distance_to_zone, arm_dist, atr_value, close_price, df, current_index, cfg)
        if score < 0.3:
            continue

        signal_id = stable_id("zone_signal", symbol, timeframe, zone.zone_id, timestamp, "long")
        signals.append(StrategySignal(
            signal_id=signal_id,
            line_id=zone.zone_id,
            symbol=symbol,
            timeframe=timeframe,
            source="zone",
            signal_type="ZONE_SUPPORT_LONG",
            direction="long",
            trigger_mode="pre_limit",
            timestamp=timestamp,
            trigger_bar_index=current_index,
            score=score,
            priority_rank=None,
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
            risk_reward=rr,
            confirming_touch_count=zone.touches,
            bars_since_last_confirming_touch=current_index - zone.last_touch_index,
            distance_to_line=abs(distance_to_zone),
            line_side="support",
            reason_code="zone_support_long",
            factor_components={
                "zone_strength": zone.strength,
                "zone_touches": float(zone.touches),
                "distance_to_zone": distance_to_zone,
                "risk_reward": rr,
            },
        ))

    for zone in resist_zones:
        # Price must be near or touching the zone from below
        distance_to_zone = zone.price_low - close_price
        if distance_to_zone < -arm_dist or distance_to_zone > arm_dist * 2:
            continue

        entry_price = zone.price_low - atr_value * 0.02
        stop_price = zone.price_high + atr_value * 0.15
        risk = abs(stop_price - entry_price)
        if risk <= 0:
            continue

        tp_price = _find_opposing_target(entry_price, "short", support_zones, atr_value, cfg.rr_target, risk)
        reward = abs(entry_price - tp_price)
        rr = reward / risk if risk > 0 else 0

        if rr < cfg.min_rr_ratio:
            continue

        score = _zone_signal_score(zone, distance_to_zone, arm_dist, atr_value, close_price, df, current_index, cfg)
        if score < 0.3:
            continue

        signal_id = stable_id("zone_signal", symbol, timeframe, zone.zone_id, timestamp, "short")
        signals.append(StrategySignal(
            signal_id=signal_id,
            line_id=zone.zone_id,
            symbol=symbol,
            timeframe=timeframe,
            source="zone",
            signal_type="ZONE_RESISTANCE_SHORT",
            direction="short",
            trigger_mode="pre_limit",
            timestamp=timestamp,
            trigger_bar_index=current_index,
            score=score,
            priority_rank=None,
            entry_price=entry_price,
            stop_price=stop_price,
            tp_price=tp_price,
            risk_reward=rr,
            confirming_touch_count=zone.touches,
            bars_since_last_confirming_touch=current_index - zone.last_touch_index,
            distance_to_line=abs(distance_to_zone),
            line_side="resistance",
            reason_code="zone_resistance_short",
            factor_components={
                "zone_strength": zone.strength,
                "zone_touches": float(zone.touches),
                "distance_to_zone": distance_to_zone,
                "risk_reward": rr,
            },
        ))

    # Sort by score descending
    signals.sort(key=lambda s: -s.score)
    return signals


def _find_opposing_target(
    entry: float, direction: str, opposing_zones: Sequence[HorizontalZone],
    atr_value: float, rr_target: float, risk: float,
) -> float:
    """Find the nearest opposing zone as take-profit target."""
    if direction == "long":
        candidates = [z.price_low for z in opposing_zones if z.price_low > entry]
        if candidates:
            return min(candidates)
        return entry + rr_target * risk  # fallback: use RR target
    else:
        candidates = [z.price_high for z in opposing_zones if z.price_high < entry]
        if candidates:
            return max(candidates)
        return entry - rr_target * risk


def _zone_signal_score(
    zone: HorizontalZone, distance: float, arm_dist: float,
    atr_value: float, close_price: float, df, bar_index: int, cfg: StrategyConfig,
) -> float:
    """Score a zone-based signal 0-1."""
    # Zone strength (already 0-100, normalize)
    strength_score = clamp(zone.strength / 100.0)

    # Proximity (closer = better)
    proximity_score = clamp(1.0 - abs(distance) / max(arm_dist * 2, 1e-10))

    # Touch count
    touch_score = clamp(zone.touches / 6.0)

    # Recency
    bars_since = bar_index - zone.last_touch_index
    recency_score = clamp(1.0 - bars_since / max(cfg.max_fresh_bars, 1))

    return (
        0.35 * strength_score
        + 0.25 * proximity_score
        + 0.20 * touch_score
        + 0.20 * recency_score
    )


__all__ = ["generate_zone_signals"]
