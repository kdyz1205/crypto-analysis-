"""Multi-timeframe confluence scoring.

Compares zones/lines from different timeframes to detect overlapping price levels.
When a support/resistance level exists across multiple timeframes, it's stronger.
"""

from __future__ import annotations

from typing import Sequence

from .config import clamp
from .types import Trendline
from .zones import HorizontalZone


def confluence_score_for_line(
    line: Trendline,
    other_tf_zones: Sequence[HorizontalZone],
    other_tf_lines: Sequence[Trendline],
    *,
    tolerance_pct: float = 0.005,
) -> float:
    """Score 0-1 for how well a trendline's current price aligns with other-TF structures.

    Args:
        line: The trendline being scored (from the primary timeframe).
        other_tf_zones: Horizontal zones from OTHER timeframes.
        other_tf_lines: Trendlines from OTHER timeframes.
        tolerance_pct: Price must be within this % to count as confluent.

    Returns:
        0.0 = no confluence, 1.0 = strong multi-TF alignment.
    """
    price = line.projected_price_current
    if price <= 0:
        return 0.0

    tolerance = price * tolerance_pct
    matches = 0
    total_sources = 0

    # Check against other-TF horizontal zones
    for zone in other_tf_zones:
        if zone.side != line.side:
            continue
        total_sources += 1
        if zone.price_low - tolerance <= price <= zone.price_high + tolerance:
            matches += 1

    # Check against other-TF trendlines
    for other_line in other_tf_lines:
        if other_line.side != line.side:
            continue
        if other_line.timeframe == line.timeframe:
            continue  # skip same TF
        total_sources += 1
        other_price = other_line.projected_price_current
        if abs(other_price - price) <= tolerance:
            matches += 1

    if total_sources == 0:
        return 0.0

    # Each match is a strong signal; even 1 match is significant
    # 1 match = 0.5, 2 matches = 0.8, 3+ = 1.0
    if matches == 0:
        return 0.0
    if matches == 1:
        return 0.5
    if matches == 2:
        return 0.8
    return 1.0


def confluence_score_for_zone(
    zone: HorizontalZone,
    other_tf_zones: Sequence[HorizontalZone],
    other_tf_lines: Sequence[Trendline],
    *,
    tolerance_pct: float = 0.005,
) -> float:
    """Score 0-1 for how well a horizontal zone aligns with other-TF structures."""
    price = zone.price_center
    if price <= 0:
        return 0.0

    tolerance = price * tolerance_pct
    matches = 0
    total_sources = 0

    for other_zone in other_tf_zones:
        if other_zone.side != zone.side:
            continue
        if other_zone.zone_id == zone.zone_id:
            continue
        total_sources += 1
        # Check if zones overlap (with tolerance)
        if other_zone.price_low - tolerance <= zone.price_high and other_zone.price_high + tolerance >= zone.price_low:
            matches += 1

    for line in other_tf_lines:
        if line.side != zone.side:
            continue
        total_sources += 1
        line_price = line.projected_price_current
        if zone.price_low - tolerance <= line_price <= zone.price_high + tolerance:
            matches += 1

    if total_sources == 0:
        return 0.0

    if matches == 0:
        return 0.0
    if matches == 1:
        return 0.5
    if matches == 2:
        return 0.8
    return 1.0


__all__ = [
    "confluence_score_for_line",
    "confluence_score_for_zone",
]
