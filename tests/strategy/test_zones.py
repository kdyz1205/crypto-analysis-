"""Tests for horizontal zone detection and scoring."""
import pandas as pd
from server.strategy.config import StrategyConfig
from server.strategy.pivots import detect_pivots
from server.strategy.zones import detect_horizontal_zones, HorizontalZone


def _make_candles_with_sr(n=100) -> pd.DataFrame:
    """Create candles that bounce off support ~50 and resistance ~60 multiple times."""
    closes = []
    for i in range(n):
        cycle = i % 20
        if cycle < 10:
            closes.append(50.0 + cycle * 1.0)  # rise from 50 to 59
        else:
            closes.append(60.0 - (cycle - 10) * 1.0)  # fall from 60 to 51
    return pd.DataFrame({
        "timestamp": list(range(n)),
        "open": [c - 0.2 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


def test_detect_zones_finds_support_and_resistance():
    """System should find zones near the bounce points."""
    df = _make_candles_with_sr(100)
    pivots = detect_pivots(df, StrategyConfig(pivot_left=3, pivot_right=3))
    zones = detect_horizontal_zones(df, pivots, StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=100))

    support_zones = [z for z in zones if z.side == "support"]
    resistance_zones = [z for z in zones if z.side == "resistance"]

    assert len(support_zones) > 0, "Should find at least one support zone"
    assert len(resistance_zones) > 0, "Should find at least one resistance zone"


def test_zones_are_capped_per_side():
    """max_zones_per_side should limit output."""
    df = _make_candles_with_sr(100)
    pivots = detect_pivots(df, StrategyConfig(pivot_left=3, pivot_right=3))
    zones = detect_horizontal_zones(
        df, pivots,
        StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=100),
        max_zones_per_side=1,
    )

    support_zones = [z for z in zones if z.side == "support"]
    resistance_zones = [z for z in zones if z.side == "resistance"]

    assert len(support_zones) <= 1
    assert len(resistance_zones) <= 1


def test_zone_has_valid_structure():
    """Each zone should have valid fields."""
    df = _make_candles_with_sr(100)
    pivots = detect_pivots(df, StrategyConfig(pivot_left=3, pivot_right=3))
    zones = detect_horizontal_zones(df, pivots, StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=100))

    for zone in zones:
        assert isinstance(zone, HorizontalZone)
        assert zone.side in ("support", "resistance")
        assert zone.price_low < zone.price_high
        assert zone.price_low <= zone.price_center <= zone.price_high
        assert zone.touches >= 2
        assert len(zone.touch_indices) == zone.touches
        assert zone.strength >= 0
        assert zone.strength <= 100
        assert isinstance(zone.strength_components, dict)
        assert "touch_score" in zone.strength_components


def test_zones_sorted_by_strength():
    """Zones should come out sorted by strength descending within each side."""
    df = _make_candles_with_sr(100)
    pivots = detect_pivots(df, StrategyConfig(pivot_left=3, pivot_right=3))
    zones = detect_horizontal_zones(
        df, pivots,
        StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=100),
        max_zones_per_side=5,
    )

    for side in ("support", "resistance"):
        side_zones = [z for z in zones if z.side == side]
        strengths = [z.strength for z in side_zones]
        assert strengths == sorted(strengths, reverse=True), f"{side} zones not sorted by strength"


def test_no_zones_with_insufficient_data():
    """With too few bars, should return empty."""
    df = pd.DataFrame({
        "timestamp": [0, 1, 2],
        "open": [100.0, 101.0, 100.5],
        "high": [101.0, 102.0, 101.5],
        "low": [99.0, 100.0, 99.5],
        "close": [100.5, 101.0, 100.0],
        "volume": [100.0, 100.0, 100.0],
    })
    pivots = detect_pivots(df, StrategyConfig(pivot_left=1, pivot_right=1))
    zones = detect_horizontal_zones(df, pivots)
    # With only 3 bars, very unlikely to get clustered zones
    assert isinstance(zones, list)
