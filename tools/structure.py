"""Tool 2: Structure Scanner — detect S/R zones, pivots, regime."""

from __future__ import annotations
import pandas as pd
from dataclasses import asdict


def scan_structure(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> dict:
    """Scan market structure. Returns zones, regime, pivots."""
    from server.strategy.config import StrategyConfig
    from server.strategy.pivots import detect_pivots
    from server.strategy.zones import detect_horizontal_zones
    from server.strategy.regime import detect_regime

    cfg = StrategyConfig()
    pivots = tuple(detect_pivots(df, cfg))
    zones = detect_horizontal_zones(df, pivots, cfg, symbol=symbol, timeframe=timeframe)
    regime = detect_regime(df, cfg)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "pivot_count": len(pivots),
        "zones": [asdict(z) for z in zones],
        "support_zones": [asdict(z) for z in zones if z.side == "support"],
        "resistance_zones": [asdict(z) for z in zones if z.side == "resistance"],
        "regime": asdict(regime),
        "current_price": float(df.iloc[-1]["close"]) if len(df) > 0 else 0,
    }
