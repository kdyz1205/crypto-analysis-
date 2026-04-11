"""Strategy 2: Short-Horizon Sweep Breakout Follower

After compression → depth swept → burst starts → follow for 2-5 ticks.
"""

from __future__ import annotations
from dataclasses import dataclass
from ..data_feed.features import MarketFeatures


@dataclass
class BreakoutSignal:
    side: str
    price: float
    size: float
    reason: str
    max_chase_ticks: int = 3
    timeout_ms: int = 3000


class SweepBreakout:
    """Follow micro-bursts after compression + depth sweep."""

    def __init__(
        self,
        compression_max: float = 3.0,
        momentum_min: float = 2.0,
        vacuum_min: float = 0.2,
        max_spread_bps: float = 5.0,
        target_ticks: int = 3,
        size_usdt: float = 6.0,
    ):
        self.compression_max = compression_max
        self.momentum_min = momentum_min
        self.vacuum_min = vacuum_min
        self.max_spread = max_spread_bps
        self.target = target_ticks
        self.size_usdt = size_usdt

    def evaluate(self, f: MarketFeatures) -> BreakoutSignal | None:
        if f.spread_bps > self.max_spread:
            return None
        if f.toxicity > 0.6:
            return None
        if f.mid <= 0:
            return None

        size = round(self.size_usdt / f.mid, 4)
        if size <= 0:
            return None

        # Upward burst: positive momentum + ask depth disappearing
        if (f.price_momentum_1 > self.momentum_min
                and f.depth_vacuum_ask > self.vacuum_min * f.ask_depth_5
                and f.burst_score > 0.5):
            price = round(f.mid + f.mid * 0.0002, 4)
            return BreakoutSignal(
                side="buy", price=price, size=size,
                reason=f"up_burst mom={f.price_momentum_1:.1f} vac={f.depth_vacuum_ask:.1f}",
            )

        # Downward burst
        if (f.price_momentum_1 < -self.momentum_min
                and f.depth_vacuum_bid > self.vacuum_min * f.bid_depth_5
                and f.burst_score > 0.5):
            price = round(f.mid - f.mid * 0.0002, 4)
            return BreakoutSignal(
                side="sell", price=price, size=size,
                reason=f"down_burst mom={f.price_momentum_1:.1f} vac={f.depth_vacuum_bid:.1f}",
            )

        return None
