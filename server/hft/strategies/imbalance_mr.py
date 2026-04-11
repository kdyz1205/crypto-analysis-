"""Strategy 1: Queue Imbalance Mean Reversion Scalp

When book is heavily imbalanced but momentum is fading → fade the imbalance.
Entry: marketable limit opposite to imbalance direction.
Exit: 1-3 ticks or 5s timeout.
"""

from __future__ import annotations
from dataclasses import dataclass
from ..data_feed.features import MarketFeatures


@dataclass
class ImbalanceSignal:
    side: str  # "buy" or "sell"
    price: float
    size: float
    reason: str
    edge_ticks: float
    timeout_ms: int = 5000


class ImbalanceMeanReversion:
    """Fade extreme order book imbalance when momentum is exhausted."""

    def __init__(
        self,
        imbalance_threshold: float = 0.55,
        micro_gap_threshold: float = 0.3,
        max_spread_bps: float = 5.0,
        target_ticks: int = 2,
        stop_ticks: int = 2,
        size_usdt: float = 6.0,
    ):
        self.imb_thresh = imbalance_threshold
        self.gap_thresh = micro_gap_threshold
        self.max_spread = max_spread_bps
        self.target = target_ticks
        self.stop = stop_ticks
        self.size_usdt = size_usdt

    def evaluate(self, f: MarketFeatures) -> ImbalanceSignal | None:
        """Check if conditions met. Returns signal or None."""
        if f.spread_bps > self.max_spread:
            return None
        if f.toxicity > 0.6:
            return None
        if f.mid <= 0:
            return None

        size = round(self.size_usdt / f.mid, 4)
        if size <= 0:
            return None

        # Bid-heavy imbalance + momentum fading → sell (fade the bid pressure)
        if f.imbalance_3 > self.imb_thresh and f.micro_gap_ticks > self.gap_thresh:
            # Momentum should be slowing (not accelerating upward)
            if f.price_momentum_1 < 1.5:  # not still surging
                price = round(f.mid + f.mid * 0.0001, 4)  # slightly above mid
                return ImbalanceSignal(
                    side="sell", price=price, size=size,
                    reason=f"bid_heavy imb={f.imbalance_3:.2f} gap={f.micro_gap_ticks:.1f}",
                    edge_ticks=self.target,
                )

        # Ask-heavy imbalance + momentum fading → buy
        if f.imbalance_3 < -self.imb_thresh and f.micro_gap_ticks < -self.gap_thresh:
            if f.price_momentum_1 > -1.5:
                price = round(f.mid - f.mid * 0.0001, 4)
                return ImbalanceSignal(
                    side="buy", price=price, size=size,
                    reason=f"ask_heavy imb={f.imbalance_3:.2f} gap={f.micro_gap_ticks:.1f}",
                    edge_ticks=self.target,
                )

        return None
