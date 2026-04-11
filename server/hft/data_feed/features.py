"""Market microstructure features — computed from order book + trade stream.

All features from the 5-strategy spec:
- I_k (top-k imbalance)
- MicroPrice / MicroGap
- TradeImbalance
- CancelPressure
- Compression
- SweepRatio
- Vacuum
- FlowBurst
- RealizedVol
- Toxicity
"""

from __future__ import annotations
from dataclasses import dataclass
from .book_builder import BookBuilder, OrderBook
import math


@dataclass
class MarketFeatures:
    """Complete feature vector for regime router and strategy decisions."""

    # Book features
    mid: float = 0.0
    spread_bps: float = 0.0
    imbalance_3: float = 0.0       # I_3
    imbalance_5: float = 0.0       # I_5
    microprice: float = 0.0
    micro_gap_ticks: float = 0.0   # (microprice - mid) / tick_size
    bid_depth_5: float = 0.0
    ask_depth_5: float = 0.0

    # Flow features
    cancel_pressure: float = 0.0
    depth_vacuum_ask: float = 0.0  # how much ask depth disappeared
    depth_vacuum_bid: float = 0.0

    # Volatility
    realized_vol_5s: float = 0.0   # realized vol over last 5 updates
    compression: float = 0.0       # range / vol ratio

    # Trade flow (estimated from price moves between snapshots)
    price_momentum_1: float = 0.0  # 1-update price change
    price_momentum_5: float = 0.0  # 5-update price change

    # Derived scores
    mr_score: float = 0.0          # mean reversion score
    burst_score: float = 0.0       # breakout score
    mm_score: float = 0.0          # market making score
    toxicity: float = 0.0          # toxic flow estimate

    @property
    def regime(self) -> str:
        """Simple regime classification."""
        if self.toxicity > 0.7 or self.spread_bps > 10:
            return "toxic"
        if self.burst_score > self.mr_score and self.burst_score > self.mm_score:
            return "burst"
        if self.mr_score > self.mm_score:
            return "mean_revert"
        return "stable_spread"


def compute_features(builder: BookBuilder, tick_size: float = 0.001) -> MarketFeatures:
    """Compute all features from current book state and history."""
    book = builder.current
    f = MarketFeatures()

    if not book.bids or not book.asks:
        return f

    f.mid = book.mid
    f.spread_bps = book.spread_bps
    f.imbalance_3 = book.imbalance(3)
    f.imbalance_5 = book.imbalance(5)
    f.microprice = book.microprice
    f.micro_gap_ticks = (f.microprice - f.mid) / tick_size if tick_size > 0 else 0
    f.bid_depth_5 = book.bid_depth(5)
    f.ask_depth_5 = book.ask_depth(5)

    # Cancel pressure
    f.cancel_pressure = builder.cancel_pressure(3)

    # Depth vacuum (how much opposing depth disappeared)
    f.depth_vacuum_ask = -builder.depth_change("ask", 3) if builder.depth_change("ask", 3) < 0 else 0
    f.depth_vacuum_bid = -builder.depth_change("bid", 3) if builder.depth_change("bid", 3) < 0 else 0

    # Price history features
    hist = builder.history
    if len(hist) >= 2:
        f.price_momentum_1 = (hist[-1].mid - hist[-2].mid) / tick_size if hist[-2].mid > 0 else 0

    if len(hist) >= 6:
        f.price_momentum_5 = (hist[-1].mid - hist[-6].mid) / tick_size if hist[-6].mid > 0 else 0

        # Realized vol (std of mid price changes over last 5 updates)
        changes = [(hist[i].mid - hist[i-1].mid) for i in range(-5, 0) if hist[i-1].mid > 0]
        if changes:
            mean_c = sum(changes) / len(changes)
            var_c = sum((c - mean_c)**2 for c in changes) / len(changes)
            f.realized_vol_5s = math.sqrt(var_c) if var_c > 0 else 0

        # Compression: range / vol
        mids = [h.mid for h in hist[-10:] if h.mid > 0]
        if mids and f.realized_vol_5s > 0:
            price_range = max(mids) - min(mids)
            f.compression = price_range / f.realized_vol_5s if f.realized_vol_5s > 0 else 0

    # ── Scoring ──────────────────────────────────────────────────────

    # Mean Reversion Score: high imbalance + low momentum + narrow spread
    f.mr_score = (
        0.4 * min(abs(f.imbalance_3), 1.0)
        + 0.3 * max(0, 1.0 - abs(f.price_momentum_1) / 3)  # low momentum = high MR
        + 0.3 * max(0, 1.0 - f.spread_bps / 5)               # narrow spread
    )

    # Burst Score: price moving + depth disappearing + high momentum
    f.burst_score = (
        0.3 * min(abs(f.price_momentum_1) / 3, 1.0)
        + 0.3 * min((f.depth_vacuum_ask + f.depth_vacuum_bid) / max(f.bid_depth_5 + f.ask_depth_5, 1), 1.0)
        + 0.2 * min(abs(f.price_momentum_5) / 5, 1.0)
        + 0.2 * max(0, 1.0 - f.compression / 10) if f.compression > 0 else 0
    )

    # Market Making Score: wide enough spread + low vol + low toxicity
    f.mm_score = (
        0.4 * min(f.spread_bps / 3, 1.0)
        + 0.3 * max(0, 1.0 - f.realized_vol_5s / (tick_size * 3)) if tick_size > 0 else 0
        + 0.3 * max(0, 1.0 - abs(f.imbalance_5))
    )

    # Toxicity: large one-sided moves + spread blowout
    f.toxicity = (
        0.4 * min(abs(f.price_momentum_5) / 10, 1.0)
        + 0.3 * min(f.spread_bps / 15, 1.0)
        + 0.3 * min(abs(f.imbalance_5), 1.0) * min(abs(f.price_momentum_1) / 2, 1.0)
    )

    return f
