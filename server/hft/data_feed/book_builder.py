"""Order book builder — maintains L2 book state from snapshots.

Bitget REST gives us snapshots every poll. We track changes between polls
to estimate cancel pressure, depth vacuum, and churn.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import time


@dataclass(slots=True)
class BookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    bids: list[BookLevel] = field(default_factory=list)  # best bid first
    asks: list[BookLevel] = field(default_factory=list)  # best ask first
    ts: float = 0.0  # local timestamp ms

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.spread / m * 10000) if m > 0 else 0.0

    @property
    def microprice(self) -> float:
        """Size-weighted mid price."""
        if not self.bids or not self.asks:
            return self.mid
        b, a = self.bids[0], self.asks[0]
        total = b.size + a.size
        if total <= 0:
            return self.mid
        return (a.price * b.size + b.price * a.size) / total

    def imbalance(self, k: int = 5) -> float:
        """Top-k level imbalance. Positive = bid heavy."""
        bid_vol = sum(b.size for b in self.bids[:k])
        ask_vol = sum(a.size for a in self.asks[:k])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def bid_depth(self, k: int = 5) -> float:
        return sum(b.size for b in self.bids[:k])

    def ask_depth(self, k: int = 5) -> float:
        return sum(a.size for a in self.asks[:k])


class BookBuilder:
    """Maintains order book state and tracks changes between updates."""

    def __init__(self):
        self.current: OrderBook = OrderBook()
        self.previous: OrderBook = OrderBook()
        self.history: list[OrderBook] = []
        self.max_history = 120  # ~2 min at 1s polling

    def update(self, raw_data: dict) -> OrderBook:
        """Update book from Bitget merge-depth response."""
        self.previous = self.current

        bids = [BookLevel(float(b[0]), float(b[1])) for b in (raw_data.get("bids") or [])]
        asks = [BookLevel(float(a[0]), float(a[1])) for a in (raw_data.get("asks") or [])]

        self.current = OrderBook(
            bids=bids,
            asks=asks,
            ts=time.time() * 1000,
        )

        self.history.append(self.current)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return self.current

    def depth_change(self, side: str, k: int = 3) -> float:
        """How much depth changed between current and previous snapshot."""
        if side == "bid":
            prev = sum(b.size for b in self.previous.bids[:k])
            curr = sum(b.size for b in self.current.bids[:k])
        else:
            prev = sum(a.size for a in self.previous.asks[:k])
            curr = sum(a.size for a in self.current.asks[:k])
        return curr - prev if prev > 0 else 0.0

    def cancel_pressure(self, k: int = 3) -> float:
        """Asymmetric cancel pressure. Positive = more ask cancels than bid."""
        bid_change = self.depth_change("bid", k)
        ask_change = self.depth_change("ask", k)
        total = self.current.bid_depth(k) + self.current.ask_depth(k)
        if total <= 0:
            return 0.0
        return (ask_change - bid_change) / total
