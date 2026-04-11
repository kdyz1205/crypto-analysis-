"""Strategy 3: Inventory-Skewed Market Making

Quote both sides, skew by inventory, widen by volatility/toxicity.
Fair = Mid + α·MicroGap + β·TradeImbalance
ReservationPrice = Fair - γ·Inventory
HalfSpread = max(MinSpread, c1·σ + c2·Tox + c3·InvRisk)
"""

from __future__ import annotations
from dataclasses import dataclass
from ..data_feed.features import MarketFeatures


@dataclass
class MMQuote:
    bid_price: float
    ask_price: float
    size: float
    skew: float  # positive = skewed to sell
    reason: str


class InventoryMarketMaker:
    """Two-sided quoting with dynamic spread and inventory skew."""

    def __init__(
        self,
        min_spread_bps: float = 1.0,
        alpha: float = 0.3,        # microprice weight
        beta: float = 0.1,         # trade imbalance weight
        gamma: float = 0.0005,     # inventory penalty per unit
        c_vol: float = 2.0,        # vol → spread multiplier
        c_tox: float = 3.0,        # toxicity → spread multiplier
        c_inv: float = 1.5,        # inventory risk → spread multiplier
        max_inventory_usdt: float = 15.0,
        size_usdt: float = 6.0,
    ):
        self.min_spread_bps = min_spread_bps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c_vol = c_vol
        self.c_tox = c_tox
        self.c_inv = c_inv
        self.max_inv = max_inventory_usdt
        self.size_usdt = size_usdt

    def evaluate(self, f: MarketFeatures, inventory: float = 0.0) -> MMQuote | None:
        """Generate bid/ask quotes. inventory = net position in base units."""
        if f.mid <= 0 or f.spread_bps > 15:
            return None
        if f.toxicity > 0.7:
            return None

        size = round(self.size_usdt / f.mid, 4)
        if size <= 0:
            return None

        # Fair value
        fair = f.mid + self.alpha * (f.microprice - f.mid) + self.beta * f.imbalance_3 * f.mid * 0.0001

        # Reservation price (penalize inventory)
        inv_value = abs(inventory * f.mid)
        reservation = fair - self.gamma * inventory * f.mid

        # Dynamic half spread
        vol_component = self.c_vol * f.realized_vol_5s / f.mid * 10000 if f.mid > 0 else 0
        tox_component = self.c_tox * f.toxicity
        inv_component = self.c_inv * (inv_value / max(self.max_inv, 1))
        half_spread_bps = max(self.min_spread_bps, vol_component + tox_component + inv_component)
        half_spread = f.mid * half_spread_bps / 10000

        bid = round(reservation - half_spread, 4)
        ask = round(reservation + half_spread, 4)

        # Skew info
        skew = (reservation - f.mid) / f.mid * 10000 if f.mid > 0 else 0

        return MMQuote(
            bid_price=bid,
            ask_price=ask,
            size=size,
            skew=round(skew, 2),
            reason=f"fair={fair:.4f} res={reservation:.4f} hs={half_spread_bps:.1f}bps inv={inventory:.4f}",
        )
