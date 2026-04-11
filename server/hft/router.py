"""Strategy 5: Microstructure Regime Router

Decides which strategy gets to trade based on current market features.
Routes to: ImbalanceMR / SweepBreakout / InventoryMM / NoTrade
"""

from __future__ import annotations
from dataclasses import dataclass
from .data_feed.features import MarketFeatures
from .strategies.imbalance_mr import ImbalanceMeanReversion, ImbalanceSignal
from .strategies.sweep_breakout import SweepBreakout, BreakoutSignal
from .strategies.inventory_mm import InventoryMarketMaker, MMQuote


@dataclass
class RoutedDecision:
    strategy: str  # "imbalance_mr" | "sweep_breakout" | "inventory_mm" | "no_trade"
    regime: str
    signal: ImbalanceSignal | BreakoutSignal | MMQuote | None
    features: MarketFeatures
    reason: str


class RegimeRouter:
    """Routes market state to the best-fit strategy."""

    def __init__(self):
        self.s1 = ImbalanceMeanReversion()
        self.s2 = SweepBreakout()
        self.s3 = InventoryMarketMaker()

        # Performance tracking
        self.stats = {
            "imbalance_mr": {"trades": 0, "pnl": 0.0},
            "sweep_breakout": {"trades": 0, "pnl": 0.0},
            "inventory_mm": {"trades": 0, "pnl": 0.0},
        }
        self.consecutive_losses = 0
        self.total_trades = 0

    def route(self, features: MarketFeatures, inventory: float = 0.0) -> RoutedDecision:
        """Determine which strategy to run and generate signal."""
        regime = features.regime

        # Kill switch
        if self.consecutive_losses >= 8:
            return RoutedDecision("no_trade", "killed", None, features, f"consecutive_losses={self.consecutive_losses}")

        # Toxic → no trade
        if regime == "toxic":
            return RoutedDecision("no_trade", regime, None, features, f"toxic={features.toxicity:.2f}")

        # Mean revert regime → try imbalance scalp
        if regime == "mean_revert":
            sig = self.s1.evaluate(features)
            if sig:
                return RoutedDecision("imbalance_mr", regime, sig, features, sig.reason)

        # Burst regime → try sweep breakout
        if regime == "burst":
            sig = self.s2.evaluate(features)
            if sig:
                return RoutedDecision("sweep_breakout", regime, sig, features, sig.reason)

        # Stable spread → market make
        if regime == "stable_spread":
            quote = self.s3.evaluate(features, inventory)
            if quote:
                return RoutedDecision("inventory_mm", regime, quote, features, quote.reason)

        # Fallback: if stable_spread but MM didn't fire, try imbalance
        if regime == "stable_spread":
            sig = self.s1.evaluate(features)
            if sig:
                return RoutedDecision("imbalance_mr", regime, sig, features, sig.reason)

        return RoutedDecision("no_trade", regime, None, features, "no_signal")

    def record_trade(self, strategy: str, pnl: float):
        """Record trade result for performance tracking."""
        self.total_trades += 1
        if strategy in self.stats:
            self.stats[strategy]["trades"] += 1
            self.stats[strategy]["pnl"] += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
