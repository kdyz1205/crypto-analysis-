"""TrendlineVariant interface. Every evolved algorithm must implement this.

The orchestrator imports variants by path, calls detect_lines, and feeds the
returned lines into the fade+flip backtest harness. Variants should be pure
functions: same candles in → same lines out.

Sanity gate (enforced before backtest): variants that violate any of these
rules are discarded without evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class EvolvedChannel:
    """A pair of near-parallel lines forming a price channel.

    Murphy Ch.5: "Once a trendline is in place, a parallel line can be
    drawn across the reaction highs to form a channel." Channels are
    structurally meaningful; pairs of lines should be rendered together,
    not as two independent lines.
    """
    upper: "EvolvedLine"
    lower: "EvolvedLine"
    width_atr: float       # average vertical distance between lines, in ATR
    parallelism: float     # 1.0 = perfectly parallel, 0 = orthogonal
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class EvolvedLine:
    """Minimal line shape consumed by the backtest harness.

    A line is NOT a single trade — it's a SEQUENCE of setups. Every touch
    at index k (k >= 2) in `touch_indices` is a distinct setup: the state
    of the line AT that touch, awaiting the next touch. The harness
    replays each touch and backtests independently.

    This matches how a trader actually uses a trendline: trade it at touch 2
    expecting touch 3, then trade it AGAIN at touch 3 expecting touch 4, etc.
    The edge per touch-number is itself something for evolution to discover.
    """
    side: str                           # "support" | "resistance"
    start_index: int                    # bar index of left anchor
    end_index: int                      # bar index of right anchor (or last touch)
    start_price: float
    end_price: float
    touch_count: int                    # total confirming touches
    touch_indices: tuple[int, ...] = () # bar indices of ALL confirming touches (ordered)
    score: float = 0.0                  # variant-internal score, unused by harness

    def price_at(self, bar_index: int) -> float:
        """Linear projection at an arbitrary bar index (using the regression line)."""
        span = self.end_index - self.start_index
        if span == 0:
            return self.start_price
        slope = (self.end_price - self.start_price) / span
        return self.start_price + slope * (bar_index - self.start_index)


class TrendlineVariant(Protocol):
    """Interface every variant must implement.

    Variants live in server/strategy/evolved/v{round}_{tag}.py and export
    a module-level `detect_lines` function matching this signature.
    """
    @staticmethod
    def detect_lines(
        candles: pd.DataFrame,
        timeframe: str,
        symbol: str,
    ) -> list[EvolvedLine]:
        ...


def sanity_check(lines: Sequence[EvolvedLine], n_bars: int) -> tuple[bool, str]:
    """Gate applied to every variant's output before backtest.

    Returns (ok, reason). Variants that fail are discarded.
    """
    if lines is None:
        return False, "returned None"
    n = len(lines)
    if n == 0:
        return True, "no lines (permitted but low fitness)"
    if n > 40:
        return False, f"too many lines ({n}) — probably noise"
    for i, l in enumerate(lines):
        if l.side not in ("support", "resistance"):
            return False, f"line {i}: invalid side {l.side!r}"
        if not (0 <= l.start_index < n_bars):
            return False, f"line {i}: start_index {l.start_index} out of range [0,{n_bars})"
        if not (0 <= l.end_index < n_bars):
            return False, f"line {i}: end_index {l.end_index} out of range"
        if l.end_index <= l.start_index:
            return False, f"line {i}: end_index <= start_index"
        if not (l.start_price == l.start_price):  # NaN check
            return False, f"line {i}: start_price is NaN"
        if not (l.end_price == l.end_price):
            return False, f"line {i}: end_price is NaN"
        if l.start_price <= 0 or l.end_price <= 0:
            return False, f"line {i}: non-positive price"
        if l.touch_count < 2:
            return False, f"line {i}: touch_count {l.touch_count} < 2"
    return True, "ok"
