from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal


OverrideMode = Literal[
    "display_only",
    "compare_only",
    "promote_to_active",
    "suppress_nearest_auto_line",
    "strategy_input_enabled",
]

EffectiveRole = Literal["support", "resistance", "on_line"]


@dataclass(frozen=True, slots=True)
class ManualTrendline:
    manual_line_id: str
    symbol: str
    timeframe: str
    # `side` = the user's original LABEL at draw-time (the trade INTENT
    # — was it drawn as a ceiling or as a floor). It does NOT update
    # when price subsequently breaks through. For "what role does this
    # line currently play vs price" use `compute_effective_role()`.
    side: Literal["resistance", "support"]
    source: str
    t_start: int
    t_end: int
    price_start: float
    price_end: float
    extend_left: bool
    extend_right: bool
    locked: bool
    label: str
    notes: str
    comparison_status: str
    override_mode: OverrideMode
    nearest_auto_line_id: str | None
    slope_diff: float | None
    projected_price_diff: float | None
    overlap_ratio: float | None
    created_at: int
    updated_at: int
    line_width: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)

    # ──────────────────────────────────────────────────────────────
    # Effective-role helpers — see TA_BASICS.md §1 (role reversal)
    # and §2 ("the line's DB `side` may be stale because the line was
    # LABELED when it was drawn, and price has since moved").
    #
    # Use line_at_ts() to project the line price at any timestamp.
    # Use compute_effective_role(price_now, ts_now) to ask:
    # "given current market price, is this line acting as floor or ceiling?"
    # ──────────────────────────────────────────────────────────────

    def line_at_ts(self, ts: int) -> float:
        """LOG interpolation between the two anchors. Matches the
        production projection in conditionals.watcher._project_line_geometry
        so role inference uses the SAME number the order-replan uses."""
        span = self.t_end - self.t_start
        if span <= 0:
            return float(self.price_start)
        ps = float(self.price_start)
        pe = float(self.price_end)
        if ts <= self.t_start:
            return ps if not self.extend_left else self._extrapolate_left(ts)
        if ts >= self.t_end:
            return pe if not self.extend_right else self._extrapolate_right(ts)
        if ps <= 0 or pe <= 0:
            slope = (pe - ps) / span
            return ps + slope * (ts - self.t_start)
        ratio = (ts - self.t_start) / span
        return math.exp(math.log(ps) + ratio * (math.log(pe) - math.log(ps)))

    def _extrapolate_left(self, ts: int) -> float:
        # Same log slope, project backwards
        span = self.t_end - self.t_start
        if span <= 0 or self.price_start <= 0 or self.price_end <= 0:
            slope = (self.price_end - self.price_start) / max(span, 1)
            return self.price_start + slope * (ts - self.t_start)
        ratio = (ts - self.t_start) / span
        return math.exp(math.log(self.price_start) + ratio * (math.log(self.price_end) - math.log(self.price_start)))

    def _extrapolate_right(self, ts: int) -> float:
        return self._extrapolate_left(ts)   # same formula, ratio just exceeds 1

    def compute_effective_role(self, current_price: float, now_ts: int,
                               *, tolerance_pct: float = 0.05) -> EffectiveRole:
        """Per TA_BASICS.md §2: line BELOW price → support; ABOVE → resistance.
        `tolerance_pct` (0.05% default) treats price within 0.05% of the
        line as 'on_line' so we don't flip-flop on noise."""
        line_now = self.line_at_ts(now_ts)
        if line_now <= 0:
            return self.side  # type: ignore[return-value]
        diff_pct = abs(current_price - line_now) / line_now
        if diff_pct < tolerance_pct / 100.0:
            return "on_line"
        return "support" if line_now < current_price else "resistance"


__all__ = ["ManualTrendline", "OverrideMode", "EffectiveRole"]
