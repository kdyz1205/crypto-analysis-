from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


OverrideMode = Literal[
    "display_only",
    "compare_only",
    "promote_to_active",
    "suppress_nearest_auto_line",
    "strategy_input_enabled",
]


@dataclass(frozen=True, slots=True)
class ManualTrendline:
    manual_line_id: str
    symbol: str
    timeframe: str
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
    line_width: float = 1.8

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["ManualTrendline", "OverrideMode"]
