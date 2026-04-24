"""Canonical data schema for trendline objects.

Every adapter, tokeniser, and model in the package consumes and
produces these types. No other trendline representation is allowed
to cross module boundaries.
"""
from __future__ import annotations

from typing import Literal, Optional

try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, validator as field_validator  # type: ignore
    _PYDANTIC_V2 = False


LineRole = Literal[
    "support", "resistance",
    "channel_upper", "channel_lower",
    "wedge_side", "triangle_side",
    "unknown",
]
Direction = Literal["up", "down", "flat"]
LabelSource = Literal["manual", "auto", "auto_approved", "auto_rejected"]


class TrendlineRecord(BaseModel):
    """A single trendline / market-structure segment.

    See docs/trendline_tokenizer/ARCHITECTURE.md §B for field semantics.
    """
    # ── identity ─────────────────────────────────────────────
    id: str
    symbol: str
    exchange: str = "bitget"
    timeframe: str  # "1m" "3m" "5m" "15m" "30m" "1h" "2h" "4h" "6h" "12h" "1d" "1w"

    # ── geometry ─────────────────────────────────────────────
    start_time: int
    end_time: int
    start_bar_index: int
    end_bar_index: int
    start_price: float
    end_price: float
    extend_right: bool = False
    extend_left: bool = False

    # ── classification ───────────────────────────────────────
    line_role: LineRole
    direction: Direction

    # ── structural behaviour ─────────────────────────────────
    touch_count: int = Field(ge=0, default=0)
    rejection_strength_atr: Optional[float] = None
    bounce_after: Optional[bool] = None
    bounce_strength_atr: Optional[float] = None
    break_after: Optional[bool] = None
    break_distance_atr: Optional[float] = None
    retested_after_break: Optional[bool] = None

    # ── context (for tokeniser buckets) ──────────────────────
    volatility_atr_pct: Optional[float] = None
    volume_z_score: Optional[float] = None
    distance_to_ma20_atr: Optional[float] = None
    distance_to_recent_high_atr: Optional[float] = None
    distance_to_recent_low_atr: Optional[float] = None

    # ── provenance ───────────────────────────────────────────
    label_source: LabelSource
    auto_method: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    quality_flags: list[str] = Field(default_factory=list)
    created_at: int
    notes: str = ""

    # ── validators ───────────────────────────────────────────
    @field_validator("end_time")
    @classmethod
    def _end_after_start(cls, v, info):
        data = info.data if hasattr(info, "data") else info
        st = data.get("start_time") if isinstance(data, dict) else getattr(data, "start_time", None)
        if st is not None and v < st:
            raise ValueError(f"end_time {v} < start_time {st}")
        return v

    @field_validator("end_bar_index")
    @classmethod
    def _end_idx_after_start(cls, v, info):
        data = info.data if hasattr(info, "data") else info
        st = data.get("start_bar_index") if isinstance(data, dict) else getattr(data, "start_bar_index", None)
        if st is not None and v < st:
            raise ValueError(f"end_bar_index {v} < start_bar_index {st}")
        return v

    # ── helpers ──────────────────────────────────────────────
    def slope_per_bar(self) -> float:
        """Price change per bar (not log-space)."""
        dur = max(1, self.end_bar_index - self.start_bar_index)
        return (self.end_price - self.start_price) / dur

    def log_slope_per_bar(self) -> float:
        """Log-price change per bar (stable across price scales)."""
        import math
        dur = max(1, self.end_bar_index - self.start_bar_index)
        if self.start_price <= 0 or self.end_price <= 0:
            return 0.0
        return (math.log(self.end_price) - math.log(self.start_price)) / dur

    def duration_bars(self) -> int:
        return max(1, self.end_bar_index - self.start_bar_index)


class TokenizedTrendline(BaseModel):
    record_id: str
    coarse_token_id: int
    fine_token_id: int
    tokenizer_version: str
    reconstruction_error: Optional[float] = None


class BucketSpec(BaseModel):
    name: str
    kind: Literal["enum", "linear", "log", "quantile"]
    edges: Optional[list[float]] = None
    values: Optional[list[str]] = None


class TokenizerConfig(BaseModel):
    version: str
    coarse_dims: list[BucketSpec]
    fine_dims: list[BucketSpec]
    feature_vector_keys: list[str] = Field(default_factory=list)
    normalization: dict = Field(default_factory=dict)


class OHLCVBar(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
