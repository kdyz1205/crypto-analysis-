from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ManualTrendlineModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    manual_line_id: str
    symbol: str
    timeframe: str
    side: Literal["resistance", "support"]
    source: str = "manual"
    t_start: int
    t_end: int
    price_start: float
    price_end: float
    extend_left: bool = False
    extend_right: bool = False
    locked: bool = False
    label: str = ""
    notes: str = ""
    comparison_status: str = "uncompared"
    override_mode: str = "display_only"
    nearest_auto_line_id: str | None = None
    slope_diff: float | None = None
    projected_price_diff: float | None = None
    overlap_ratio: float | None = None
    created_at: int
    updated_at: int
    line_width: float = 1.8


class ManualTrendlineCreateRequest(BaseModel):
    symbol: str
    timeframe: str
    side: Literal["resistance", "support"]
    t_start: int
    t_end: int
    price_start: float
    price_end: float
    extend_left: bool = False
    extend_right: bool = False
    locked: bool = False
    label: str = ""
    notes: str = ""
    override_mode: str = "display_only"
    line_width: float = Field(default=1.8, ge=0.5, le=8.0)


class ManualTrendlineUpdateRequest(BaseModel):
    t_start: int | None = None
    t_end: int | None = None
    price_start: float | None = None
    price_end: float | None = None
    extend_left: bool | None = None
    extend_right: bool | None = None
    locked: bool | None = None
    label: str | None = None
    notes: str | None = None
    override_mode: str | None = None
    line_width: float | None = Field(default=None, ge=0.5, le=8.0)


class ManualTrendlineListResponse(BaseModel):
    drawings: list[ManualTrendlineModel] = Field(default_factory=list)


class ManualTrendlineResponse(BaseModel):
    drawing: ManualTrendlineModel


class ManualTrendlineClearResponse(BaseModel):
    removed: int


__all__ = [
    "ManualTrendlineClearResponse",
    "ManualTrendlineCreateRequest",
    "ManualTrendlineListResponse",
    "ManualTrendlineModel",
    "ManualTrendlineResponse",
    "ManualTrendlineUpdateRequest",
]
