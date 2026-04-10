from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class HistoryCoverageModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    exchange: str | None = None
    dataSourceMode: str | None = None
    dataSourceKind: str | None = None
    requestedDays: int | None = None
    baseInterval: str | None = None
    resampledFromInterval: str | None = None
    sourceBarCount: int | None = None
    historyMode: str | None = None
    loadedBarCount: int | None = None
    earliestLoadedTimestamp: Any | None = None
    latestLoadedTimestamp: Any | None = None
    listingStartTimestamp: Any | None = None
    isFullHistory: bool | None = None
    isTruncated: bool | None = None
    truncationReason: str | None = None
    analysisInputBarCount: int | None = None
    analysisEarliestTimestamp: Any | None = None
    analysisLatestTimestamp: Any | None = None
    analysisWasTrimmed: bool | None = None


__all__ = ["HistoryCoverageModel"]
