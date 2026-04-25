"""Feedback schemas. Three event types flow through the same JSONL log,
distinguished by `event_type`.
"""
from __future__ import annotations
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..schemas.trendline import TrendlineRecord


FeedbackEvent = Literal["corrected_trendline", "signal_accepted", "signal_rejected"]


class CorrectedTrendline(BaseModel):
    event_type: Literal["corrected_trendline"] = "corrected_trendline"
    created_at: int
    user: Optional[str] = None
    # original auto record (None if user drew from scratch)
    original_id: Optional[str] = None
    # the corrected line, in canonical form
    corrected: TrendlineRecord
    # free-form reason / category
    reason_code: Optional[str] = None
    notes: str = ""


class SignalAccepted(BaseModel):
    event_type: Literal["signal_accepted"] = "signal_accepted"
    created_at: int
    user: Optional[str] = None
    signal_id: str           # id of the SignalRecord from the paper log
    artifact_name: str       # which model produced the signal
    tokenizer_version: str
    symbol: str
    timeframe: str
    action: str              # BOUNCE | BREAK | WAIT (echo)
    realized_outcome: Optional[str] = None  # filled later by labeler
    notes: str = ""


class SignalRejected(BaseModel):
    event_type: Literal["signal_rejected"] = "signal_rejected"
    created_at: int
    user: Optional[str] = None
    signal_id: str
    artifact_name: str
    tokenizer_version: str
    symbol: str
    timeframe: str
    action: str
    reason_code: Optional[str] = None  # e.g. "wrong_role", "stale_data", "low_conviction"
    notes: str = ""


# Convenience union for parsing arbitrary lines from the store
def parse_feedback_line(line: dict) -> CorrectedTrendline | SignalAccepted | SignalRejected:
    et = line.get("event_type")
    if et == "corrected_trendline":
        return CorrectedTrendline(**line)
    if et == "signal_accepted":
        return SignalAccepted(**line)
    if et == "signal_rejected":
        return SignalRejected(**line)
    raise ValueError(f"unknown event_type: {et!r}")
