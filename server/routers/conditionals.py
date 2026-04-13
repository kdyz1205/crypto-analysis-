"""HTTP API for conditional orders tied to manually-drawn trendlines.

Endpoints:
  POST   /api/conditionals                           — create from a manual line
  GET    /api/conditionals                           — list all
  GET    /api/conditionals/{id}                      — detail + events log
  POST   /api/conditionals/{id}/cancel               — manual cancel
  DELETE /api/conditionals/{id}                      — hard delete
  POST   /api/drawings/manual/{id}/analyze           — compute pattern stats for a drawn line
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..conditionals import (
    ConditionalEvent,
    ConditionalOrder,
    ConditionalOrderStore,
    OrderConfig,
    TriggerConfig,
    now_ts,
)
from ..drawings.store import ManualTrendlineStore
from .strategy import _normalize_symbol


router = APIRouter(prefix="/api", tags=["conditionals"])

_store = ConditionalOrderStore()
_drawings = ManualTrendlineStore()


# ─────────────────────────────────────────────────────────────
# Request/response schemas
# ─────────────────────────────────────────────────────────────
class _TriggerReq(BaseModel):
    tolerance_atr: float = 0.2
    poll_seconds: int = 60
    max_age_seconds: int = 48 * 3600
    max_distance_atr: float = 5.0
    break_threshold_atr: float = 0.5


class _OrderReq(BaseModel):
    direction: Literal["long", "short"] | None = None  # auto from (side, kind) if None
    order_kind: Literal["bounce", "breakout"] = "bounce"
    entry_offset_points: float | None = None  # absolute price offset, takes precedence
    entry_offset_atr: float = 0.0
    stop_points: float | None = None
    stop_atr: float = 0.3
    rr_target: float | None = 2.0
    tp_price: float | None = None
    notional_usd: float | None = None
    equity_pct: float | None = None
    risk_pct: float | None = 0.005
    submit_to_exchange: bool = False
    exchange_mode: Literal["paper", "live"] = "paper"


class ConditionalCreateReq(BaseModel):
    manual_line_id: str = Field(..., description="FK to ManualTrendline")
    trigger: _TriggerReq = Field(default_factory=_TriggerReq)
    order: _OrderReq = Field(default_factory=_OrderReq)
    pattern_stats: dict[str, Any] = Field(default_factory=dict,
        description="Whatever the /analyze endpoint returned at creation time. "
                    "Stored verbatim for later audit.")


def _auto_direction(side: str, kind: str) -> str:
    """Direction matrix:
        support + bounce     → long  (buy the bounce)
        support + breakout   → short (sell the breakdown)
        resistance + bounce  → short (sell the rejection)
        resistance + breakout → long (buy the breakout)
    """
    if side == "support":
        return "long" if kind == "bounce" else "short"
    # resistance
    return "short" if kind == "bounce" else "long"


def _auto_poll_seconds(timeframe: str) -> int:
    table = {
        "1m": 5, "3m": 10, "5m": 15, "15m": 15, "30m": 30,
        "1h": 60, "2h": 60, "4h": 60,
        "6h": 120, "12h": 180, "1d": 300, "1w": 900,
    }
    return table.get(timeframe, 60)


@router.post("/conditionals")
async def api_create_conditional(req: ConditionalCreateReq):
    drawing = _drawings.get(req.manual_line_id)
    if drawing is None:
        raise HTTPException(404, f"manual line not found: {req.manual_line_id}")

    # Auto-derive direction from (line side, order kind). The user can still
    # override, but the default covers all 4 combinations — it's no longer
    # true that "long cannot be on a resistance line" (a breakout trader
    # DOES go long on a resistance break).
    kind = req.order.order_kind
    direction = req.order.direction or _auto_direction(drawing.side, kind)

    poll = req.trigger.poll_seconds or _auto_poll_seconds(drawing.timeframe)

    order = ConditionalOrder(
        conditional_id=_mk_id(drawing.manual_line_id),
        manual_line_id=drawing.manual_line_id,
        symbol=drawing.symbol,
        timeframe=drawing.timeframe,
        side=drawing.side,
        t_start=drawing.t_start,
        t_end=drawing.t_end,
        price_start=drawing.price_start,
        price_end=drawing.price_end,
        pattern_stats_at_create=dict(req.pattern_stats),
        trigger=TriggerConfig(
            tolerance_atr=req.trigger.tolerance_atr,
            poll_seconds=poll,
            max_age_seconds=req.trigger.max_age_seconds,
            max_distance_atr=req.trigger.max_distance_atr,
            break_threshold_atr=req.trigger.break_threshold_atr,
        ),
        order=OrderConfig(
            direction=direction,  # type: ignore
            order_kind=kind,
            entry_offset_points=req.order.entry_offset_points,
            entry_offset_atr=req.order.entry_offset_atr,
            stop_points=req.order.stop_points,
            stop_atr=req.order.stop_atr,
            rr_target=req.order.rr_target,
            tp_price=req.order.tp_price,
            notional_usd=req.order.notional_usd,
            equity_pct=req.order.equity_pct,
            risk_pct=req.order.risk_pct,
            submit_to_exchange=req.order.submit_to_exchange,
            exchange_mode=req.order.exchange_mode,
        ),
        status="pending",
        created_at=now_ts(),
        updated_at=now_ts(),
    )
    try:
        _store.create(order)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True, "conditional": order.to_dict()}


@router.get("/conditionals")
async def api_list_conditionals(
    status: Literal["pending", "triggered", "cancelled", "failed", "all"] = Query("all"),
    symbol: str | None = Query(None),
):
    items = _store.list_all(
        status=None if status == "all" else status,
        symbol=_normalize_symbol(symbol) if symbol else None,
    )
    return {
        "ok": True,
        "count": len(items),
        "conditionals": [i.to_dict() for i in items],
    }


@router.get("/conditionals/{conditional_id}")
async def api_get_conditional(conditional_id: str):
    item = _store.get(conditional_id)
    if item is None:
        raise HTTPException(404, f"conditional not found: {conditional_id}")
    return {"ok": True, "conditional": item.to_dict()}


@router.post("/conditionals/{conditional_id}/cancel")
async def api_cancel_conditional(conditional_id: str, reason: str = Query("manual_cancel")):
    item = _store.get(conditional_id)
    if item is None:
        raise HTTPException(404, "not found")
    if item.status != "pending":
        raise HTTPException(409, f"cannot cancel (status={item.status})")
    _store.append_event(conditional_id, ConditionalEvent(
        ts=now_ts(), kind="cancelled", message=reason,
    ))
    updated = _store.set_status(conditional_id, "cancelled", reason=reason)
    return {"ok": True, "conditional": updated.to_dict() if updated else None}


@router.delete("/conditionals/{conditional_id}")
async def api_delete_conditional(conditional_id: str):
    if not _store.delete(conditional_id):
        raise HTTPException(404, "not found")
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# Pattern stats for a drawn line (no conditional yet)
# ─────────────────────────────────────────────────────────────
class AnalyzeReq(BaseModel):
    manual_line_id: str
    k: int = Field(30, ge=5, le=100)


@router.post("/drawings/manual/analyze")
async def api_analyze_drawing(req: AnalyzeReq):
    """Compute pattern statistics for a drawn line.

    Uses the pattern engine's match_pattern to find similar historical
    structures and return P(bounce)/P(break)/P(fake)/EV.
    """
    drawing = _drawings.get(req.manual_line_id)
    if drawing is None:
        raise HTTPException(404, f"manual line not found: {req.manual_line_id}")

    try:
        from tools.pattern_engine import match_pattern
    except ImportError as e:
        return {
            "ok": False,
            "error": f"pattern engine not available: {e}",
            "stats": _fallback_stats(drawing),
        }

    # Load candles for the symbol/timeframe
    try:
        from ..data_service import get_ohlcv_with_df
        polars_df, _ = await get_ohlcv_with_df(
            drawing.symbol, drawing.timeframe, None, days=365,
            history_mode="fast_window",
            include_price_precision=False,
            include_render_payload=False,
        )
        if polars_df is None or polars_df.is_empty():
            raise ValueError("no candles available")
        pdf = polars_df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
        pdf = pdf.rename(columns={"open_time": "timestamp"})
        import pandas as _pd
        pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(_pd.Timestamp(v).timestamp()))
    except Exception as e:
        return {
            "ok": False,
            "error": f"candle load failed: {e}",
            "stats": _fallback_stats(drawing),
        }

    # Map the drawing's timestamps to bar indices in pdf
    ts_arr = pdf["timestamp"].values
    a1_idx = _nearest_bar_index(ts_arr, drawing.t_start)
    a2_idx = _nearest_bar_index(ts_arr, drawing.t_end)
    if a1_idx is None or a2_idx is None or a2_idx <= a1_idx:
        return {
            "ok": False,
            "error": "could not map line anchors to bars",
            "stats": _fallback_stats(drawing),
        }

    try:
        import asyncio as _asyncio
        match_result = await _asyncio.to_thread(
            match_pattern,
            pdf,
            a1_idx, float(drawing.price_start),
            a2_idx, float(drawing.price_end),
            drawing.side, drawing.symbol, drawing.timeframe, req.k,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": f"match_pattern failed: {e}",
            "stats": _fallback_stats(drawing),
        }

    if not match_result or not isinstance(match_result, dict):
        return {
            "ok": True,
            "stats": _fallback_stats(drawing),
            "note": "match_pattern returned no data",
        }

    return {
        "ok": True,
        "stats": match_result.get("stats") or _fallback_stats(drawing),
        "anomaly": match_result.get("anomaly") or {},
        "similar_count": match_result.get("n_similar", 0),
        "manual_line_id": drawing.manual_line_id,
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _mk_id(manual_line_id: str) -> str:
    h = hashlib.sha1(f"{manual_line_id}|{time.time_ns()}".encode()).hexdigest()
    return f"cond_{h[:14]}"


def _nearest_bar_index(ts_arr, target_ts: int) -> int | None:
    import bisect
    ts_list = ts_arr.tolist() if hasattr(ts_arr, "tolist") else list(ts_arr)
    if not ts_list:
        return None
    idx = bisect.bisect_left(ts_list, int(target_ts))
    if idx >= len(ts_list):
        return len(ts_list) - 1
    if idx > 0 and (ts_list[idx] - target_ts) > (target_ts - ts_list[idx - 1]):
        return idx - 1
    return idx


def _fallback_stats(drawing) -> dict:
    """Minimal stats when pattern engine can't help — just line geometry."""
    span = max(1, drawing.t_end - drawing.t_start)
    slope_per_bar = (drawing.price_end - drawing.price_start) / span
    return {
        "sample_size": 0,
        "p_bounce": None,
        "p_break": None,
        "p_fake_break": None,
        "expected_value": None,
        "confidence": 0.0,
        "trustworthiness": "none",
        "overfit_flag": "insufficient_samples",
        "line_geometry": {
            "slope_per_sec": slope_per_bar,
            "price_start": drawing.price_start,
            "price_end": drawing.price_end,
            "span_seconds": span,
        },
    }


__all__ = ["router"]
