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

import asyncio
import hashlib
import os
import time
from datetime import datetime, timezone
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
from ..conditionals.exchange_cancel import cancel_exchange_order_for_cond
from ..drawings.store import ManualTrendlineStore
from .strategy import _normalize_symbol


router = APIRouter(prefix="/api", tags=["conditionals"])

_store = ConditionalOrderStore()
_drawings = ManualTrendlineStore()
_MARK_PRICE_CACHE: dict[str, tuple[float, float]] = {}
_LEVERAGE_SET_CACHE: dict[tuple[str, str, int], float] = {}
_MARK_PRICE_CACHE_SECONDS = 2.0
_LEVERAGE_SET_CACHE_SECONDS = 3600.0


def _require_manual_exchange_enabled(mode: str, adapter) -> None:
    """Gate direct manual-line exchange orders with live safety flags."""
    if not adapter.has_api_keys():
        raise HTTPException(503, "Bitget API keys not configured")

    enable_live = os.environ.get("ENABLE_LIVE_TRADING", "false").lower() == "true"
    confirm_live = os.environ.get("CONFIRM_LIVE_TRADING", "false").lower() == "true"
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"

    if not enable_live:
        raise HTTPException(
            403,
            {
                "reason": "enable_live_trading_disabled",
                "message": "ENABLE_LIVE_TRADING must be true before placing exchange plan orders.",
            },
        )

    if mode == "live":
        blocked = []
        if not confirm_live:
            blocked.append("confirm_live_trading_disabled")
        if dry_run:
            blocked.append("dry_run_enabled")
        if blocked:
            raise HTTPException(
                403,
                {
                    "reason": "live_trading_not_confirmed",
                    "blocking_reasons": blocked,
                    "message": "Real-money manual line orders require CONFIRM_LIVE_TRADING=true and DRY_RUN=false.",
                },
            )


# ─────────────────────────────────────────────────────────────
# Request/response schemas
# ─────────────────────────────────────────────────────────────
class _TriggerReq(BaseModel):
    tolerance_atr: float = 0.2
    poll_seconds: int = 60
    # User policy: no time/drift expiry. Lines live until user deletes.
    max_age_seconds: int = 0
    max_distance_atr: float = 0.0
    break_threshold_atr: float = 0.5


class _OrderReq(BaseModel):
    direction: Literal["long", "short"] | None = None  # auto from (side, kind) if None
    order_kind: Literal["bounce", "breakout"] = "bounce"
    # Legacy absolute-offset fields — kept in schema for back-compat but
    # the watcher no longer reads them. Use *_pct_of_line instead.
    entry_offset_points: float | None = None
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
    # Line-relative % — the primary path. tolerance_pct_of_line is the
    # entry buffer ("put my order X% above the support line to guarantee
    # fill"); stop_offset_pct_of_line is the stop on the other side.
    tolerance_pct_of_line: float = 0.0
    stop_offset_pct_of_line: float = 0.0
    # Cross-margin leverage. If set, notional = account_equity * leverage.
    leverage: float | None = None
    # Auto-reverse on stop-loss.
    reverse_enabled: bool = False
    reverse_entry_offset_pct: float = 0.0
    reverse_stop_offset_pct: float = 0.0
    reverse_rr_target: float | None = None
    reverse_leverage: float | None = None


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


def _project_manual_line_price(drawing, ts: int) -> float:
    """Project a manual drawing's price at timestamp `ts`.

    Uses LOG interpolation to match lightweight-charts' visual rendering
    when the chart is in Log mode (our default). A straight line drawn
    between 2 points on a log-scale axis = constant growth rate in price
    space. Linear-in-price interpolation would be 0.47% off on typical
    small-cap slopes (user report 2026-04-21 BAS/4h).
    """
    import math
    span = int(drawing.t_end) - int(drawing.t_start)
    if span <= 0:
        return float(drawing.price_start)
    if ts <= int(drawing.t_start) and not bool(getattr(drawing, "extend_left", False)):
        return float(drawing.price_start)
    if ts >= int(drawing.t_end) and not bool(getattr(drawing, "extend_right", True)):
        return float(drawing.price_end)
    p_start = float(drawing.price_start)
    p_end = float(drawing.price_end)
    if p_start <= 0 or p_end <= 0:
        # Fallback to linear for degenerate cases
        slope = (p_end - p_start) / span
        return p_start + slope * (ts - int(drawing.t_start))
    ratio = (ts - int(drawing.t_start)) / span
    return math.exp(math.log(p_start) + ratio * (math.log(p_end) - math.log(p_start)))


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
        extend_left=bool(drawing.extend_left),
        extend_right=bool(drawing.extend_right),
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
            tolerance_pct_of_line=req.order.tolerance_pct_of_line,
            stop_offset_pct_of_line=req.order.stop_offset_pct_of_line,
            leverage=req.order.leverage,
            reverse_enabled=req.order.reverse_enabled,
            reverse_entry_offset_pct=req.order.reverse_entry_offset_pct,
            reverse_stop_offset_pct=req.order.reverse_stop_offset_pct,
            reverse_rr_target=req.order.reverse_rr_target,
            reverse_leverage=req.order.reverse_leverage,
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
    # Allow cancel of pending (virtual watcher) AND triggered (live Bitget
    # order). A triggered conditional has an exchange_order_id which we
    # need to cancel on Bitget itself before flipping local state.
    if item.status not in ("pending", "triggered"):
        raise HTTPException(409, f"cannot cancel (status={item.status})")

    bitget_cancel = None
    if item.exchange_order_id:
        bitget_cancel = await cancel_exchange_order_for_cond(item)
        if not bitget_cancel.get("ok"):
            _store.append_event(conditional_id, ConditionalEvent(
                ts=now_ts(),
                kind="exchange_error",
                message="manual cancel refused: Bitget cancel not confirmed",
                extra={"bitget_cancel": bitget_cancel},
            ))
            raise HTTPException(
                409,
                {
                    "reason": "bitget_cancel_not_confirmed",
                    "message": (
                        "Bitget cancel was not confirmed; local conditional was "
                        "left active so the UI does not hide a live order."
                    ),
                    "bitget_cancel": bitget_cancel,
                },
            )

    _store.append_event(conditional_id, ConditionalEvent(
        ts=now_ts(), kind="cancelled", message=reason,
        extra={"bitget_cancel": bitget_cancel},
    ))
    updated = _store.set_status(conditional_id, "cancelled", reason=reason)
    return {
        "ok": True,
        "bitget_cancelled": None if bitget_cancel is None else bool(bitget_cancel.get("ok")),
        "bitget_cancel": bitget_cancel,
        "conditional": updated.to_dict() if updated else None,
    }


@router.delete("/conditionals/{conditional_id}")
async def api_delete_conditional(
    conditional_id: str,
    force: bool = Query(False, description="force delete even with live Bitget order"),
):
    item = _store.get(conditional_id)
    if item is None:
        raise HTTPException(404, "not found")

    bitget_cancel = None
    # If there's a live Bitget order, try to cancel it first
    if item.exchange_order_id and item.status == "triggered":
        bitget_cancel = await cancel_exchange_order_for_cond(item)
        if not bitget_cancel.get("ok") and not force:
            _store.append_event(conditional_id, ConditionalEvent(
                ts=now_ts(),
                kind="exchange_error",
                message="delete refused: Bitget cancel not confirmed",
                extra={"bitget_cancel": bitget_cancel},
            ))
            raise HTTPException(
                409,
                {
                    "reason": "bitget_cancel_not_confirmed",
                    "message": (
                        f"Refusing to delete local record because Bitget order "
                        f"{item.exchange_order_id} may still be live."
                    ),
                    "bitget_cancel": bitget_cancel,
                },
            )

    if not _store.delete(conditional_id):
        raise HTTPException(404, "not found (race?)")
    return {
        "ok": True,
        "bitget_cancelled": None if bitget_cancel is None else bool(bitget_cancel.get("ok")),
        "bitget_cancel": bitget_cancel,
    }


# ─────────────────────────────────────────────────────────────
# Place a REAL pending limit order on Bitget, derived from a drawn line.
# This bypasses the virtual watcher — the order sits on the exchange,
# visible in the user's Bitget app, and either fills at the line price
# or is cancelled manually.
# ─────────────────────────────────────────────────────────────
class PlaceLineOrderReq(BaseModel):
    manual_line_id: str
    direction: Literal["long", "short"]
    kind: Literal["bounce", "break"] = "bounce"
    tolerance_pct: float = 0.1
    stop_offset_pct: float = 0.04
    size_usdt: float = Field(..., gt=0, description="Notional to commit, in USDT")
    leverage: int = Field(5, ge=1, le=100)
    # User 2026-04-22: if equity_pct > 0, store it so replan can
    # dynamically resize qty as account equity changes. Frontend sends
    # whichever mode user chose. size_usdt is still authoritative for
    # the INITIAL placement (computed client-side from current equity).
    equity_pct: float | None = None
    mode: Literal["demo", "live"] = "live"
    rr_target: float = 2.0
    notify_tg: bool = Field(True, description="Send TG alert on trigger")
    reverse_enabled: bool = False
    reverse_entry_offset_pct: float = 0.0
    reverse_stop_offset_pct: float = 0.0
    reverse_rr_target: float | None = None
    reverse_leverage: float | None = None
    # Unix seconds. Client passes the CURRENT LIVE BAR's open_ts (= the
    # rightmost candle's `time` on the chart). Server projects the line
    # at this ts so the entry matches the height the user visually sees
    # on today's bar. Falls back to wall-clock now if not provided.
    # See TA_BASICS.md + 2026-04-23 ZEC incident.
    reference_ts: int | None = None


_TF_LADDER = ["1m", "3m", "5m", "15m", "1h", "4h", "1d", "1w"]


async def multi_tf_coherence(symbol: str, timeframe: str, line_price: float) -> dict:
    """Check if the user's line price aligns with auto-detected zones/lines
    on HIGHER timeframes (1-2 steps up the ladder).

    A "match" = any horizontal zone center or active trendline projection
    within 0.5% of `line_price`. More matches = stronger structural support.

    Returns:
      {
        "score": 0..2,             # number of higher TFs with a match
        "matches": [               # list of TF + price + reason for each hit
          {"tf": "15m", "price": 44.41, "kind": "zone", "strength": 61.3},
          ...
        ],
        "checked_tfs": ["15m", "1h"],
      }
    """
    try:
        idx = _TF_LADDER.index(timeframe)
    except ValueError:
        return {"score": 0, "matches": [], "checked_tfs": []}
    higher = _TF_LADDER[idx + 1 : idx + 3]  # next 2 TFs
    if not higher or line_price <= 0:
        return {"score": 0, "matches": [], "checked_tfs": []}

    matches: list[dict] = []
    tol = line_price * 0.005  # 0.5% match window
    sym = symbol.upper()

    import httpx
    async with httpx.AsyncClient(timeout=8.0) as client:
        for tf in higher:
            try:
                r = await client.get(
                    "http://127.0.0.1:8000/api/strategy/snapshot",
                    params={"symbol": sym, "interval": tf},
                )
                if r.status_code != 200:
                    continue
                snap = (r.json() or {}).get("snapshot") or {}
                hit = None
                for z in (snap.get("horizontal_zones") or []):
                    pc = float(z.get("price_center") or 0)
                    if abs(pc - line_price) <= tol:
                        hit = {"tf": tf, "price": pc, "kind": "zone",
                               "strength": float(z.get("strength") or 0)}
                        break
                if hit is None:
                    for ln in (snap.get("active_lines") or []):
                        pp = float(ln.get("projected_price_current") or 0)
                        if pp > 0 and abs(pp - line_price) <= tol:
                            hit = {"tf": tf, "price": pp, "kind": "line",
                                   "strength": float(ln.get("line_score") or 0)}
                            break
                if hit:
                    matches.append(hit)
            except Exception:
                continue

    return {"score": len(matches), "matches": matches, "checked_tfs": higher}


async def volatility_context(symbol: str, timeframe: str) -> dict:
    """Single source of truth for "how volatile is this symbol/TF right now".

    Used by:
      • _derive_recommendations to floor tolerance/stop on ATR
      • watcher replan threshold (drift_pct comparison)
      • any future smart features (regime detection, distance-adaptive, etc.)

    Returns:
      {
        "atr_abs": float,    # absolute ATR in price points
        "atr_pct": float,    # ATR as percent of mark (e.g. 0.45 = 0.45%)
        "mark":    float,    # current mark price
        "regime":  "tight" | "normal" | "wide",
      }
    All fields fall back to safe defaults if data unavailable.
    """
    mark = await _fetch_bitget_mark_price(symbol) or 0.0
    atr_abs = 0.0
    bb_state = "normal"
    bbw_pct = 0.0
    bbw_percentile = 0.5
    try:
        from server.conditionals.watcher import _fetch_current_atr
        atr_abs = await _fetch_current_atr(symbol, timeframe) or 0.0
    except Exception:
        pass
    # P3: BB-width squeeze detection
    try:
        from server.data_service import get_ohlcv_with_df
        polars_df, _ = await get_ohlcv_with_df(
            symbol, timeframe, None, days=14,
            history_mode="fast_window",
            include_price_precision=False,
            include_render_payload=False,
        )
        if polars_df is not None and not polars_df.is_empty():
            pdf2 = polars_df.tail(120).to_pandas()
            import numpy as _np
            close = pdf2["close"].astype(float).values
            if len(close) >= 25:
                window = 20
                ma = _np.array([close[max(0,i-window+1):i+1].mean() for i in range(len(close))])
                sd = _np.array([close[max(0,i-window+1):i+1].std() for i in range(len(close))])
                bbw = (4 * sd) / _np.maximum(ma, 1e-9) * 100  # band width as %
                bbw_now = float(bbw[-1])
                bbw_hist = bbw[max(0, len(bbw)-100):]
                bbw_pct = bbw_now
                # percentile of current BBW vs last 100 bars
                rank = float((bbw_hist <= bbw_now).sum()) / len(bbw_hist)
                bbw_percentile = round(rank, 3)
                if rank <= 0.20:
                    bb_state = "squeeze"      # bottom 20% → consolidation
                elif rank >= 0.80:
                    bb_state = "expansion"    # top 20% → already moving
                else:
                    bb_state = "normal"
    except Exception:
        pass
    atr_pct = (atr_abs / mark * 100.0) if mark > 0 else 0.0
    if atr_pct < 0.25:
        regime = "tight"
    elif atr_pct < 0.8:
        regime = "normal"
    else:
        regime = "wide"

    return {
        "atr_abs": atr_abs,
        "atr_pct": atr_pct,
        "mark": mark,
        "regime": regime,
        "bb_state": bb_state,
        "bbw_pct": round(bbw_pct, 3),
        "bbw_percentile": bbw_percentile,
    }


async def _fetch_bitget_mark_price(symbol: str) -> float | None:
    """Fetch real-time mark price from Bitget public ticker (no auth)."""
    import httpx
    normalized = symbol.upper()
    now = time.time()
    cached = _MARK_PRICE_CACHE.get(normalized)
    if cached and now - cached[0] <= _MARK_PRICE_CACHE_SECONDS:
        return cached[1]

    url = "https://api.bitget.com/api/v2/mix/market/ticker"
    try:
        timeout = httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=0.5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params={"symbol": normalized, "productType": "USDT-FUTURES"})
            if r.status_code != 200:
                return None
            j = r.json()
            data = j.get("data") or []
            if not data:
                return None
            row = data[0] if isinstance(data, list) else data
            mp = row.get("markPrice") or row.get("lastPr")
            if not mp:
                return None
            price = float(mp)
            _MARK_PRICE_CACHE[normalized] = (now, price)
            return price
    except Exception:
        return None


@router.post("/conditionals/place-line-order")   # legacy alias for cached frontends
@router.post("/drawings/manual/place-line-order")
async def api_place_line_order(req: PlaceLineOrderReq):
    """Submit a Bitget trigger-market plan order for a manual line."""
    drawing = _drawings.get(req.manual_line_id)
    if drawing is None:
        raise HTTPException(404, f"manual line not found: {req.manual_line_id}")

    # Project the line at the CURRENT LIVE BAR's open_ts.
    #
    # History (2026-04-20 → 2026-04-23):
    #   v1: exact click ms → ~2.5h drift inside a 4h bar (BAS report).
    #   v2: snap `floor(now/tf_sec)*tf_sec` → worked for 4h but assumed
    #       UTC-00:00 bar alignment. For Bitget 1d (UTC+8 anchor, bar
    #       opens at UTC 16:00), this floors to UTC 00:00 which is NOT
    #       the actual bar boundary — drifts up to 16h.
    #   v3: wall-clock now → better but user's eye reads the bar at its
    #       chart-rendered X position (= bar_open_ts), not at "now"
    #       midway through the bar. ZEC 1d: wall-clock now gave 317.83,
    #       visual read 315.81. Gap = 2 points = ~10h of line slope.
    #
    #   v4 (this): use the CURRENT LIVE BAR's open_ts as the reference.
    #     - Client passes `reference_ts` = time of the rightmost candle
    #       on their chart (= Bitget-exchange-aligned bar_open_ts, which
    #       for 1d is UTC 16:00, for 4h/1h/15m is the intraday boundary).
    #     - Server uses this directly. Matches the exact pixel position
    #       where the user sees the line cross the live candle.
    #     - If the client doesn't send it (older UI), fall back to now.
    from server.conditionals.types import _TF_SECONDS  # noqa: F401 — kept for any future guard
    now_ts_ = now_ts()
    ref_ts = int(req.reference_ts) if req.reference_ts and req.reference_ts > 0 else 0
    # Guard: reference_ts mustn't be before the line's first anchor
    # (nonsensical) or wildly in the future (indicates a bug).
    if ref_ts and ref_ts < int(drawing.t_start) - 86400:
        ref_ts = 0
    if ref_ts and ref_ts > int(now_ts_) + 86400:
        ref_ts = 0

    # Server-side fallback: deterministic math anchor. No network calls.
    #
    # Bitget USDT-M futures 1d bars open at UTC 16:00 each day (= UTC+8
    # midnight, the Asian-exchange convention). All intraday bars align
    # to UTC hour/minute boundaries. For any timeframe we can compute
    # the current live bar's open_ts with pure arithmetic on `now`.
    #
    # Previous attempts — direct Bitget fetch, data_service.get_ohlcv
    # — both went through httpx to Bitget and the pool wedged
    # intermittently, causing silent fallbacks to wall-clock now. After
    # 3 successful placements the 4th would time out and skew again.
    # This version has zero async/network dependency.
    if not ref_ts:
        tf = drawing.timeframe
        _ONE_DAY = 86400
        _BITGET_1D_ANCHOR_SEC = 16 * 3600  # UTC 16:00 = UTC+8 00:00
        _ONE_WEEK = 7 * _ONE_DAY
        tf_to_seconds = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
        }
        now_i = int(now_ts_)
        if tf == "1d":
            # floor to nearest (anchor + N*day)
            ref_ts = ((now_i - _BITGET_1D_ANCHOR_SEC) // _ONE_DAY) * _ONE_DAY + _BITGET_1D_ANCHOR_SEC
        elif tf == "1w":
            # 1w bars open at UTC+8 midnight on a specific day; keep daily-anchor
            # floor as safe default (same cadence as 1d for slope on weekly).
            ref_ts = ((now_i - _BITGET_1D_ANCHOR_SEC) // _ONE_WEEK) * _ONE_WEEK + _BITGET_1D_ANCHOR_SEC
        elif tf in tf_to_seconds:
            sec = tf_to_seconds[tf]
            ref_ts = (now_i // sec) * sec
        else:
            ref_ts = now_i
        print(
            f"[place-line-order] ref_ts computed {drawing.symbol} {tf}: "
            f"ref_ts={ref_ts} ({datetime.fromtimestamp(ref_ts, tz=timezone.utc).isoformat()}) "
            f"from now={now_i}",
            flush=True,
        )

    if not ref_ts:
        # Absolute last-resort (should be unreachable after the math block above).
        ref_ts = int(now_ts_)

    line_now = _project_manual_line_price(drawing, ref_ts)
    ref_price = float(line_now)
    if ref_price <= 0:
        raise HTTPException(
            400,
            f"line has no usable price (computed ref={ref_price:.6f}, "
            f"price_start={drawing.price_start}, price_end={drawing.price_end}, "
            f"t_start={drawing.t_start}, t_end={drawing.t_end}, now={now_ts_}). "
            f"This shouldn't happen after the clamp - report it."
        )

    # Trust the user's direction. Only safety guard: reject obviously stale
    # lines far away from the current mark, using ATR as the scale.
    # 2026-04-22: wrap in wait_for(1.5s) so a slow Bitget ticker doesn't
    # stall the whole placement. User reported mark_ms=6842 when Bitget
    # throttled → popup timed out after 30s → user retried 3 times →
    # 3 duplicate orders. Mark is only used for a 100× sanity guard;
    # skipping it is fine if Bitget is slow.
    timing_start = time.perf_counter()
    mark_price = None
    try:
        mark_price = await asyncio.wait_for(
            _fetch_bitget_mark_price(drawing.symbol),
            timeout=1.5,
        )
    except asyncio.TimeoutError:
        print(f"[place-line-order] mark fetch >1.5s, proceeding without sanity guard for {drawing.symbol}", flush=True)
    mark_elapsed_ms = int((time.perf_counter() - timing_start) * 1000)
    # Distance-from-mark check REMOVED 2026-04-21. User strategy depends on
    # placing orders at structurally-important levels that can be 20-50%+
    # away from current price — the whole EDGE is "wait for price to come
    # to the line". Iterations 10% → 25% → removed. Only keep a minimal
    # sanity guard against truly broken inputs (price ≤ 0, or > 100×
    # mark which suggests a UI coordinate-transform bug).
    if mark_price and mark_price > 0 and ref_price > 0:
        ratio = ref_price / mark_price
        if ratio > 100 or ratio < 0.01:
            msg = (
                f"line_price_bug_suspected: {drawing.symbol} "
                f"line@{ref_price:.4f} vs mark@{mark_price:.4f} "
                f"(ratio={ratio:.2f}×). This is 100× off — looks like "
                f"a drawing-coord bug. Redraw the line."
            )
            print(f"[place-line-order] REJECTED: {msg}", flush=True)
            raise HTTPException(400, msg)

    print(
        f"[place-line-order] line={drawing.price_start:.4f}->{drawing.price_end:.4f} "
        f"ref@{ref_price:.4f} mark={mark_price} kind={req.kind} dir={req.direction} "
        f"mark_ms={mark_elapsed_ms}"
    )

    # Compute prices from the bar-open-projected line:
    # LONG:  entry just ABOVE the line (waiting for price to pull down
    #        to the level from above; Bitget plan fires on mark ≤ trigger),
    #        stop BELOW the line (break-through kill switch).
    # SHORT: entry just BELOW the line (rejection from above), stop ABOVE.
    #
    # NOTE: bounce vs breakout order_kind is stored but does NOT flip these
    # formulas. The formulas are MODE-AGNOSTIC because Bitget's trigger-
    # market plan naturally handles both semantics via the trigger price's
    # position relative to current mark:
    #   - mark > trigger + a LONG plan → "wait for dip" (bounce long)
    #   - mark < trigger + a LONG plan → "wait for break up" (breakout long)
    # Same plan body, different interpretations. The user sets trigger via
    # (line × buffer), Bitget figures out the rest.
    offset = req.tolerance_pct / 100.0
    stop_offset = max(0.0, req.stop_offset_pct) / 100.0
    if req.direction == "long":
        entry_price = ref_price * (1.0 + offset)
        stop_price = ref_price * (1.0 - stop_offset)
    else:
        entry_price = ref_price * (1.0 - offset)
        stop_price = ref_price * (1.0 + stop_offset)

    # SL distance cap: use USER's configured total (buffer + stop).
    # Normal bounce fill: entry ≈ trigger, SL at line - stop, distance
    # = buffer + stop → cap not triggered (SL already within user's setup).
    # Breakout fill: market blew past trigger, entry > trigger (long)
    # or < trigger (short). SL still at line-based but distance from
    # ACTUAL entry can be 3-5% — WAY over what user set (user report
    # BAS: setup 1.1%, actual stop 3.5% because breakout fill).
    # Fix: cap SL at entry × (1 ∓ user's buffer+stop %) so the real
    # stop distance matches the user's intended risk regardless of
    # fill quality.
    max_sl_pct = (req.tolerance_pct + max(0.0, req.stop_offset_pct)) / 100.0
    if req.direction == "long":
        min_sl_cap = entry_price * (1.0 - max_sl_pct)
        if stop_price < min_sl_cap:
            print(f"[place-line-order] SL distance-capped at setup total: "
                  f"line-based {stop_price:.6f} -> {min_sl_cap:.6f} "
                  f"(entry {entry_price:.6f}, cap {max_sl_pct*100:.2f}% = buffer+stop)", flush=True)
            stop_price = min_sl_cap
    else:
        max_sl_cap = entry_price * (1.0 + max_sl_pct)
        if stop_price > max_sl_cap:
            print(f"[place-line-order] SL distance-capped at setup total: "
                  f"line-based {stop_price:.6f} -> {max_sl_cap:.6f} "
                  f"(entry {entry_price:.6f}, cap {max_sl_pct*100:.2f}% = buffer+stop)", flush=True)
            stop_price = max_sl_cap

    # Recompute TP using capped risk (so TP adjusts too)
    if req.direction == "long":
        risk = entry_price - stop_price
        tp_price = entry_price + risk * req.rr_target
    else:
        risk = stop_price - entry_price
        tp_price = entry_price - risk * req.rr_target

    quantity = req.size_usdt / entry_price

    from ..execution.types import OrderIntent
    intent = OrderIntent(
        order_intent_id=f"line_{req.manual_line_id[-12:]}_{int(time.time())}",
        signal_id="manual",
        line_id=req.manual_line_id,
        client_order_id=f"line_{int(time.time()*1000)}",
        symbol=drawing.symbol.upper(),
        timeframe=drawing.timeframe,
        side=req.direction,         # type: ignore
        order_type="market",
        trigger_mode="manual",
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
        quantity=quantity,
        status="pending",
        reason="manual_line_draw",
        created_at_bar=-1,
        created_at_ts=time.time(),
        post_only=False,
    )

    from ..execution.live_adapter import LiveExecutionAdapter
    adapter = LiveExecutionAdapter()
    _require_manual_exchange_enabled(req.mode, adapter)
    leverage_key = (req.mode, drawing.symbol.upper(), int(req.leverage))
    leverage_cached_at = _LEVERAGE_SET_CACHE.get(leverage_key, 0.0)
    leverage_elapsed_ms = 0
    if time.time() - leverage_cached_at > _LEVERAGE_SET_CACHE_SECONDS:
        leverage_start = time.perf_counter()
        try:
            leverage_resp = await asyncio.wait_for(
                adapter._bitget_request(
                    "POST", "/api/v2/mix/account/set-leverage",
                    mode=req.mode,
                    body={
                        "symbol": drawing.symbol.upper(),
                        "productType": "USDT-FUTURES",
                        "marginCoin": "USDT",
                        "leverage": str(int(req.leverage)),
                    },
                ),
                timeout=3.0,
            )
            if leverage_resp.get("code") == "00000":
                _LEVERAGE_SET_CACHE[leverage_key] = time.time()
            else:
                print(f"[place-line-order] set-leverage rejected: {leverage_resp}", flush=True)
        except Exception as e:
            print(f"[place-line-order] set-leverage warning: {e}", flush=True)
        leverage_elapsed_ms = int((time.perf_counter() - leverage_start) * 1000)

    submit_start = time.perf_counter()
    result = await adapter.submit_live_plan_entry(intent, mode=req.mode, trigger_price=entry_price)
    submit_elapsed_ms = int((time.perf_counter() - submit_start) * 1000)
    total_elapsed_ms = int((time.perf_counter() - timing_start) * 1000)
    print(
        f"[place-line-order] bitget_submit ok={result.get('ok')} "
        f"mark_ms={mark_elapsed_ms} leverage_ms={leverage_elapsed_ms} "
        f"submit_ms={submit_elapsed_ms} total_ms={total_elapsed_ms}",
        flush=True,
    )
    if not result.get("ok"):
        return {
            "ok": False,
            "reason": result.get("reason", "bitget_rejected"),
            "result": result,
        }

    # Persist a record for our UI / cancel path
    cond = ConditionalOrder(
        conditional_id=_mk_id(req.manual_line_id),
        manual_line_id=req.manual_line_id,
        symbol=drawing.symbol,
        timeframe=drawing.timeframe,
        side=drawing.side,
        t_start=drawing.t_start,
        t_end=drawing.t_end,
        price_start=drawing.price_start,
        price_end=drawing.price_end,
        extend_left=bool(drawing.extend_left),
        extend_right=bool(drawing.extend_right),
        pattern_stats_at_create={},
        trigger=TriggerConfig(
            tolerance_atr=req.tolerance_pct / 0.3,
            poll_seconds=60,
            max_age_seconds=24 * 3600,
            max_distance_atr=200.0,
            break_threshold_atr=0.5,
        ),
        order=OrderConfig(
            direction=req.direction,
            order_kind="breakout" if req.kind == "break" else "bounce",
            # Store the absolute entry offset so replan can re-apply tolerance
            entry_offset_points=ref_price * req.tolerance_pct / 100.0,
            stop_points=ref_price * max(0.0, req.stop_offset_pct) / 100.0,
            tolerance_pct_of_line=req.tolerance_pct,
            stop_offset_pct_of_line=max(0.0, req.stop_offset_pct),
            rr_target=req.rr_target,
            tp_price=tp_price,
            notional_usd=req.size_usdt,
            # equity_pct + leverage stored so replan can dynamically
            # resize qty as equity changes. User 2026-04-22 spec.
            equity_pct=req.equity_pct if (req.equity_pct and req.equity_pct > 0) else None,
            leverage=float(req.leverage) if req.leverage else None,
            submit_to_exchange=True,
            exchange_mode=req.mode,
            reverse_enabled=req.reverse_enabled,
            reverse_entry_offset_pct=req.reverse_entry_offset_pct,
            reverse_stop_offset_pct=0.0,
            reverse_rr_target=req.reverse_rr_target,
            reverse_leverage=req.reverse_leverage,
        ),
        status="triggered",                                  # plan placed on Bitget
        created_at=now_ts(),
        updated_at=now_ts(),
        triggered_at=now_ts(),
        exchange_order_id=result.get("exchange_order_id"),
        fill_price=entry_price,
        fill_qty=quantity,
    )
    # Adaptive precision: small-cap coins (MYX 0.4) need 5 decimals, large
    # caps (HYPE 44) need 3-4. Pick based on price magnitude.
    def _fmt(p: float) -> str:
        if p >= 1000: return f"{p:.2f}"
        if p >= 100: return f"{p:.3f}"
        if p >= 10: return f"{p:.3f}"
        if p >= 1: return f"{p:.4f}"
        if p >= 0.1: return f"{p:.5f}"
        if p >= 0.01: return f"{p:.6f}"
        return f"{p:.7f}"

    try:
        _store.create(cond)
        # Seed watcher's pending-oids cache with the brand-new oid so the
        # next replan tick doesn't mistakenly skip this cond (within 3s
        # TTL window) thinking "plan not pending".
        try:
            from ..conditionals.watcher import _add_pending_oid_to_cache
            _add_pending_oid_to_cache(req.mode, str(result.get("exchange_order_id") or ""))
        except Exception:
            pass
        _store.append_event(cond.conditional_id, ConditionalEvent(
            ts=now_ts(),
            kind="exchange_submitted",
            price=ref_price,
            line_price=ref_price,
            message=(
                f"Bitget trigger-market plan placed: {req.direction} trigger@{_fmt(entry_price)} "
                f"sl@{_fmt(stop_price)} tp@{_fmt(tp_price)} qty={quantity:.6f} "
                f"orderId={result.get('exchange_order_id')}"
            ),
        ))
        try:
            from ..strategy.drawing_learner import capture_user_order_intent
            capture_user_order_intent(
                manual_line_id=req.manual_line_id,
                symbol=drawing.symbol,
                timeframe=drawing.timeframe,
                side=drawing.side,
                direction=req.direction,
                order_kind="breakout" if req.kind == "break" else "bounce",
                line_price=ref_price,
                entry_price=entry_price,
                stop_price=stop_price,
                tp_price=tp_price,
                tolerance_pct=req.tolerance_pct,
                stop_offset_pct=max(0.0, req.stop_offset_pct),
                rr_target=req.rr_target,
                size_usdt=req.size_usdt,
                exchange_order_id=str(result.get("exchange_order_id") or ""),
            )
        except Exception as exc:
            print(f"[drawing_learner] order-intent capture err: {exc}", flush=True)
    except Exception as e:
        # CRITICAL: cond create failed BUT Bitget order was placed.
        # We now have a ghost — Bitget order with no local cond.
        # Try to cancel the Bitget order so we don't leak.
        print(f"[place-line-order] cond create failed: {e} — rolling back Bitget order", flush=True)
        try:
            cancel_resp = await adapter.cancel_plan_order_any_type(
                drawing.symbol.upper(), str(result.get("exchange_order_id", "")), req.mode,
            )
            if not cancel_resp.get("ok"):
                await adapter.cancel_order(
                    drawing.symbol.upper(), str(result.get("exchange_order_id", "")), req.mode,
                )
            print(f"[place-line-order] rolled back ghost order {result.get('exchange_order_id')}", flush=True)
        except Exception as e2:
            print(f"[place-line-order] ROLLBACK FAILED — manual cleanup needed for {result.get('exchange_order_id')}: {e2}", flush=True)
        raise HTTPException(500, f"cond persistence failed after Bitget place: {e}")

    return {
        "ok": True,
        "conditional": cond.to_dict(),
        "exchange_order_id": result.get("exchange_order_id"),
        "message": (
            f"Bitget trigger-market plan submitted: {req.direction} trigger {_fmt(entry_price)} · "
            f"SL {_fmt(stop_price)} · TP {_fmt(tp_price)}."
        ),
    }



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
        # Line anchors out of historical range (e.g. drawn into the future).
        # Pattern matching impossible, but volatility / smart adjustments
        # still work — return a graceful "no patterns" response with vol_ctx
        # so the modal can still show ATR / recommendations.
        vol_ctx = await volatility_context(drawing.symbol, drawing.timeframe)
        return {
            "ok": True,
            "stats": _fallback_stats(drawing),
            "recommendations": _apply_smart_adjustments(
                _default_recommendations(), vol_ctx, None, None, None,
            ),
            "volatility": vol_ctx,
            "similar_lines": [],
            "note": "line anchors outside historical range — no pattern match, vol-only recs",
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
            "recommendations": _default_recommendations(),
            "similar_lines": [],
            "note": "match_pattern returned no data",
        }

    stats = match_result.get("stats") or _fallback_stats(drawing)

    # Convert top_matches' bar indices into time-based anchors so the
    # frontend can render them as overlay lines on the chart.
    similar_lines: list[dict] = []
    for entry in (match_result.get("top_matches") or [])[:20]:
        try:
            rec = entry.get("record") if isinstance(entry, dict) else None
            if not rec:
                continue
            a1 = rec.get("anchor1_idx")
            a2 = rec.get("anchor2_idx")
            p1 = rec.get("anchor1_price")
            p2 = rec.get("anchor2_price")
            if a1 is None or a2 is None or p1 is None or p2 is None:
                continue
            if a1 < 0 or a2 < 0 or a1 >= len(ts_arr) or a2 >= len(ts_arr):
                continue
            similar_lines.append({
                "t_start": int(ts_arr[int(a1)]),
                "t_end": int(ts_arr[int(a2)]),
                "price_start": float(p1),
                "price_end": float(p2),
                "distance": float(entry.get("distance", 0)),
                "outcome": rec.get("outcome", {}),
            })
        except Exception:
            continue

    vol_ctx = await volatility_context(drawing.symbol, drawing.timeframe)

    # ── P1: count touches on the user's actual line + distance from mark ──
    line_quality = _count_line_touches(
        pdf, a1_idx, a2_idx,
        float(drawing.price_start), float(drawing.price_end),
        atr_pct=float(vol_ctx.get("atr_pct") or 0.5),
        side=drawing.side,
    )
    distance_ctx = {}
    line_now = float(drawing.price_end)  # right anchor; close enough for UI
    mark = float(vol_ctx.get("mark") or 0)
    if mark > 0:
        gap_pct = abs(line_now - mark) / mark * 100.0
        distance_ctx = {
            "gap_pct": round(gap_pct, 3),
            "zone": "near" if gap_pct < 0.5 else ("medium" if gap_pct < 2.0 else "far"),
        }

    # P2: multi-TF coherence — does this line align with higher-TF structure?
    coherence = await multi_tf_coherence(drawing.symbol, drawing.timeframe, line_now)

    return {
        "ok": True,
        "stats": stats,
        "recommendations": _derive_recommendations(
            stats, match_result.get("top_matches") or [], vol_ctx,
            line_quality=line_quality, distance_ctx=distance_ctx,
            coherence=coherence,
        ),
        "volatility": vol_ctx,
        "line_quality": line_quality,
        "distance": distance_ctx,
        "coherence": coherence,
        "anomaly": match_result.get("anomaly") or {},
        "similar_count": stats.get("sample_size", 0),
        "similar_lines": similar_lines,
        "manual_line_id": drawing.manual_line_id,
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _count_line_touches(
    pdf,
    a1_idx: int,
    a2_idx: int,
    price1: float,
    price2: float,
    *,
    atr_pct: float,
    side: str,
    min_gap_bars: int = 3,
) -> dict:
    """Count how many distinct candle touches the line has between its anchors.

    A "touch" means the candle's wick reached the projected line price within
    `atr_pct × 0.5` percent. Consecutive touches within `min_gap_bars` bars
    collapse into one (avoids counting flat consolidations as N touches).

    Returns:
        {
          "touch_count": int,   # >= 2 (the two anchors always count)
          "strength": float,    # 0..1 — how reliable this line looks
        }
    """
    try:
        import numpy as _np
        if a2_idx <= a1_idx or a1_idx < 0:
            return {"touch_count": 2, "strength": 0.5}
        span = a2_idx - a1_idx
        if span <= 0:
            return {"touch_count": 2, "strength": 0.5}
        slope = (price2 - price1) / span
        highs = pdf["high"].values
        lows = pdf["low"].values
        # tolerance: half of one bar ATR
        tol_pct = max(atr_pct * 0.5, 0.05)
        # Count touches in the anchor window (inclusive)
        touches = []
        for i in range(a1_idx, a2_idx + 1):
            line_p = price1 + slope * (i - a1_idx)
            tol = line_p * tol_pct / 100.0
            if side == "support":
                # support = wick low touches the line from above
                if abs(lows[i] - line_p) <= tol or (lows[i] <= line_p <= highs[i]):
                    touches.append(i)
            else:
                if abs(highs[i] - line_p) <= tol or (lows[i] <= line_p <= highs[i]):
                    touches.append(i)
        # Collapse consecutive touches
        distinct = []
        for t in touches:
            if not distinct or (t - distinct[-1]) >= min_gap_bars:
                distinct.append(t)
        n_touches = max(len(distinct), 2)
        # Strength: 2 touches = 0.5 baseline, 3 = 0.7, 4 = 0.85, 5+ = 1.0
        strength_map = {2: 0.5, 3: 0.7, 4: 0.85}
        strength = strength_map.get(n_touches, 1.0)
        return {"touch_count": n_touches, "strength": strength}
    except Exception:
        return {"touch_count": 2, "strength": 0.5}


def _derive_recommendations(
    stats: dict,
    top_matches: list,
    vol_ctx: dict | None = None,
    *,
    line_quality: dict | None = None,
    distance_ctx: dict | None = None,
    coherence: dict | None = None,
) -> dict:
    """Turn raw pattern stats + raw outcome distribution into actionable
    order parameters. Uses percentiles when matches are available, falls
    back to averages when not.
    """
    n = stats.get("sample_size", 0) or 0
    p_bounce = float(stats.get("p_bounce", 0) or 0)
    avg_ret_atr = float(stats.get("avg_return_atr", 0) or 0)
    avg_dd_atr = float(stats.get("avg_drawdown_atr", 0) or 0)
    confidence = float(stats.get("confidence", 0) or 0)

    if n == 0:
        return _apply_smart_adjustments(
            _default_recommendations(), vol_ctx, line_quality, distance_ctx, coherence,
        )

    # Pull raw drawdown / return distributions from top_matches
    drawdowns: list[float] = []
    returns: list[float] = []
    for m in top_matches:
        try:
            rec = m.get("record") if isinstance(m, dict) else None
            if not rec:
                continue
            outcome = rec.get("outcome") or {}
            drawdowns.append(float(outcome.get("max_drawdown_atr", 0) or 0))
            returns.append(float(outcome.get("max_return_atr", 0) or 0))
        except Exception:
            continue

    # Distribution-based stop: use 80th percentile of historical drawdowns
    # (covers 80% of cases, but not the 20% blowouts that we'd never want
    # to risk against). Convert ATR to % using 0.5% as 1-ATR proxy.
    if drawdowns:
        dd80 = _percentile(drawdowns, 0.80)
        stop_atr = max(dd80, avg_dd_atr * 0.6)
    else:
        stop_atr = avg_dd_atr * 0.5
    stop_pct = round(max(0.2, min(1.5, stop_atr * 0.5)), 2)

    # Distribution-based TP: 60th percentile of historical max returns
    if returns:
        ret60 = _percentile(returns, 0.60)
        target_atr = max(ret60, avg_ret_atr * 0.7)
    else:
        target_atr = avg_ret_atr
    rr_raw = (target_atr / max(stop_atr, 0.5)) if stop_atr > 0 else 2.0
    rr_target = round(max(1.5, min(8.0, rr_raw)), 1)

    # Tolerance: tighter when confidence is high
    base_tol = 0.15 if confidence >= 0.5 else 0.25
    if p_bounce >= 0.65:
        base_tol *= 0.7

    # ── ATR floor: tolerance must be ≥ 0.4 × bar ATR%, stop ≥ 1.2 × bar ATR%
    # Otherwise wick noise insta-triggers and insta-stops in volatile regimes.
    atr_pct = float((vol_ctx or {}).get("atr_pct") or 0.0)
    regime = (vol_ctx or {}).get("regime") or "normal"
    if atr_pct > 0:
        tol_floor = atr_pct * 0.4
        stop_floor = atr_pct * 1.2
        base_tol = max(base_tol, tol_floor)
        stop_pct = max(stop_pct, round(stop_floor, 2))

    # ── P1: multi-touch line strength: stronger lines → tighter tol, higher RR ──
    line_strength = float((line_quality or {}).get("strength") or 0.5)
    touch_count = int((line_quality or {}).get("touch_count") or 2)
    if touch_count >= 4:
        base_tol *= 0.8         # very strong line → tighter trigger
        rr_target = round(min(8.0, rr_target * 1.15), 1)
    elif touch_count == 3:
        base_tol *= 0.9
        rr_target = round(min(8.0, rr_target * 1.05), 1)
    # 2 touches = no adjustment

    # ── P1: distance-adaptive tolerance ──
    dist_zone = (distance_ctx or {}).get("zone", "medium")
    if dist_zone == "far":
        base_tol *= 1.4
    elif dist_zone == "near":
        base_tol *= 0.85

    # ── P2: multi-TF coherence — aligned with higher-TF zones boosts confidence
    coh_score = int((coherence or {}).get("score") or 0)
    if coh_score >= 2:
        base_tol *= 0.85
        rr_target = round(min(8.0, rr_target * 1.2), 1)
    elif coh_score == 1:
        base_tol *= 0.92
        rr_target = round(min(8.0, rr_target * 1.1), 1)

    # ── P3: BB squeeze — imminent move, slightly wider stop + RR boost
    bb_state = (vol_ctx or {}).get("bb_state", "normal")
    if bb_state == "squeeze":
        stop_pct = round(stop_pct * 1.15, 2)
        rr_target = round(min(8.0, rr_target * 1.1), 1)

    tolerance_pct = round(max(0.05, min(1.5, base_tol)), 2)
    stop_pct = round(max(0.2, min(5.0, stop_pct)), 2)
    # Re-derive RR if stop got widened by ATR floor
    if atr_pct > 0 and stop_atr > 0:
        rr_raw = (target_atr / max(stop_atr, atr_pct * 0.6 / 0.5))
        rr_target = round(max(1.5, min(8.0, rr_raw)), 1)

    return {
        "tolerance_pct": tolerance_pct,
        "stop_pct": tolerance_pct,
        "rr_target": rr_target,
        "confidence_label": stats.get("trustworthiness", "low"),
        "atr_pct": round(atr_pct, 3),
        "regime": regime,
        "touch_count": touch_count,
        "line_strength": line_strength,
        "distance_zone": dist_zone,
        "coherence_score": coh_score,
        "bb_state": bb_state,
        "rationale": (
            f"{n} 条相似形态: 反弹 {int(p_bounce*100)}% · "
            f"DD80% {stop_atr:.1f}ATR · SL=line · "
            f"目标 {target_atr:.1f}ATR -> RR {rr_target}"
            + (f" · ATR={atr_pct:.2f}% ({regime})" if atr_pct > 0 else "")
            + (f" · {touch_count}触" if touch_count > 2 else "")
            + (f" · 距线{dist_zone}" if dist_zone != "medium" else "")
            + (f" · {coh_score}TF对齐" if coh_score > 0 else "")
            + (" · BB挤压" if bb_state == "squeeze" else (" · BB扩张" if bb_state == "expansion" else ""))
        ),
    }


def _default_recommendations() -> dict:
    return {
        "tolerance_pct": 0.2,
        "stop_pct": 0.2,
        "rr_target": 2.0,
        "confidence_label": "none",
        "rationale": "无历史样本 — 使用保守默认值",
    }


def _apply_smart_adjustments(
    recs: dict,
    vol_ctx: dict | None,
    line_quality: dict | None = None,
    distance_ctx: dict | None = None,
    coherence: dict | None = None,
) -> dict:
    """Apply ATR floor + touch-strength + distance-zone adjustments.

    Used for the n==0 (no historical patterns) recommendation path so that
    even fresh lines benefit from volatility/touch/distance awareness — the
    full _derive_recommendations body applies the same logic for n>0.
    """
    out = dict(recs)
    atr_pct = float((vol_ctx or {}).get("atr_pct") or 0.0)
    regime = (vol_ctx or {}).get("regime", "normal")

    # ATR floor
    if atr_pct > 0:
        tol_floor = atr_pct * 0.4
        stop_floor = atr_pct * 1.2
        out["tolerance_pct"] = max(out.get("tolerance_pct", 0.2), tol_floor)
        out["stop_pct"] = max(out.get("stop_pct", 0.5), stop_floor)

    # Touch strength
    touch_count = int((line_quality or {}).get("touch_count") or 2)
    if touch_count >= 4:
        out["tolerance_pct"] *= 0.8
        out["rr_target"] = round(min(8.0, out.get("rr_target", 2.0) * 1.15), 1)
    elif touch_count == 3:
        out["tolerance_pct"] *= 0.9
        out["rr_target"] = round(min(8.0, out.get("rr_target", 2.0) * 1.05), 1)

    # Distance zone
    dist_zone = (distance_ctx or {}).get("zone", "medium")
    if dist_zone == "far":
        out["tolerance_pct"] *= 1.4
    elif dist_zone == "near":
        out["tolerance_pct"] *= 0.85

    # Multi-TF coherence: aligned with higher-TF zones → tighter, more confident
    coh_score = int((coherence or {}).get("score") or 0)
    if coh_score >= 2:
        out["tolerance_pct"] *= 0.85
        out["rr_target"] = round(min(8.0, out.get("rr_target", 2.0) * 1.2), 1)
    elif coh_score == 1:
        out["tolerance_pct"] *= 0.92
        out["rr_target"] = round(min(8.0, out.get("rr_target", 2.0) * 1.1), 1)

    # P3: BB squeeze regime — imminent breakout expected, slightly widen stop
    bb_state = (vol_ctx or {}).get("bb_state", "normal")
    if bb_state == "squeeze":
        out["stop_pct"] *= 1.15
        out["rr_target"] = round(min(8.0, out.get("rr_target", 2.0) * 1.1), 1)

    # Final clamp + rounding
    out["tolerance_pct"] = round(max(0.05, min(1.5, out["tolerance_pct"])), 2)
    out["stop_pct"] = out["tolerance_pct"]
    out["atr_pct"] = round(atr_pct, 3)
    out["regime"] = regime
    out["touch_count"] = touch_count
    out["distance_zone"] = dist_zone
    out["coherence_score"] = coh_score
    out["bb_state"] = bb_state

    parts = []
    if "rationale" in out:
        parts.append(out["rationale"])
    if atr_pct > 0:
        parts.append(f"ATR={atr_pct:.2f}% ({regime})")
    if touch_count > 2:
        parts.append(f"{touch_count}触")
    if dist_zone != "medium":
        parts.append(f"距线{dist_zone}")
    if coh_score > 0:
        parts.append(f"{coh_score}个高TF对齐")
    if bb_state == "squeeze":
        parts.append("BB挤压")
    elif bb_state == "expansion":
        parts.append("BB扩张")
    out["rationale"] = " · ".join(parts)
    return out


def _apply_atr_floor(recs: dict, vol_ctx: dict | None) -> dict:
    """Backward-compat shim — calls _apply_smart_adjustments with no line/distance."""
    return _apply_smart_adjustments(recs, vol_ctx, None, None)


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
