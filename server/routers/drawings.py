from __future__ import annotations

import asyncio
import json
import math
import time as _time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..data_service import get_ohlcv_with_df
from ..drawings.store import ManualTrendlineStore, now_ts
from ..drawings.types import ManualTrendline
from ..schemas.drawings import (
    ManualTrendlineClearResponse,
    ManualTrendlineCreateRequest,
    ManualTrendlineListResponse,
    ManualTrendlineModel,
    ManualTrendlineResponse,
    ManualTrendlineUpdateRequest,
)
from ..strategy import StrategyConfig, build_latest_snapshot
from ..strategy.display_filter import build_display_line_meta
from .strategy import DEFAULT_DAYS_BY_INTERVAL, _config_with_market_precision, _normalize_symbol

router = APIRouter(prefix="/api/drawings", tags=["drawings"])

store = ManualTrendlineStore()


@router.get("", response_model=ManualTrendlineListResponse)
async def api_list_drawings(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
    enrich: bool = Query(False, description="Run comparison vs auto-detected lines (slow, opt-in)"),
):
    """List manual drawings. By default we DO NOT enrich with auto-line
    comparison — enrichment triggers a 730d OHLCV fetch from Bitget per
    call, which deadlocks the event loop under normal frontend polling.
    The UI doesn't use `comparison_status` / `nearest_auto_line_id`
    fields anymore, so the extra work is pure waste.
    Pass ?enrich=true only when you actually need those fields.
    """
    normalized_symbol = _normalize_symbol(symbol)
    drawings = store.list(symbol=normalized_symbol, timeframe=timeframe)
    if enrich:
        drawings = await _enrich_drawings(drawings, normalized_symbol, timeframe)
    return {"drawings": [ManualTrendlineModel.model_validate(item.to_dict()) for item in drawings]}


@router.get("/all", response_model=ManualTrendlineListResponse)
async def api_list_all_drawings():
    """List ALL manual drawings across every symbol/TF.

    Used by the sidebar "我的手画线" panel to show a grouped view: user
    picks a coin, expands it, sees all their drawings on that coin,
    clicks one to jump the chart to that symbol+tf. Never enriches —
    the grouped view doesn't use comparison data.
    """
    drawings = store.list()
    return {"drawings": [ManualTrendlineModel.model_validate(item.to_dict()) for item in drawings]}


@router.post("", response_model=ManualTrendlineResponse)
async def api_create_drawing(req: ManualTrendlineCreateRequest):
    """Create a manual drawing. FAST PATH — no enrichment.

    Previously this called `_enrich_drawing` which loads full OHLCV +
    builds a strategy snapshot + ranks against candidate auto-lines.
    That took 1-8 seconds, blocking the user's draw gesture and making
    the line appear to "disappear" after release. Enrichment now happens
    LAZILY in the GET endpoint instead — list_drawings calls
    `_enrich_drawings` which is batched per symbol/tf and cached.
    """
    normalized_symbol = _normalize_symbol(req.symbol)
    start, end = _ordered_line_points(req.t_start, req.price_start, req.t_end, req.price_end)
    created = ManualTrendline(
        manual_line_id=_manual_line_id(normalized_symbol, req.timeframe, req.side, req.t_start, req.t_end),
        symbol=normalized_symbol,
        timeframe=req.timeframe,
        side=req.side,
        source="manual",
        t_start=start[0],
        t_end=end[0],
        price_start=start[1],
        price_end=end[1],
        extend_left=bool(req.extend_left),
        extend_right=bool(req.extend_right),
        locked=bool(req.locked),
        label=req.label.strip(),
        notes=req.notes.strip(),
        comparison_status="uncompared",
        override_mode=req.override_mode,
        nearest_auto_line_id=None,
        slope_diff=None,
        projected_price_diff=None,
        overlap_ratio=None,
        created_at=now_ts(),
        updated_at=now_ts(),
        line_width=_clamp_line_width(req.line_width),
    )
    store.upsert(created)

    _schedule_drawing_ml_capture(created, stage="created")

    return {"drawing": ManualTrendlineModel.model_validate(created.to_dict())}


@router.patch("/{manual_line_id}", response_model=ManualTrendlineResponse)
async def api_update_drawing(manual_line_id: str, req: ManualTrendlineUpdateRequest):
    existing = store.get(manual_line_id)
    if existing is None:
        raise HTTPException(404, f"Unknown manual_line_id: {manual_line_id}")

    raw_t_start = int(req.t_start) if req.t_start is not None else existing.t_start
    raw_t_end = int(req.t_end) if req.t_end is not None else existing.t_end
    raw_price_start = float(req.price_start) if req.price_start is not None else existing.price_start
    raw_price_end = float(req.price_end) if req.price_end is not None else existing.price_end
    start, end = _ordered_line_points(raw_t_start, raw_price_start, raw_t_end, raw_price_end)
    updated = replace(
        existing,
        t_start=start[0],
        t_end=end[0],
        price_start=start[1],
        price_end=end[1],
        extend_left=bool(req.extend_left) if req.extend_left is not None else existing.extend_left,
        extend_right=bool(req.extend_right) if req.extend_right is not None else existing.extend_right,
        locked=bool(req.locked) if req.locked is not None else existing.locked,
        label=req.label.strip() if req.label is not None else existing.label,
        notes=req.notes.strip() if req.notes is not None else existing.notes,
        override_mode=req.override_mode if req.override_mode is not None else existing.override_mode,
        line_width=_clamp_line_width(req.line_width) if req.line_width is not None else existing.line_width,
        updated_at=now_ts(),
    )
    # Skip enrichment on PATCH — same reason as GET list, it pulls 730d
    # of Bitget data and blocks the event loop.
    drawing = updated
    store.upsert(drawing)
    # If the anchors moved (drag), kick any triggered conditionals on
    # this line to replan immediately against the new geometry instead
    # of waiting the usual 15m/1h bar interval.
    anchors_moved = (
        req.t_start is not None or req.t_end is not None
        or req.price_start is not None or req.price_end is not None
    )
    if anchors_moved:
        try:
            from ..conditionals.watcher import force_replan_line
            force_replan_line(drawing.manual_line_id)
        except Exception as exc:
            print(f"[drawings.patch] force_replan err: {exc}", flush=True)
        # Log the adjustment with before/after + affected orders so we
        # can later answer "why did I move the line and what changed?"
        # User 2026-04-21: "记录下为什么调整, 调整后发生了什么变化".
        try:
            asyncio.create_task(_log_line_adjustment(existing, drawing, req))
        except Exception as exc:
            print(f"[drawings.patch] adjustment log err: {exc}", flush=True)
    _schedule_drawing_ml_capture(drawing, stage="updated")

    return {"drawing": ManualTrendlineModel.model_validate(drawing.to_dict())}


_USER_LABELS_FILE = Path("data") / "user_drawing_labels.jsonl"


class _LabelReq(BaseModel):
    quality: str   # "good" | "bad" | "mediocre"
    pattern_type: str | None = None      # e.g. "ascending_triangle", "head_shoulders"
    notes: str | None = None             # free-form user note
    tags: list[str] | None = None


@router.post("/{manual_line_id}/label")
async def api_label_drawing(manual_line_id: str, req: _LabelReq):
    """Mark a drawing as good/bad pattern, with optional type + notes.

    User 2026-04-22: "这条线形态不错,告诉系统采纳这个特征". Writes a
    rich label record that ML training can use as positive/negative
    samples. Also captures the current market state as features so
    the model learns the CONTEXT where the pattern looked good.
    """
    drawing = store.get(manual_line_id)
    if drawing is None:
        raise HTTPException(404, f"Unknown manual_line_id: {manual_line_id}")

    quality = (req.quality or "").lower()
    if quality not in ("good", "bad", "mediocre"):
        raise HTTPException(400, f"quality must be good/bad/mediocre, got {quality!r}")

    # Capture market context at label time (best-effort, non-fatal)
    features: dict = {}
    try:
        from ..conditionals.watcher import _fetch_market_price, _fetch_current_atr
        mark = await _fetch_market_price(drawing.symbol) or 0.0
        atr = await _fetch_current_atr(drawing.symbol, drawing.timeframe) or 0.0
        if mark > 0:
            features["mark_price"] = mark
            features["atr_abs"] = atr
            features["atr_pct"] = (atr / mark * 100) if mark > 0 else 0
            # Distance of current price from the drawing's projected line
            import math
            span = max(1, drawing.t_end - drawing.t_start)
            now_ts_ = int(_time.time())
            ratio = (now_ts_ - drawing.t_start) / span
            ratio = max(-2.0, min(3.0, ratio))  # clamp absurd extrapolations
            try:
                if drawing.price_start > 0 and drawing.price_end > 0:
                    line_now = math.exp(
                        math.log(drawing.price_start)
                        + ratio * (math.log(drawing.price_end) - math.log(drawing.price_start))
                    )
                else:
                    slope = (drawing.price_end - drawing.price_start) / span
                    line_now = drawing.price_start + slope * (now_ts_ - drawing.t_start)
                features["line_now"] = line_now
                features["price_vs_line_pct"] = ((mark - line_now) / line_now * 100) if line_now > 0 else 0
            except Exception:
                pass
    except Exception:
        pass

    record = {
        "event": "user_labeled_drawing",
        "ts": int(_time.time()),
        "ts_iso": pd.Timestamp.utcnow().isoformat(),
        "manual_line_id": manual_line_id,
        "symbol": drawing.symbol,
        "timeframe": drawing.timeframe,
        "side": drawing.side,
        "t_start": drawing.t_start,
        "t_end": drawing.t_end,
        "price_start": drawing.price_start,
        "price_end": drawing.price_end,
        "extend_left": drawing.extend_left,
        "extend_right": drawing.extend_right,
        "quality": quality,
        "pattern_type": req.pattern_type,
        "user_notes": req.notes,
        "tags": req.tags or [],
        "features_at_label": features,
    }
    _USER_LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_USER_LABELS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    return {"ok": True, "label": record}


@router.delete("/{manual_line_id}", response_model=ManualTrendlineClearResponse)
async def api_delete_drawing(manual_line_id: str):
    # Capture the line + current market features BEFORE removing it so we
    # get the negative-signal training row ("user removed this line"). This
    # is the only place 'deleted' events enter user_drawings_ml.jsonl.
    existing = store.get(manual_line_id)
    if existing is not None:
        _schedule_drawing_ml_capture(existing, stage="deleted")
    removed = 1 if store.delete(manual_line_id) else 0
    if removed == 0:
        raise HTTPException(404, f"Unknown manual_line_id: {manual_line_id}")
    # Cascade: cancel all pending/triggered conditionals for this line.
    # User policy: deleting a line means "I'm done with this edge" —
    # any outstanding orders on it must be cancelled at the same moment.
    try:
        from ..conditionals import ConditionalOrderStore, ConditionalEvent, now_ts as _nt
        cstore = ConditionalOrderStore()
        for cond in cstore.list_all(manual_line_id=manual_line_id):
            if cond.status in ("pending", "triggered"):
                cstore.set_status(
                    cond.conditional_id,
                    "cancelled",
                    reason=f"line deleted: {manual_line_id}",
                )
                try:
                    cstore.append_event(cond.conditional_id, ConditionalEvent(
                        ts=_nt(), kind="cancelled",
                        message=f"parent line {manual_line_id} was deleted",
                    ))
                except Exception:
                    pass
    except Exception as exc:
        # Don't block the delete on cascade errors
        print(f"[drawings.delete] cascade cancel failed: {exc}", flush=True)
    return {"removed": removed}


@router.post("/clear", response_model=ManualTrendlineClearResponse)
async def api_clear_drawings(
    symbol: str | None = Query(None),
    timeframe: str | None = Query(None),
):
    normalized_symbol = _normalize_symbol(symbol) if symbol else None
    targets = store.list(symbol=normalized_symbol, timeframe=timeframe)
    removed = store.clear(symbol=normalized_symbol, timeframe=timeframe)
    if removed:
        _cancel_conditionals_for_lines(
            [item.manual_line_id for item in targets],
            reason_prefix="lines cleared",
        )
    return {"removed": removed}


def _cancel_conditionals_for_lines(line_ids: list[str], *, reason_prefix: str) -> int:
    """Mark pending/triggered trade plans cancelled when parent drawings go away."""
    if not line_ids:
        return 0
    cancelled = 0
    try:
        from ..conditionals import ConditionalOrderStore, ConditionalEvent, now_ts as _nt

        cstore = ConditionalOrderStore()
        for manual_line_id in line_ids:
            for cond in cstore.list_all(manual_line_id=manual_line_id):
                if cond.status not in ("pending", "triggered"):
                    continue
                cstore.set_status(
                    cond.conditional_id,
                    "cancelled",
                    reason=f"{reason_prefix}: {manual_line_id}",
                )
                cancelled += 1
                try:
                    cstore.append_event(cond.conditional_id, ConditionalEvent(
                        ts=_nt(), kind="cancelled",
                        message=f"parent line {manual_line_id} was removed",
                    ))
                except Exception:
                    pass
    except Exception as exc:
        # Do not block drawing deletion on cascade bookkeeping errors.
        print(f"[drawings.clear] cascade cancel failed: {exc}", flush=True)
    return cancelled


_LINE_ADJUSTMENT_LOG = Path("data") / "line_adjustments.jsonl"


def _log_slope_pct_per_hour(p_start: float, p_end: float, t_start: int, t_end: int) -> float:
    """% per hour (CAGR) along the line in log space."""
    span = t_end - t_start
    if span <= 0 or p_start <= 0 or p_end <= 0:
        return 0.0
    total = math.log(p_end) - math.log(p_start)
    hours = span / 3600.0
    return (math.exp(total / hours) - 1.0) * 100.0


async def _log_line_adjustment(before: ManualTrendline, after: ManualTrendline,
                               req: ManualTrendlineUpdateRequest) -> None:
    """Append a rich record of each line adjustment.

    Captures everything needed to later answer:
      - What moved (before → after, deltas)
      - Which conditionals were affected
      - Market context (current price, ATR, MA state)
      - Whether force_replan succeeded (watcher logs the replan event
        separately — we link by conditional_id)

    Output: data/line_adjustments.jsonl (append-only)
    Each line = 1 JSON record per PATCH with anchor changes.
    """
    from ..conditionals.store import ConditionalOrderStore
    try:
        cond_store = ConditionalOrderStore()
        # Related conditionals: any active (pending / triggered / filled)
        affected: list[dict] = []
        for c in cond_store.list_all(manual_line_id=after.manual_line_id):
            if c.status not in ("pending", "triggered", "filled"):
                continue
            affected.append({
                "conditional_id": c.conditional_id,
                "status": c.status,
                "direction": (c.order.direction if c.order else None),
                "exchange_order_id": c.exchange_order_id,
                "fill_price_at_adjust": c.fill_price,
                "tolerance_pct": c.order.tolerance_pct_of_line if c.order else None,
                "stop_offset_pct": c.order.stop_offset_pct_of_line if c.order else None,
            })

        # Market context
        mark_price = None
        atr_pct = None
        try:
            from ..conditionals.watcher import _fetch_market_price, _fetch_current_atr
            mark_price = await _fetch_market_price(after.symbol)
            atr_raw = await _fetch_current_atr(after.symbol, after.timeframe)
            if mark_price and atr_raw:
                atr_pct = (atr_raw / mark_price) * 100.0
        except Exception:
            pass

        # Deltas
        dt_start = int(after.t_start) - int(before.t_start)
        dt_end = int(after.t_end) - int(before.t_end)
        dp_start = float(after.price_start) - float(before.price_start)
        dp_end = float(after.price_end) - float(before.price_end)
        old_slope = _log_slope_pct_per_hour(before.price_start, before.price_end,
                                            before.t_start, before.t_end)
        new_slope = _log_slope_pct_per_hour(after.price_start, after.price_end,
                                            after.t_start, after.t_end)

        record = {
            "event": "line_adjusted",
            "ts": int(_time.time()),
            "ts_iso": pd.Timestamp.utcnow().isoformat(),
            "manual_line_id": after.manual_line_id,
            "symbol": after.symbol,
            "timeframe": after.timeframe,
            "side": after.side,
            "before": {
                "t_start": before.t_start, "t_end": before.t_end,
                "price_start": before.price_start, "price_end": before.price_end,
                "extend_left": before.extend_left, "extend_right": before.extend_right,
                "slope_pct_per_hour_log": round(old_slope, 4),
            },
            "after": {
                "t_start": after.t_start, "t_end": after.t_end,
                "price_start": after.price_start, "price_end": after.price_end,
                "extend_left": after.extend_left, "extend_right": after.extend_right,
                "slope_pct_per_hour_log": round(new_slope, 4),
            },
            "delta": {
                "t_start_sec": dt_start,
                "t_end_sec": dt_end,
                "price_start_abs": dp_start,
                "price_end_abs": dp_end,
                "price_start_pct": (dp_start / before.price_start * 100) if before.price_start else 0,
                "price_end_pct": (dp_end / before.price_end * 100) if before.price_end else 0,
                "slope_pct_per_hour_delta": round(new_slope - old_slope, 4),
            },
            "market_context": {
                "mark_price": mark_price,
                "atr_pct": round(atr_pct, 3) if atr_pct else None,
            },
            "affected_conditionals": affected,
            "affected_count": len(affected),
            # User optional reason — they can later tag adjustments via
            # a note field. For now, accept what's on the request body
            # (notes) so the user can pass "reason: MA structure broke".
            "user_reason": req.notes if (req.notes and req.notes.strip()) else None,
        }
        _LINE_ADJUSTMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_LINE_ADJUSTMENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[line_adjustment_log] err: {exc}", flush=True)


def _schedule_drawing_ml_capture(drawing: ManualTrendline, *, stage: str) -> None:
    reason = {"deleted": "user_delete", "updated": "user_update"}.get(stage)
    try:
        asyncio.create_task(_capture_drawing_for_ml(drawing, stage=stage, reason=reason))
    except RuntimeError:
        # No running loop in unusual test contexts; fall back to the basic
        # record so the drawing action is still captured.
        try:
            from ..strategy.drawing_learner import capture_user_drawing
            capture_user_drawing(
                manual_line_id=drawing.manual_line_id,
                symbol=drawing.symbol,
                timeframe=drawing.timeframe,
                side=drawing.side,
                price_start=float(drawing.price_start),
                price_end=float(drawing.price_end),
                t_start=drawing.t_start,
                t_end=drawing.t_end,
                capture_stage=f"{stage}_basic",
                reason=reason,
            )
        except Exception as exc:
            print(f"[drawing_learner] fallback capture err: {exc}", flush=True)


async def _capture_drawing_for_ml(drawing: ManualTrendline, *, stage: str, reason: str | None = None) -> None:
    try:
        from ..strategy.drawing_learner import capture_user_drawing

        df = None
        htf_df = None
        try:
            polars_df, _ = await get_ohlcv_with_df(
                drawing.symbol,
                drawing.timeframe,
                None,
                days=14,
                history_mode="fast_window",
                include_price_precision=False,
                include_render_payload=False,
            )
            if polars_df is not None and not polars_df.is_empty():
                df = _standardize_strategy_candles(polars_df)
        except Exception as exc:
            print(f"[drawing_learner] OHLCV feature load err: {exc}", flush=True)
        htf = _higher_timeframe(drawing.timeframe)
        if htf:
            try:
                htf_polars, _ = await get_ohlcv_with_df(
                    drawing.symbol,
                    htf,
                    None,
                    days=60,
                    history_mode="fast_window",
                    include_price_precision=False,
                    include_render_payload=False,
                )
                if htf_polars is not None and not htf_polars.is_empty():
                    htf_df = _standardize_strategy_candles(htf_polars)
            except Exception as exc:
                print(f"[drawing_learner] HTF feature load err: {exc}", flush=True)

        rec = capture_user_drawing(
            manual_line_id=drawing.manual_line_id,
            symbol=drawing.symbol,
            timeframe=drawing.timeframe,
            side=drawing.side,
            price_start=float(drawing.price_start),
            price_end=float(drawing.price_end),
            t_start=drawing.t_start,
            t_end=drawing.t_end,
            df=df,
            htf_df=htf_df,
            capture_stage=stage if df is not None else f"{stage}_basic",
            reason=reason,
        )
        feature_count = len((rec or {}).get("features") or {})
        print(
            f"[drawing_learner] captured {drawing.manual_line_id} "
            f"stage={stage} features={feature_count}",
            flush=True,
        )
    except Exception as exc:
        print(f"[drawing_learner] capture err: {exc}", flush=True)


async def _enrich_drawings(
    drawings: list[ManualTrendline],
    symbol: str,
    timeframe: str,
) -> list[ManualTrendline]:
    if not drawings:
        return []
    snapshot, candles_df, cfg = await _load_auto_snapshot(symbol, timeframe)
    return [_compare_manual_to_auto(drawing, snapshot, candles_df, cfg) for drawing in drawings]


async def _enrich_drawing(drawing: ManualTrendline) -> ManualTrendline:
    try:
        snapshot, candles_df, cfg = await _load_auto_snapshot(drawing.symbol, drawing.timeframe)
    except Exception:
        return drawing
    return _compare_manual_to_auto(drawing, snapshot, candles_df, cfg)


async def _load_auto_snapshot(symbol: str, timeframe: str):
    polars_df, payload = await get_ohlcv_with_df(
        symbol,
        timeframe,
        None,
        DEFAULT_DAYS_BY_INTERVAL.get(timeframe, 365),
        include_price_precision=True,
        include_render_payload=False,
    )
    if polars_df is None or polars_df.is_empty():
        raise ValueError(f"No data for {symbol} {timeframe}")
    candles_df = _standardize_strategy_candles(polars_df)
    if len(candles_df) > 500:
        candles_df = candles_df.iloc[-500:].reset_index(drop=True)
    price_precision = payload.get("pricePrecision") if isinstance(payload, dict) else None
    cfg = _config_with_market_precision(StrategyConfig(), price_precision)
    snapshot = build_latest_snapshot(candles_df, cfg, symbol=symbol, timeframe=timeframe)
    return snapshot, candles_df, cfg


def _standardize_strategy_candles(polars_df) -> pd.DataFrame:
    pdf = polars_df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    pdf["timestamp"] = pdf["timestamp"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    for column in ("open", "high", "low", "close", "volume"):
        pdf[column] = pd.to_numeric(pdf[column], errors="raise")
    return pdf[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _compare_manual_to_auto(
    drawing: ManualTrendline,
    snapshot,
    candles_df: pd.DataFrame,
    cfg: StrategyConfig,
) -> ManualTrendline:
    if candles_df.empty:
        return drawing

    display_meta = build_display_line_meta(candles_df, snapshot.candidate_lines, config=cfg)
    candidate_lines = [
        line
        for line in snapshot.candidate_lines
        if line.side == drawing.side and display_meta.get(line.line_id) is not None and display_meta[line.line_id].display_class != "debug"
    ]
    if not candidate_lines:
        return replace(
            drawing,
            comparison_status="no_nearby_auto",
            nearest_auto_line_id=None,
            slope_diff=None,
            projected_price_diff=None,
            overlap_ratio=None,
        )

    manual_start_idx = _nearest_bar_index(candles_df, drawing.t_start)
    manual_end_idx = _nearest_bar_index(candles_df, drawing.t_end)
    if manual_end_idx == manual_start_idx:
        manual_end_idx = min(len(candles_df) - 1, manual_start_idx + 1)
    manual_slope = (drawing.price_end - drawing.price_start) / max(manual_end_idx - manual_start_idx, 1)
    current_index = len(candles_df) - 1
    manual_projected_current = drawing.price_start + (manual_slope * (current_index - manual_start_idx))
    atr = float(cfg.tolerance(0.0, float(candles_df.iloc[current_index]["close"])))

    ranked: list[tuple[float, Any]] = []
    for line in candidate_lines:
        projected_diff = abs(line.projected_price_current - manual_projected_current)
        slope_diff = abs(line.slope - manual_slope)
        overlap_ratio = _overlap_ratio(manual_start_idx, manual_end_idx, line.anchor_indices[0], current_index)
        score = projected_diff + (slope_diff * 100.0) - overlap_ratio
        ranked.append((score, (line, slope_diff, projected_diff, overlap_ratio)))
    ranked.sort(key=lambda item: item[0])
    nearest_line, slope_diff, projected_diff, overlap_ratio = ranked[0][1]

    if slope_diff <= 0.01 and projected_diff <= max(atr * 2.0, cfg.tick_size * 5):
        comparison_status = "supports_auto"
    elif slope_diff <= 0.03 and projected_diff <= max(atr * 4.0, cfg.tick_size * 10):
        comparison_status = "near_auto"
    else:
        comparison_status = "conflicts_auto"

    return replace(
        drawing,
        comparison_status=comparison_status,
        nearest_auto_line_id=nearest_line.line_id,
        slope_diff=round(slope_diff, 6),
        projected_price_diff=round(projected_diff, 6),
        overlap_ratio=round(overlap_ratio, 4),
    )


def _overlap_ratio(left_start: int, left_end: int, right_start: int, right_end: int) -> float:
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    if union <= 0:
        return 0.0
    return overlap / union


def _nearest_bar_index(candles_df: pd.DataFrame, target_ts: int) -> int:
    timestamps = candles_df["timestamp"].tolist()
    best_index = 0
    best_distance = 10**18
    for index, timestamp in enumerate(timestamps):
        distance = abs(int(timestamp) - int(target_ts))
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _manual_line_id(symbol: str, timeframe: str, side: str, t_start: int, t_end: int) -> str:
    return f"manual-{symbol}-{timeframe}-{side}-{min(t_start, t_end)}-{max(t_start, t_end)}"


def _ordered_line_points(
    t_start: int | float,
    price_start: int | float,
    t_end: int | float,
    price_end: int | float,
) -> tuple[tuple[int, float], tuple[int, float]]:
    left = (int(t_start), float(price_start))
    right = (int(t_end), float(price_end))
    if right[0] < left[0]:
        return right, left
    return left, right


def _higher_timeframe(timeframe: str) -> str | None:
    return {
        "1m": "5m",
        "5m": "15m",
        "15m": "1h",
        "1h": "4h",
        "4h": "1d",
    }.get(timeframe)


def _clamp_line_width(value: float | int | None) -> float:
    # Floor 0.2 so hair-thin lines (0.2 / 0.3) survive round-tripping.
    # Must match the Pydantic validator in schemas/drawings.py.
    try:
        width = float(value)
    except (TypeError, ValueError):
        width = 0.3
    return max(0.2, min(width, 8.0))


__all__ = ["router", "store"]
