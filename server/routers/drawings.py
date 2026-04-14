from __future__ import annotations

from dataclasses import replace
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
    created = ManualTrendline(
        manual_line_id=_manual_line_id(normalized_symbol, req.timeframe, req.side, req.t_start, req.t_end),
        symbol=normalized_symbol,
        timeframe=req.timeframe,
        side=req.side,
        source="manual",
        t_start=min(req.t_start, req.t_end),
        t_end=max(req.t_start, req.t_end),
        price_start=float(req.price_start),
        price_end=float(req.price_end),
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
    )
    store.upsert(created)
    return {"drawing": ManualTrendlineModel.model_validate(created.to_dict())}


@router.patch("/{manual_line_id}", response_model=ManualTrendlineResponse)
async def api_update_drawing(manual_line_id: str, req: ManualTrendlineUpdateRequest):
    existing = store.get(manual_line_id)
    if existing is None:
        raise HTTPException(404, f"Unknown manual_line_id: {manual_line_id}")

    updated = replace(
        existing,
        t_start=int(req.t_start) if req.t_start is not None else existing.t_start,
        t_end=int(req.t_end) if req.t_end is not None else existing.t_end,
        price_start=float(req.price_start) if req.price_start is not None else existing.price_start,
        price_end=float(req.price_end) if req.price_end is not None else existing.price_end,
        extend_left=bool(req.extend_left) if req.extend_left is not None else existing.extend_left,
        extend_right=bool(req.extend_right) if req.extend_right is not None else existing.extend_right,
        locked=bool(req.locked) if req.locked is not None else existing.locked,
        label=req.label.strip() if req.label is not None else existing.label,
        notes=req.notes.strip() if req.notes is not None else existing.notes,
        override_mode=req.override_mode if req.override_mode is not None else existing.override_mode,
        updated_at=now_ts(),
    )
    updated = replace(updated, t_start=min(updated.t_start, updated.t_end), t_end=max(updated.t_start, updated.t_end))
    # Skip enrichment on PATCH — same reason as GET list, it pulls 730d
    # of Bitget data and blocks the event loop.
    drawing = updated
    store.upsert(drawing)
    # If the anchors moved (drag), kick any triggered conditionals on
    # this line to replan immediately against the new geometry instead
    # of waiting the usual 15m/1h bar interval.
    if (
        req.t_start is not None or req.t_end is not None
        or req.price_start is not None or req.price_end is not None
    ):
        try:
            from ..conditionals.watcher import force_replan_line
            force_replan_line(drawing.manual_line_id)
        except Exception as exc:
            print(f"[drawings.patch] force_replan err: {exc}", flush=True)
    return {"drawing": ManualTrendlineModel.model_validate(drawing.to_dict())}


@router.delete("/{manual_line_id}", response_model=ManualTrendlineClearResponse)
async def api_delete_drawing(manual_line_id: str):
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
    removed = store.clear(symbol=normalized_symbol, timeframe=timeframe)
    return {"removed": removed}


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


__all__ = ["router", "store"]
