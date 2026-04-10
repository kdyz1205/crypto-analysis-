from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict, replace
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..data_service import get_ohlcv_with_df
from ..schemas.strategy import (
    StrategyConfigResponse,
    StrategyLineModel,
    StrategyLineStateModel,
    StrategyPivotModel,
    StrategyReplayResponse,
    StrategySignalModel,
    StrategySignalStateModel,
    StrategySnapshotModel,
    StrategySnapshotResponse,
    StrategyTouchPointModel,
    serialize_config_response,
)
from ..strategy import StrategyConfig, build_latest_snapshot, build_tail_snapshots, replay_strategy
from ..strategy.display_filter import (
    build_display_line_meta,
    collapse_display_invalidations,
    filter_display_touch_indices,
)

router = APIRouter(prefix="/api/strategy", tags=["strategy"])

VALID_INTERVALS = {"5m", "15m", "1h", "4h", "1d"}
DEFAULT_DAYS_BY_INTERVAL = {
    "5m": 7,
    "15m": 21,
    "1h": 120,
    "4h": 365,
    "1d": 365,
}
SNAPSHOT_CACHE_LIMIT = 32
REPLAY_CACHE_LIMIT = 16
FAST_REPLAY_TAIL_THRESHOLD = 12
_snapshot_cache: OrderedDict[tuple[Any, ...], StrategySnapshotResponse] = OrderedDict()
_replay_cache: OrderedDict[tuple[Any, ...], StrategyReplayResponse] = OrderedDict()


@router.get("/config", response_model=StrategyConfigResponse)
async def api_strategy_config(
    symbol: str | None = Query(None, description="Optional symbol for precision-aware config"),
    interval: str = Query("4h", description="5m, 15m, 1h, 4h, 1d"),
):
    if interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval. Must be one of: {sorted(VALID_INTERVALS)}")

    normalized_symbol = _normalize_symbol(symbol) if symbol else None
    cfg = StrategyConfig()
    price_precision: int | None = None

    if normalized_symbol:
        try:
            _, market_payload, price_precision, cfg = await _load_strategy_inputs(
                normalized_symbol,
                interval,
                end_time=None,
                days=DEFAULT_DAYS_BY_INTERVAL.get(interval, 120),
                analysis_bars=None,
            )
            if market_payload:
                price_precision = market_payload.get("pricePrecision", price_precision)
        except Exception:
            # Config should still be available even if market lookup fails.
            pass

    return serialize_config_response(
        cfg,
        symbol=normalized_symbol,
        interval=interval,
        price_precision=price_precision,
    )


@router.get("/snapshot", response_model=StrategySnapshotResponse)
async def api_strategy_snapshot(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("4h", description="5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int | None = Query(None, description="Days of data to load before analysis"),
    analysis_bars: int = Query(500, ge=120, le=2000, description="Max bars sent through replay/strategy core"),
):
    if interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval. Must be one of: {sorted(VALID_INTERVALS)}")

    normalized_symbol = _normalize_symbol(symbol)
    requested_days = days or DEFAULT_DAYS_BY_INTERVAL.get(interval, 120)

    try:
        candles_df, _, price_precision, cfg = await _load_strategy_inputs(
            normalized_symbol,
            interval,
            end_time=end_time,
            days=requested_days,
            analysis_bars=analysis_bars,
        )
        cache_key = _snapshot_cache_key(candles_df, normalized_symbol, interval, price_precision)
        cached = _get_cached_snapshot(cache_key)
        if cached is not None:
            return cached
        response = await asyncio.to_thread(
            _build_strategy_snapshot_response,
            candles_df,
            cfg,
            normalized_symbol,
            interval,
            price_precision,
        )
        return _store_cached_snapshot(cache_key, response)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@router.get("/replay", response_model=StrategyReplayResponse)
async def api_strategy_replay(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("4h", description="5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int | None = Query(None, description="Days of data to load before analysis"),
    analysis_bars: int = Query(500, ge=120, le=2000, description="Max bars sent through replay/strategy core"),
    tail: int | None = Query(None, ge=1, le=500, description="Optional number of latest snapshots to return"),
):
    if interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval. Must be one of: {sorted(VALID_INTERVALS)}")

    normalized_symbol = _normalize_symbol(symbol)
    requested_days = days or DEFAULT_DAYS_BY_INTERVAL.get(interval, 120)

    try:
        candles_df, _, price_precision, cfg = await _load_strategy_inputs(
            normalized_symbol,
            interval,
            end_time=end_time,
            days=requested_days,
            analysis_bars=analysis_bars,
        )
        cache_key = _replay_cache_key(candles_df, normalized_symbol, interval, price_precision, tail)
        cached = _get_cached_replay(cache_key)
        if cached is not None:
            return cached
        response = await asyncio.to_thread(
            _build_strategy_replay_response,
            candles_df,
            cfg,
            normalized_symbol,
            interval,
            price_precision,
            tail,
        )
        return _store_cached_replay(cache_key, response)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


async def _load_strategy_inputs(
    symbol: str,
    interval: str,
    *,
    end_time: str | None,
    days: int,
    analysis_bars: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], int | None, StrategyConfig]:
    polars_df, market_payload = await get_ohlcv_with_df(
        symbol,
        interval,
        end_time,
        days,
        include_price_precision=True,
        include_render_payload=False,
    )
    if polars_df is None or polars_df.is_empty():
        raise ValueError(f"No data for {symbol} {interval}")

    candles_df = _standardize_strategy_candles(polars_df)
    if analysis_bars is not None and len(candles_df) > analysis_bars:
        candles_df = candles_df.iloc[-analysis_bars:].reset_index(drop=True)

    price_precision = market_payload.get("pricePrecision") if isinstance(market_payload, dict) else None
    cfg = _config_with_market_precision(StrategyConfig(), price_precision)
    return candles_df, market_payload, price_precision, cfg


def _standardize_strategy_candles(polars_df) -> pd.DataFrame:
    pdf = polars_df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    pdf["timestamp"] = pdf["timestamp"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    for column in ("open", "high", "low", "close", "volume"):
        pdf[column] = pd.to_numeric(pdf[column], errors="raise")
    return pdf[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _config_with_market_precision(config: StrategyConfig, price_precision: int | None) -> StrategyConfig:
    if price_precision is None:
        return config
    tick_size = 1.0 if int(price_precision) <= 0 else float(10 ** (-int(price_precision)))
    return replace(config, tick_size=tick_size)


def _normalize_symbol(symbol: str | None) -> str:
    if not symbol:
        raise ValueError("symbol is required")
    normalized = symbol.upper().replace("/", "")
    return normalized if normalized.endswith("USDT") else f"{normalized}USDT"


def _build_strategy_snapshot_response(
    candles_df: pd.DataFrame,
    cfg: StrategyConfig,
    symbol: str,
    interval: str,
    price_precision: int | None,
) -> StrategySnapshotResponse:
    snapshot = build_latest_snapshot(candles_df, cfg, symbol=symbol, timeframe=interval)
    snapshot_payload = _serialize_snapshot(snapshot, candles_df)
    return StrategySnapshotResponse(
        symbol=symbol,
        interval=interval,
        barCount=int(len(candles_df)),
        analysisBarCount=int(len(candles_df)),
        pricePrecision=price_precision,
        tickSize=float(cfg.tick_size),
        config=asdict(cfg),
        snapshot=snapshot_payload,
    )


def _snapshot_cache_key(
    candles_df: pd.DataFrame,
    symbol: str,
    interval: str,
    price_precision: int | None,
) -> tuple[Any, ...]:
    return (
        symbol,
        interval,
        int(len(candles_df)),
        _recent_candle_signature(candles_df),
        price_precision,
    )


def _get_cached_snapshot(cache_key: tuple[Any, ...]) -> StrategySnapshotResponse | None:
    cached = _snapshot_cache.get(cache_key)
    if cached is None:
        return None
    _snapshot_cache.move_to_end(cache_key)
    return cached


def _store_cached_snapshot(
    cache_key: tuple[Any, ...],
    response: StrategySnapshotResponse,
) -> StrategySnapshotResponse:
    _snapshot_cache[cache_key] = response
    _snapshot_cache.move_to_end(cache_key)
    while len(_snapshot_cache) > SNAPSHOT_CACHE_LIMIT:
        _snapshot_cache.popitem(last=False)
    return response


def _replay_cache_key(
    candles_df: pd.DataFrame,
    symbol: str,
    interval: str,
    price_precision: int | None,
    tail: int | None,
) -> tuple[Any, ...]:
    return (
        symbol,
        interval,
        int(len(candles_df)),
        _recent_candle_signature(candles_df),
        price_precision,
        tail,
    )


def _get_cached_replay(cache_key: tuple[Any, ...]) -> StrategyReplayResponse | None:
    cached = _replay_cache.get(cache_key)
    if cached is None:
        return None
    _replay_cache.move_to_end(cache_key)
    return cached


def _store_cached_replay(
    cache_key: tuple[Any, ...],
    response: StrategyReplayResponse,
) -> StrategyReplayResponse:
    _replay_cache[cache_key] = response
    _replay_cache.move_to_end(cache_key)
    while len(_replay_cache) > REPLAY_CACHE_LIMIT:
        _replay_cache.popitem(last=False)
    return response


def _recent_candle_signature(candles_df: pd.DataFrame, *, window: int = 8) -> tuple[Any, ...]:
    tail = candles_df.tail(window)
    signature: list[Any] = []
    for row in tail.itertuples(index=False):
        signature.extend(
            (
                int(row.timestamp),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
            )
        )
    return tuple(signature)


def _build_strategy_replay_response(
    candles_df: pd.DataFrame,
    cfg: StrategyConfig,
    symbol: str,
    interval: str,
    price_precision: int | None,
    tail: int | None,
) -> StrategyReplayResponse:
    if tail is not None and tail <= FAST_REPLAY_TAIL_THRESHOLD:
        snapshots = build_tail_snapshots(
            candles_df,
            cfg,
            symbol=symbol,
            timeframe=interval,
            tail=tail,
        )
    else:
        replay_result = replay_strategy(candles_df, cfg, symbol=symbol, timeframe=interval)
        snapshots = replay_result.snapshots[-tail:] if tail else replay_result.snapshots
    timestamps = [int(value) for value in candles_df["timestamp"].tolist()]
    highs = [float(value) for value in candles_df["high"].tolist()]
    lows = [float(value) for value in candles_df["low"].tolist()]
    snapshot_payloads = [
        _serialize_snapshot(snapshot, candles_df, timestamps=timestamps, highs=highs, lows=lows)
        for snapshot in snapshots
    ]
    return StrategyReplayResponse(
        symbol=symbol,
        interval=interval,
        barCount=int(len(candles_df)),
        analysisBarCount=int(len(candles_df)),
        snapshotCount=len(snapshot_payloads),
        pricePrecision=price_precision,
        tickSize=float(cfg.tick_size),
        config=asdict(cfg),
        snapshots=snapshot_payloads,
    )


def _serialize_snapshot(
    snapshot,
    candles_df: pd.DataFrame,
    *,
    timestamps: list[int] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> StrategySnapshotModel:
    timestamps = timestamps or [int(value) for value in candles_df["timestamp"].tolist()]
    highs = highs or [float(value) for value in candles_df["high"].tolist()]
    lows = lows or [float(value) for value in candles_df["low"].tolist()]
    current_index = snapshot.bar_index
    current_time = timestamps[current_index]
    next_time = _next_timestamp(timestamps, current_index)
    line_state_map = {state.line_id: state for state in snapshot.line_states}
    active_line_ids = {state.line_id for state in snapshot.active_lines}
    display_meta = build_display_line_meta(candles_df, snapshot.candidate_lines)

    candidate_lines = [
        _serialize_line(
            line,
            line_state_map.get(line.line_id),
            display_meta=display_meta,
            timestamps=timestamps,
            current_time=current_time,
            next_time=next_time,
        )
        for line in snapshot.candidate_lines
    ]
    active_lines = [line for line in candidate_lines if line.line_id in active_line_ids]

    return StrategySnapshotModel(
        bar_index=snapshot.bar_index,
        timestamp=snapshot.timestamp,
        pivots=[StrategyPivotModel.model_validate(asdict(pivot)) for pivot in snapshot.pivots],
        candidate_lines=candidate_lines,
        active_lines=active_lines,
        line_states=[
            _serialize_line_state(state, snapshot.candidate_lines, display_meta=display_meta, timestamps=timestamps)
            for state in snapshot.line_states
        ],
        touch_points=_serialize_touch_points(
            snapshot.candidate_lines,
            display_meta=display_meta,
            timestamps=timestamps,
            highs=highs,
            lows=lows,
        ),
        signals=[StrategySignalModel.model_validate(asdict(signal)) for signal in snapshot.signals],
        signal_states=[
            StrategySignalStateModel.model_validate(asdict(signal_state))
            for signal_state in snapshot.signal_states
        ],
        invalidations=_serialize_invalidations(
            snapshot.candidate_lines,
            snapshot.invalidations,
            display_meta=display_meta,
            timestamps=timestamps,
        ),
        orders=[],
    )


def _serialize_line(
    line,
    line_state,
    *,
    display_meta,
    timestamps: list[int],
    current_time: int,
    next_time: int,
) -> StrategyLineModel:
    state = line_state.state if line_state is not None else line.state
    invalidation_reason = (
        line_state.invalidation_reason
        if line_state is not None and line_state.invalidation_reason is not None
        else line.invalidation_reason
    )
    meta = display_meta.get(line.line_id)
    invalidation_timestamp = (
        timestamps[line.invalidation_index]
        if line.invalidation_index is not None and 0 <= line.invalidation_index < len(timestamps)
        else None
    )
    return StrategyLineModel(
        line_id=line.line_id,
        symbol=line.symbol,
        timeframe=line.timeframe,
        side=line.side,
        state=state,
        t_start=timestamps[line.anchor_indices[0]],
        t_end=current_time,
        price_start=float(line.anchor_prices[0]),
        price_end=float(line.projected_price_current),
        slope=float(line.slope),
        intercept=float(line.intercept),
        anchor_indices=list(line.anchor_indices),
        anchor_prices=[float(price) for price in line.anchor_prices],
        anchor_timestamps=[timestamps[index] for index in line.anchor_indices],
        confirming_touch_indices=list(line.confirming_touch_indices),
        bar_touch_indices=list(line.bar_touch_indices),
        touch_count=int(line.confirming_touch_count),
        confirming_touch_count=int(line.confirming_touch_count),
        bar_touch_count=int(line.bar_touch_count),
        line_score=float(line.score),
        score_components={key: float(value) for key, value in dict(line.score_components).items()},
        projected_price_current=float(line.projected_price_current),
        projected_price_next=float(line.projected_price_next),
        projected_time_current=current_time,
        projected_time_next=next_time,
        is_active=state in {"confirmed", "armed", "triggered"},
        is_invalidated=state in {"invalidated", "expired"},
        invalidation_reason=invalidation_reason,
        invalidation_bar_index=line.invalidation_index,
        invalidation_timestamp=invalidation_timestamp,
        display_rank=meta.display_rank if meta is not None else None,
        display_class=meta.display_class if meta is not None else "debug",
        line_usability_score=meta.line_usability_score if meta is not None else 0.0,
        last_quality_touch_index=meta.last_quality_touch_index if meta is not None else line.latest_confirming_touch_index,
        collapsed_invalidation_count=meta.collapsed_invalidation_count if meta is not None else 1,
    )


def _serialize_touch_points(
    lines,
    *,
    display_meta,
    timestamps: list[int],
    highs: list[float],
    lows: list[float],
) -> list[StrategyTouchPointModel]:
    points: list[StrategyTouchPointModel] = []
    seen: set[tuple[str, int, str]] = set()

    for line in lines:
        meta = display_meta.get(line.line_id)
        if meta is None or meta.display_class == "debug":
            continue
        confirming_indices, bar_touch_indices = filter_display_touch_indices(line)
        for index in confirming_indices:
            key = (line.line_id, int(index), "confirming")
            if key in seen or index >= len(timestamps):
                continue
            seen.add(key)
            price = highs[index] if line.side == "resistance" else lows[index]
            line_value = (line.slope * index) + line.intercept
            points.append(
                StrategyTouchPointModel(
                    line_id=line.line_id,
                    timestamp=timestamps[index],
                    bar_index=int(index),
                    price=price,
                    touch_type="confirming",
                    residual=abs(price - line_value),
                    is_confirming_touch=True,
                    side=line.side,
                    display_visible=True,
                    display_class="confirming",
                )
            )
        for index in bar_touch_indices:
            key = (line.line_id, int(index), "bar")
            if key in seen or index >= len(timestamps):
                continue
            seen.add(key)
            price = highs[index] if line.side == "resistance" else lows[index]
            line_value = (line.slope * index) + line.intercept
            points.append(
                StrategyTouchPointModel(
                    line_id=line.line_id,
                    timestamp=timestamps[index],
                    bar_index=int(index),
                    price=price,
                    touch_type="bar",
                    residual=abs(price - line_value),
                    is_confirming_touch=False,
                    side=line.side,
                    display_visible=True,
                    display_class="bar",
                )
            )

    points.sort(key=lambda point: (point.timestamp, point.line_id, point.touch_type))
    return points


def _serialize_invalidations(
    lines,
    invalidations,
    *,
    display_meta,
    timestamps: list[int],
) -> list[StrategyLineStateModel]:
    line_lookup = {line.line_id: line for line in lines}
    collapsed = collapse_display_invalidations(
        [line_lookup[state.line_id] for state in invalidations if state.line_id in line_lookup],
        display_meta,
    )
    state_lookup = {state.line_id: state for state in invalidations}
    serialized: list[StrategyLineStateModel] = []
    for line, collapsed_count in collapsed:
        state = state_lookup.get(line.line_id)
        if state is None:
            continue
        serialized.append(
            _serialize_line_state(
                state,
                lines,
                display_meta=display_meta,
                timestamps=timestamps,
                collapsed_invalidation_count=collapsed_count,
            )
        )
    return serialized


def _serialize_line_state(
    state,
    lines,
    *,
    display_meta,
    timestamps: list[int],
    collapsed_invalidation_count: int = 1,
) -> StrategyLineStateModel:
    line_lookup = {line.line_id: line for line in lines}
    line = line_lookup.get(state.line_id)
    meta = display_meta.get(state.line_id)
    invalidation_bar_index = line.invalidation_index if line is not None else None
    invalidation_timestamp = (
        timestamps[invalidation_bar_index]
        if invalidation_bar_index is not None and 0 <= invalidation_bar_index < len(timestamps)
        else None
    )
    payload = asdict(state)
    payload.update(
        {
            "invalidation_bar_index": invalidation_bar_index,
            "invalidation_timestamp": invalidation_timestamp,
            "display_rank": meta.display_rank if meta is not None else None,
            "display_class": meta.display_class if meta is not None else "debug",
            "line_usability_score": meta.line_usability_score if meta is not None else 0.0,
            "collapsed_invalidation_count": collapsed_invalidation_count,
        }
    )
    return StrategyLineStateModel.model_validate(payload)


def _next_timestamp(timestamps: list[int], current_index: int) -> int:
    current_time = timestamps[current_index]
    if current_index > 0:
        delta = current_time - timestamps[current_index - 1]
        if delta > 0:
            return current_time + delta
    if len(timestamps) > 1:
        delta = timestamps[1] - timestamps[0]
        if delta > 0:
            return current_time + delta
    return current_time


__all__ = ["router"]
