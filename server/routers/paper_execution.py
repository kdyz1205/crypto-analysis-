from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..data_service import get_ohlcv_with_df
from ..drawings import augment_snapshot_with_manual_signals
from ..history_coverage import build_analysis_history
from ..execution import PaperExecutionConfig, PaperExecutionEngine
from ..execution.types import dataclass_to_dict
from ..schemas.paper_execution import (
    PaperExecutionConfigResponse,
    PaperExecutionConfigUpdateRequest,
    PaperExecutionResetRequest,
    PaperExecutionStateResponse,
    PaperExecutionStepRequest,
    PaperExecutionStepResponse,
    PaperKillSwitchRequest,
)
from ..strategy import ReplayResult, StrategyConfig, apply_strategy_overrides, build_latest_snapshot

router = APIRouter(prefix="/api/paper-execution", tags=["paper-execution"])

VALID_INTERVALS = {"5m", "15m", "1h", "4h", "1d"}
VALID_HISTORY_MODES = {"fast_window", "full_history"}
DEFAULT_DAYS_BY_INTERVAL = {
    "5m": 7,
    "15m": 21,
    "1h": 120,
    "4h": 365,
    "1d": 365,
}

paper_engine = PaperExecutionEngine()


@router.get("/state", response_model=PaperExecutionStateResponse)
async def api_paper_execution_state():
    return {"state": dataclass_to_dict(paper_engine.get_state())}


@router.post("/reset", response_model=PaperExecutionStateResponse)
async def api_paper_execution_reset(req: PaperExecutionResetRequest | None = None):
    if req is not None and req.starting_equity is not None:
        paper_engine.update_config(starting_equity=float(req.starting_equity))
    return {"state": dataclass_to_dict(paper_engine.reset())}


@router.get("/config", response_model=PaperExecutionConfigResponse)
async def api_paper_execution_config():
    return {"config": dataclass_to_dict(paper_engine.config)}


@router.post("/config", response_model=PaperExecutionConfigResponse)
async def api_paper_execution_set_config(req: PaperExecutionConfigUpdateRequest):
    changes = {key: value for key, value in req.model_dump().items() if value is not None}
    if changes:
        paper_engine.update_config(**changes)
    return {"config": dataclass_to_dict(paper_engine.config)}


@router.post("/kill-switch", response_model=PaperExecutionStateResponse)
async def api_paper_execution_kill_switch(req: PaperKillSwitchRequest):
    paper_engine.set_manual_kill_switch(req.blocked, req.reason)
    return {"state": dataclass_to_dict(paper_engine.get_state())}


@router.post("/step", response_model=PaperExecutionStepResponse)
async def api_paper_execution_step(req: PaperExecutionStepRequest):
    if req.interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval. Must be one of: {sorted(VALID_INTERVALS)}")
    if req.history_mode not in VALID_HISTORY_MODES:
        raise HTTPException(400, f"Invalid history_mode. Must be one of: {sorted(VALID_HISTORY_MODES)}")

    normalized_symbol = _normalize_symbol(req.symbol)
    requested_days = req.days or DEFAULT_DAYS_BY_INTERVAL.get(req.interval, 120)

    try:
        candles_df, strategy_cfg, history = await _load_strategy_inputs(
            normalized_symbol,
            req.interval,
            requested_days,
            req.history_mode,
            req.analysis_bars,
        )
        strategy_cfg = apply_strategy_overrides(
            strategy_cfg,
            lookback_bars=req.lookback_bars,
            min_touches=req.min_touches,
            confirm_threshold=req.confirm_threshold,
            score_threshold=req.score_threshold,
            rr_target=req.rr_target,
        )
        stream_key = f"{normalized_symbol}:{req.interval}"
        last_processed = paper_engine.last_processed_bar_by_stream.get(stream_key, -1)
        target_bar = last_processed + 1 if req.bar_index is None else req.bar_index
        max_index = len(candles_df) - 1
        if target_bar > max_index:
            target_bar = max_index
        if target_bar < last_processed + 1:
            state = paper_engine.get_state()
            return {
                "stream": stream_key,
                "processedBars": [],
                "lastProcessedBar": last_processed,
                "history": history,
                "state": dataclass_to_dict(state),
            }
        replay_result, snapshot_offset = _build_step_replay_result(
            candles_df,
            strategy_cfg,
            normalized_symbol,
            req.interval,
            start_bar=max(0, last_processed + 1),
            end_bar=target_bar,
            enabled_trigger_modes=tuple(req.trigger_modes),
            strategy_window_bars=req.strategy_window_bars,
        )
        result = paper_engine.step(
            normalized_symbol,
            req.interval,
            candles_df,
            replay_result,
            bar_index=req.bar_index,
            snapshot_offset=snapshot_offset,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    return {
        "stream": result["stream"],
        "processedBars": result["processedBars"],
        "lastProcessedBar": result["lastProcessedBar"],
        "history": history,
        "state": dataclass_to_dict(result["state"]),
    }


async def _load_strategy_inputs(
    symbol: str,
    interval: str,
    days: int,
    history_mode: str,
    analysis_bars: int,
) -> tuple[pd.DataFrame, StrategyConfig, dict]:
    polars_df, market_payload = await get_ohlcv_with_df(symbol, interval, None, days, history_mode=history_mode)
    if polars_df is None or polars_df.is_empty():
        raise ValueError(f"No data for {symbol} {interval}")

    candles_df = _standardize_strategy_candles(polars_df)
    if len(candles_df) > analysis_bars:
        candles_df = candles_df.iloc[-analysis_bars:].reset_index(drop=True)

    price_precision = market_payload.get("pricePrecision") if isinstance(market_payload, dict) else None
    history = build_analysis_history(market_payload, candles_df)
    return candles_df, _config_with_market_precision(StrategyConfig(), price_precision), history


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


def _build_step_replay_result(
    candles_df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    symbol: str,
    interval: str,
    *,
    start_bar: int,
    end_bar: int,
    enabled_trigger_modes: tuple[str, ...] | list[str] | None = None,
    strategy_window_bars: int | None = None,
) -> tuple[ReplayResult, int]:
    if end_bar < start_bar:
        return ReplayResult(symbol=symbol, timeframe=interval, snapshots=tuple()), start_bar

    snapshots = []
    for current_bar in range(start_bar, end_bar + 1):
        prefix_start = max(0, current_bar - strategy_window_bars + 1) if strategy_window_bars else 0
        prefix = candles_df.iloc[prefix_start : current_bar + 1].reset_index(drop=True)
        snapshot = build_latest_snapshot(
            prefix,
            strategy_cfg,
            symbol=symbol,
            timeframe=interval,
            enabled_trigger_modes=enabled_trigger_modes,
        )
        snapshot = augment_snapshot_with_manual_signals(
            snapshot,
            prefix,
            strategy_cfg,
            symbol=symbol,
            timeframe=interval,
            enabled_trigger_modes=enabled_trigger_modes,
        )
        snapshots.append(snapshot)

    return ReplayResult(symbol=symbol, timeframe=interval, snapshots=tuple(snapshots)), start_bar


__all__ = ["paper_engine", "router"]
