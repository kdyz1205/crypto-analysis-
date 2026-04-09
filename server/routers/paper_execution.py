from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..data_service import get_ohlcv_with_df
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
from ..strategy import StrategyConfig, replay_strategy

router = APIRouter(prefix="/api/paper-execution", tags=["paper-execution"])

VALID_INTERVALS = {"5m", "15m", "1h", "4h", "1d"}
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

    normalized_symbol = _normalize_symbol(req.symbol)
    requested_days = req.days or DEFAULT_DAYS_BY_INTERVAL.get(req.interval, 120)

    try:
        candles_df, strategy_cfg = await _load_strategy_inputs(
            normalized_symbol,
            req.interval,
            requested_days,
            req.analysis_bars,
        )
        replay_result = replay_strategy(candles_df, strategy_cfg, symbol=normalized_symbol, timeframe=req.interval)
        result = paper_engine.step(
            normalized_symbol,
            req.interval,
            candles_df,
            replay_result,
            bar_index=req.bar_index,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    return {
        "stream": result["stream"],
        "processedBars": result["processedBars"],
        "lastProcessedBar": result["lastProcessedBar"],
        "state": dataclass_to_dict(result["state"]),
    }


async def _load_strategy_inputs(
    symbol: str,
    interval: str,
    days: int,
    analysis_bars: int,
) -> tuple[pd.DataFrame, StrategyConfig]:
    polars_df, market_payload = await get_ohlcv_with_df(symbol, interval, None, days)
    if polars_df is None or polars_df.is_empty():
        raise ValueError(f"No data for {symbol} {interval}")

    candles_df = _standardize_strategy_candles(polars_df)
    if len(candles_df) > analysis_bars:
        candles_df = candles_df.iloc[-analysis_bars:].reset_index(drop=True)

    price_precision = market_payload.get("pricePrecision") if isinstance(market_payload, dict) else None
    return candles_df, _config_with_market_precision(StrategyConfig(), price_precision)


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


__all__ = ["paper_engine", "router"]
