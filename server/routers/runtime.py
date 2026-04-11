from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..execution import PaperExecutionConfig
from ..execution.types import dataclass_to_dict
from ..runtime import RuntimeStrategyConfig, SubaccountRuntimeManager
from ..schemas.runtime import (
    RuntimeEventsResponse,
    RuntimeInstanceCreateRequest,
    RuntimeInstanceListResponse,
    RuntimeInstanceResponse,
    RuntimeInstanceUpdateRequest,
    RuntimeKillSwitchRequest,
    RuntimeTickRequest,
)

from ..strategy.catalog import list_templates, get_template
from ..strategy.evolution import EvolutionEngine
from ..strategy.sr_strategy import build_market_context, evaluate_bar
from ..strategy.indicators import list_indicators

evolution_engine = EvolutionEngine()

router = APIRouter(prefix="/api/runtime", tags=["runtime"])
runtime_manager = SubaccountRuntimeManager()


@router.get("/catalog")
async def api_strategy_catalog():
    """List all available strategy templates."""
    return {"templates": list_templates()}


@router.get("/leaderboard")
async def api_leaderboard(limit: int = Query(20, ge=1, le=50)):
    """Get strategy evolution leaderboard — top performing variants."""
    return {
        **evolution_engine.get_stats(),
        "leaderboard": evolution_engine.get_leaderboard(limit),
    }


@router.post("/leaderboard/start")
async def api_start_evolution():
    """Start the strategy evolution engine."""
    evolution_engine.start()
    return {"ok": True, "message": "Evolution engine started"}


@router.post("/leaderboard/stop")
async def api_stop_evolution():
    """Stop the strategy evolution engine."""
    evolution_engine.stop()
    return {"ok": True, "message": "Evolution engine stopped"}


@router.post("/leaderboard/{variant_id}/copy")
async def api_copy_variant(variant_id: str, live_mode: str = Query("disabled")):
    """Copy a leaderboard variant into a runtime instance."""
    config = evolution_engine.copy_to_runtime_config(variant_id)
    if not config:
        raise HTTPException(404, f"Variant not found: {variant_id}")

    paper_config = PaperExecutionConfig(**config.pop("paper_config"))
    strategy_config = RuntimeStrategyConfig(**config.pop("strategy_config"))
    record = runtime_manager.create_instance(
        **config,
        live_mode=live_mode,
        paper_config=paper_config,
        strategy_config=strategy_config,
    )
    try:
        runtime_manager.start_instance(record.config.instance_id)
    except Exception:
        pass
    return {"ok": True, "instance": dataclass_to_dict(record)}


@router.get("/sr-context")
async def api_sr_context(
    symbol: str = Query("HYPEUSDT"),
    timeframe: str = Query("4h"),
):
    """Get full SR market context + strategy decision for current bar."""
    from ..data_service import get_ohlcv_with_df
    from ..strategy.config import StrategyConfig
    from dataclasses import asdict

    normalized = symbol.upper().replace("/", "")
    if not normalized.endswith("USDT"):
        normalized += "USDT"

    df_polars, _ = await get_ohlcv_with_df(normalized, timeframe, end_time=None, days=120, history_mode="fast_window")
    if df_polars is None or df_polars.is_empty():
        raise HTTPException(404, f"No data for {normalized} {timeframe}")

    pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(pd.Timestamp(v).timestamp()))
    for col in ("open", "high", "low", "close", "volume"):
        pdf[col] = pd.to_numeric(pdf[col], errors="raise")

    ctx = build_market_context(pdf, StrategyConfig(), symbol=normalized, timeframe=timeframe)
    decision = evaluate_bar(pdf, ctx)

    return {
        "context": {
            "symbol": ctx.symbol,
            "timeframe": ctx.timeframe,
            "current_price": ctx.current_price,
            "atr": round(ctx.atr, 4),
            "trend_state": ctx.trend_state,
            "trend_strength": round(ctx.trend_strength, 3),
            "zones": [asdict(z) for z in ctx.zones],
            "nearest_support": asdict(ctx.nearest_support) if ctx.nearest_support else None,
            "nearest_resistance": asdict(ctx.nearest_resistance) if ctx.nearest_resistance else None,
            "dist_support_atr": round(ctx.distance_to_nearest_support_atr, 2),
            "dist_resistance_atr": round(ctx.distance_to_nearest_resistance_atr, 2),
        },
        "decision": asdict(decision),
    }


@router.get("/indicators")
async def api_list_indicators():
    """List all available technical indicators (200+)."""
    indicators = list_indicators()
    by_category = {}
    for ind in indicators:
        cat = ind["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(ind)
    return {"total": len(indicators), "by_category": by_category, "indicators": indicators}


@router.post("/catalog/{template_id}/launch")
async def api_launch_from_catalog(
    template_id: str,
    symbol: str = Query(..., description="e.g. BTCUSDT"),
    timeframe: str = Query("4h"),
    live_mode: str = Query("disabled", description="disabled | demo | live"),
    starting_equity: float = Query(10000.0, ge=10),
):
    """Create and start a runtime instance from a strategy template."""
    template = get_template(template_id)
    if not template:
        raise HTTPException(404, f"Unknown template: {template_id}")
    if timeframe not in template.supported_timeframes:
        raise HTTPException(400, f"{template.name} does not support {timeframe}. Use: {template.supported_timeframes}")

    params = template.default_params
    paper_config = PaperExecutionConfig(
        starting_equity=starting_equity,
        risk_per_trade=params.get("risk_per_trade", 0.003),
    )
    strategy_config = RuntimeStrategyConfig(
        enabled_trigger_modes=tuple(template.default_trigger_modes),
        lookback_bars=params.get("lookback_bars", 80),
        min_touches=params.get("min_touches"),
        rr_target=params.get("rr_target"),
        window_bars=params.get("lookback_bars", 100),
    )
    record = runtime_manager.create_instance(
        label=f"{symbol}-{template.name}",
        symbol=symbol,
        timeframe=timeframe,
        history_mode="fast_window",
        analysis_bars=500,
        days=365,
        tick_interval_seconds=60,
        live_mode=live_mode,
        paper_config=paper_config,
        strategy_config=strategy_config,
    )
    # Auto-start
    try:
        runtime_manager.start_instance(record.config.instance_id)
    except Exception:
        pass
    return {"instance": dataclass_to_dict(record), "template": template.name}


@router.get("/instances", response_model=RuntimeInstanceListResponse)
async def api_runtime_instances():
    return {"instances": [dataclass_to_dict(record) for record in runtime_manager.list_instances()]}


@router.post("/instances", response_model=RuntimeInstanceResponse)
async def api_runtime_create_instance(req: RuntimeInstanceCreateRequest):
    paper_config = PaperExecutionConfig(**req.paper_config.model_dump()) if req.paper_config is not None else None
    strategy_config = RuntimeStrategyConfig(**req.strategy_config.model_dump()) if req.strategy_config is not None else None
    record = runtime_manager.create_instance(
        label=req.label,
        symbol=req.symbol,
        timeframe=req.timeframe,
        subaccount_label=req.subaccount_label,
        history_mode=req.history_mode,
        analysis_bars=req.analysis_bars,
        days=req.days,
        tick_interval_seconds=req.tick_interval_seconds,
        auto_restart_on_boot=req.auto_restart_on_boot,
        live_mode=req.live_mode,
        auto_live_preview=req.auto_live_preview,
        auto_live_submit=req.auto_live_submit,
        notes=req.notes,
        paper_config=paper_config,
        strategy_config=strategy_config,
    )
    return {"instance": dataclass_to_dict(record)}


@router.patch("/instances/{instance_id}", response_model=RuntimeInstanceResponse)
async def api_runtime_update_instance(instance_id: str, req: RuntimeInstanceUpdateRequest):
    try:
        changes = {key: value for key, value in req.model_dump().items() if value is not None}
        if "paper_config" in changes and changes["paper_config"] is not None:
            changes["paper_config"] = changes["paper_config"].model_dump()
        if "strategy_config" in changes and changes["strategy_config"] is not None:
            changes["strategy_config"] = changes["strategy_config"].model_dump()
        record = runtime_manager.update_instance(instance_id, **changes)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown runtime instance: {instance_id}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"instance": dataclass_to_dict(record)}


@router.delete("/instances/{instance_id}", response_model=RuntimeInstanceListResponse)
async def api_runtime_delete_instance(instance_id: str):
    try:
        runtime_manager.delete_instance(instance_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown runtime instance: {instance_id}") from exc
    return {"instances": [dataclass_to_dict(record) for record in runtime_manager.list_instances()]}


@router.post("/instances/{instance_id}/start", response_model=RuntimeInstanceResponse)
async def api_runtime_start_instance(instance_id: str):
    try:
        record = runtime_manager.start_instance(instance_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown runtime instance: {instance_id}") from exc
    return {"instance": dataclass_to_dict(record)}


@router.post("/instances/{instance_id}/stop", response_model=RuntimeInstanceResponse)
async def api_runtime_stop_instance(instance_id: str):
    try:
        record = runtime_manager.stop_instance(instance_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown runtime instance: {instance_id}") from exc
    return {"instance": dataclass_to_dict(record)}


@router.post("/instances/{instance_id}/tick", response_model=RuntimeInstanceResponse)
async def api_runtime_tick_instance(instance_id: str, req: RuntimeTickRequest):
    try:
        record = await runtime_manager.tick_instance(instance_id, bar_index=req.bar_index)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown runtime instance: {instance_id}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"instance": dataclass_to_dict(record)}


@router.post("/instances/{instance_id}/kill-switch", response_model=RuntimeInstanceResponse)
async def api_runtime_kill_switch(instance_id: str, req: RuntimeKillSwitchRequest):
    try:
        record = runtime_manager.set_instance_kill_switch(instance_id, req.blocked, req.reason)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown runtime instance: {instance_id}") from exc
    return {"instance": dataclass_to_dict(record)}


@router.get("/events", response_model=RuntimeEventsResponse)
async def api_runtime_events(instance_id: str | None = Query(None), limit: int = Query(50, ge=1, le=200)):
    return {"events": runtime_manager.get_events(instance_id=instance_id, limit=limit)}


__all__ = ["router", "runtime_manager"]
