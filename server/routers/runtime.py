from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..execution import PaperExecutionConfig
from ..execution.types import dataclass_to_dict
from ..runtime import SubaccountRuntimeManager
from ..schemas.runtime import (
    RuntimeEventsResponse,
    RuntimeInstanceCreateRequest,
    RuntimeInstanceListResponse,
    RuntimeInstanceResponse,
    RuntimeInstanceUpdateRequest,
    RuntimeKillSwitchRequest,
    RuntimeTickRequest,
)

router = APIRouter(prefix="/api/runtime", tags=["runtime"])
runtime_manager = SubaccountRuntimeManager()


@router.get("/instances", response_model=RuntimeInstanceListResponse)
async def api_runtime_instances():
    return {"instances": [dataclass_to_dict(record) for record in runtime_manager.list_instances()]}


@router.post("/instances", response_model=RuntimeInstanceResponse)
async def api_runtime_create_instance(req: RuntimeInstanceCreateRequest):
    paper_config = PaperExecutionConfig(**req.paper_config.model_dump()) if req.paper_config is not None else None
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
    )
    return {"instance": dataclass_to_dict(record)}


@router.patch("/instances/{instance_id}", response_model=RuntimeInstanceResponse)
async def api_runtime_update_instance(instance_id: str, req: RuntimeInstanceUpdateRequest):
    try:
        changes = {key: value for key, value in req.model_dump().items() if value is not None}
        if "paper_config" in changes and changes["paper_config"] is not None:
            changes["paper_config"] = changes["paper_config"]
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
