from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..execution.live_adapter import LiveExecutionAdapter
from ..execution.live_engine import LiveBridgeConfig, LiveExecutionEngine
from ..execution.types import OrderIntent
from ..schemas.live_execution import (
    LiveCloseRequest,
    LiveCloseResponse,
    LiveExecutionStatusResponse,
    LivePreviewRequest,
    LivePreviewResponse,
    LiveReconcileRequest,
    LiveReconcileResponse,
    LiveSubmitRequest,
    LiveSubmitResponse,
)
from .paper_execution import paper_engine

router = APIRouter(prefix="/api/live-execution", tags=["live-execution"])


def _adapter_provider() -> LiveExecutionAdapter:
    return LiveExecutionAdapter()


live_engine = LiveExecutionEngine(
    adapter_provider=_adapter_provider,
    config=LiveBridgeConfig.from_env(),
)


@router.get("/status", response_model=LiveExecutionStatusResponse)
async def api_live_execution_status():
    return {"status": live_engine.get_status()}


@router.post("/reconcile", response_model=LiveReconcileResponse)
async def api_live_execution_reconcile(req: LiveReconcileRequest):
    mode = _normalize_mode(req.mode)
    reconciliation = await live_engine.reconcile_startup(mode)
    return {"reconciliation": reconciliation}


@router.post("/preview", response_model=LivePreviewResponse)
async def api_live_execution_preview(req: LivePreviewRequest):
    mode = _normalize_mode(req.mode)
    intent = _resolve_intent(req.order_intent_id, req.signal_id)
    result = await live_engine.preview_live_submission(intent, mode=mode)
    return {"result": result}


@router.post("/submit", response_model=LiveSubmitResponse)
async def api_live_execution_submit(req: LiveSubmitRequest):
    mode = _normalize_mode(req.mode)
    intent = _resolve_intent(req.order_intent_id, None)
    result = await live_engine.execute_live_submission(intent, mode=mode, confirm=req.confirm)
    return {"result": result}


@router.post("/close", response_model=LiveCloseResponse)
async def api_live_execution_close(req: LiveCloseRequest):
    mode = _normalize_mode(req.mode)
    result = await live_engine.close_live_position(req.symbol, mode=mode, confirm=req.confirm)
    return {"result": result}


def _resolve_intent(order_intent_id: str | None, signal_id: str | None) -> OrderIntent:
    if signal_id:
        intent = paper_engine.order_manager.get_intent(signal_id)
        if intent is not None:
            return intent
        raise HTTPException(404, f"Unknown signal_id: {signal_id}")

    if order_intent_id:
        for intent in paper_engine.order_manager.get_intents():
            if intent.order_intent_id == order_intent_id:
                return intent
        raise HTTPException(404, f"Unknown order_intent_id: {order_intent_id}")

    raise HTTPException(400, "order_intent_id or signal_id is required")


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized not in {"demo", "live"}:
        raise HTTPException(400, "mode must be one of: demo, live")
    return normalized


__all__ = ["live_engine", "router"]
