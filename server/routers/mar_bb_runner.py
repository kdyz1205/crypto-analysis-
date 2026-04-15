"""HTTP API for the MA Ribbon + BB Exit live runner."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..strategy.mar_bb_runner import (
    DEFAULT_RUNNER_CFG,
    get_state,
    manual_kick,
    start_runner,
    stop_runner,
)

router = APIRouter(prefix="/api/mar-bb", tags=["mar_bb_runner"])


class StartReq(BaseModel):
    top_n: int | None = None
    timeframe: str | None = None
    scan_interval_s: int | None = None
    notional_usd: float | None = None
    leverage: int | None = None
    max_concurrent_positions: int | None = None
    mode: str | None = None
    min_bars: int | None = None
    dry_run: bool | None = None


@router.post("/start")
async def api_start(req: StartReq):
    config = {k: v for k, v in req.model_dump().items() if v is not None}
    return start_runner(config)


@router.post("/stop")
async def api_stop():
    return stop_runner()


@router.get("/state")
async def api_state():
    return {"ok": True, "state": get_state(), "default_config": DEFAULT_RUNNER_CFG}


@router.post("/kick")
async def api_kick():
    """Force one immediate scan (bypasses interval timer)."""
    s = await manual_kick()
    return {"ok": True, "state": s}


__all__ = ["router"]
