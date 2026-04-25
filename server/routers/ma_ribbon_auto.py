"""HTTP API for MA-ribbon auto-execution control."""
from __future__ import annotations
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from server.strategy.ma_ribbon_auto_state import (
    AutoState, load_state, save_state, _STATE_PATH_DEFAULT,
    current_ramp_cap_pct, _to_dict,
)


router = APIRouter(prefix="/api/ma_ribbon_auto", tags=["ma_ribbon_auto"])
_STATE_PATH: Path = _STATE_PATH_DEFAULT


def _state() -> AutoState:
    return load_state(path=_STATE_PATH)


def _save(state: AutoState) -> None:
    save_state(state, path=_STATE_PATH)


@router.get("/status")
def get_status() -> dict:
    s = _state()
    now = int(time.time())
    return {
        "enabled": s.enabled,
        "halted": s.halted,
        "halt_reason": s.halt_reason,
        "locked_until_utc": s.locked_until_utc,
        "first_enabled_at_utc": s.first_enabled_at_utc,
        "current_ramp_cap_pct": current_ramp_cap_pct(s, now),
        "config": _to_dict(s.config),
        "ledger": {
            "open_positions_count": len(s.ledger.open_positions),
            "realized_pnl_usd_cumulative": s.ledger.realized_pnl_usd_cumulative,
        },
        "pending_signals_count": len(s.pending_signals),
        "errors_recent_count": len(s.errors_recent),
    }


class EnableRequest(BaseModel):
    confirm_acknowledged_p2_gate: bool = False
    confirm_first_day_cap_2pct: bool = False
    strategy_capital_usd: float = 0.0


@router.post("/enable")
def enable(req: EnableRequest) -> dict:
    if not (req.confirm_acknowledged_p2_gate and req.confirm_first_day_cap_2pct):
        raise HTTPException(400, detail="both confirm flags required")
    if req.strategy_capital_usd <= 0:
        raise HTTPException(400, detail="strategy_capital_usd must be > 0")
    s = _state()
    s.enabled = True
    s.config.strategy_capital_usd = req.strategy_capital_usd
    if s.first_enabled_at_utc is None:
        s.first_enabled_at_utc = int(time.time())
    _save(s)
    return get_status()


@router.post("/disable")
def disable() -> dict:
    s = _state()
    s.enabled = False
    _save(s)
    return get_status()


@router.post("/config")
def update_config(payload: dict = Body(...)) -> dict:
    s = _state()
    if "layer_risk_pct" in payload:
        for layer, val in payload["layer_risk_pct"].items():
            if not isinstance(val, (int, float)) or val <= 0 or val > 0.05:
                raise HTTPException(400, detail=f"layer_risk_pct[{layer}] = {val} out of (0, 0.05]")
            s.config.layer_risk_pct[layer] = float(val)
    if "max_concurrent_orders" in payload:
        v = payload["max_concurrent_orders"]
        if not isinstance(v, int) or v < 1 or v > 200:
            raise HTTPException(400, detail="max_concurrent_orders must be 1..200")
        s.config.max_concurrent_orders = v
    if "dd_halt_pct" in payload:
        v = payload["dd_halt_pct"]
        if not 0 < v <= 0.5:
            raise HTTPException(400, detail="dd_halt_pct must be in (0, 0.5]")
        s.config.dd_halt_pct = float(v)
    if "per_symbol_risk_cap_pct" in payload:
        v = payload["per_symbol_risk_cap_pct"]
        if not 0 < v <= 0.10:
            raise HTTPException(400, detail="per_symbol_risk_cap_pct must be in (0, 0.10]")
        s.config.per_symbol_risk_cap_pct = float(v)
    if "ribbon_buffer_pct" in payload:
        for tf, val in payload["ribbon_buffer_pct"].items():
            if not 0 < val <= 0.30:
                raise HTTPException(400, detail=f"ribbon_buffer_pct[{tf}] = {val} out of range")
            s.config.ribbon_buffer_pct[tf] = float(val)
    _save(s)
    return get_status()


@router.post("/emergency_stop")
async def emergency_stop_endpoint(payload: dict = Body(...)) -> dict:
    from server.strategy.ma_ribbon_auto_scanner import emergency_stop
    s = _state()
    reason = payload.get("reason", "manual")
    await emergency_stop(s, now_utc=int(time.time()), reason=reason)
    _save(s)
    return get_status()
