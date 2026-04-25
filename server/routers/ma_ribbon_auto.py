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
