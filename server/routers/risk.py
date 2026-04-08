"""
Risk management routes.
Currently: risk-limits endpoint.
Phase 3 adds: risk events, exposure meters, block reason schema.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.dependencies import get_agent

router = APIRouter(prefix="/api/agent", tags=["risk"])


class RiskLimitsRequest(BaseModel):
    max_position_pct: float | None = None
    max_total_exposure_pct: float | None = None
    max_daily_loss_pct: float | None = None
    max_drawdown_pct: float | None = None
    max_positions: int | None = None
    cooldown_seconds: int | None = None


@router.post("/risk-limits")
async def api_agent_risk_limits(req: RiskLimitsRequest):
    """Update risk limits. Values are in percentage (e.g. 5 means 5%)."""
    agent = get_agent()
    changes = []

    if req.max_position_pct is not None and 0.5 <= req.max_position_pct <= 25:
        agent.trader.risk.max_position_pct = req.max_position_pct / 100.0
        changes.append(f"max_position_pct={req.max_position_pct}%")
    if req.max_total_exposure_pct is not None and 1 <= req.max_total_exposure_pct <= 100:
        agent.trader.risk.max_total_exposure_pct = req.max_total_exposure_pct / 100.0
        changes.append(f"max_total_exposure_pct={req.max_total_exposure_pct}%")
    if req.max_daily_loss_pct is not None and 0.1 <= req.max_daily_loss_pct <= 20:
        agent.trader.risk.max_daily_loss_pct = req.max_daily_loss_pct / 100.0
        changes.append(f"max_daily_loss_pct={req.max_daily_loss_pct}%")
    if req.max_drawdown_pct is not None and 1 <= req.max_drawdown_pct <= 50:
        agent.trader.risk.max_drawdown_pct = req.max_drawdown_pct / 100.0
        changes.append(f"max_drawdown_pct={req.max_drawdown_pct}%")
    if req.max_positions is not None and 1 <= req.max_positions <= 20:
        agent.trader.risk.max_positions = req.max_positions
        changes.append(f"max_positions={req.max_positions}")
    if req.cooldown_seconds is not None and 0 <= req.cooldown_seconds <= 86400:
        agent.trader.risk.cooldown_seconds = req.cooldown_seconds
        changes.append(f"cooldown={req.cooldown_seconds}s")

    if changes:
        print(f"[Agent] Risk limits updated: {', '.join(changes)}")
    return {"ok": True, "changes": changes, "risk_limits": {
        "max_position_pct": agent.trader.risk.max_position_pct * 100,
        "max_total_exposure_pct": agent.trader.risk.max_total_exposure_pct * 100,
        "max_daily_loss_pct": agent.trader.risk.max_daily_loss_pct * 100,
        "max_drawdown_pct": agent.trader.risk.max_drawdown_pct * 100,
        "max_positions": agent.trader.risk.max_positions,
        "cooldown_seconds": agent.trader.risk.cooldown_seconds,
    }}
