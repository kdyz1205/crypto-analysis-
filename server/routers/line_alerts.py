"""HTTP API for trendline price alerts (TradingView-style)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..strategy.line_alerts import add_alert, remove_alert, list_alerts

router = APIRouter(prefix="/api/alerts", tags=["line_alerts"])


class AddAlertReq(BaseModel):
    symbol: str
    timeframe: str
    slope: float
    intercept: float
    kind: str = "support"        # support | resistance
    mode: str = "single"         # single | repeat
    label: str = ""


class RemoveAlertReq(BaseModel):
    alert_id: str


@router.post("/add")
async def api_add_alert(req: AddAlertReq):
    a = add_alert(
        symbol=req.symbol, timeframe=req.timeframe,
        slope=req.slope, intercept=req.intercept,
        kind=req.kind, mode=req.mode, label=req.label,
    )
    return {"ok": True, "alert_id": a.alert_id}


@router.post("/remove")
async def api_remove_alert(req: RemoveAlertReq):
    removed = remove_alert(req.alert_id)
    return {"ok": removed}


@router.get("/list")
async def api_list_alerts():
    return {"alerts": list_alerts()}
