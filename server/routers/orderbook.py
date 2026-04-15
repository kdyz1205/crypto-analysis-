"""API for real-time orderbook features."""
from __future__ import annotations
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/orderbook", tags=["orderbook"])


@router.get("/features")
async def api_features(symbol: str | None = Query(None)):
    """Get real-time market microstructure features."""
    from server.hft.orderbook_service import get_features, get_all_features
    if symbol:
        f = get_features(symbol)
        if f is None:
            return {"ok": False, "reason": f"no data for {symbol}"}
        return {"ok": True, "symbol": symbol, "features": {
            "mid": f.mid, "spread_bps": f.spread_bps,
            "imbalance_3": f.imbalance_3, "imbalance_5": f.imbalance_5,
            "microprice": f.microprice, "cancel_pressure": f.cancel_pressure,
            "regime": f.regime, "toxicity": f.toxicity,
        }}
    return {"ok": True, "features": get_all_features()}


@router.get("/status")
async def api_status():
    """Orderbook service status."""
    from server.hft.orderbook_service import get_status
    return {"ok": True, **get_status()}


__all__ = ["router"]
