"""
On-chain / Smart Money proxy routes.
Proxies requests to the smart-money API running on port 8002.
"""

from typing import Optional

import httpx
from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/api/onchain", tags=["onchain"])

SMART_MONEY_BASE = "http://127.0.0.1:8002"


async def _sm_proxy(method: str, path: str, body: Optional[dict] = None):
    """Proxy a request to the smart-money API."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if method == "GET":
                resp = await client.get(f"{SMART_MONEY_BASE}{path}")
            elif method == "POST":
                resp = await client.post(f"{SMART_MONEY_BASE}{path}", json=body)
            elif method == "PUT":
                resp = await client.put(f"{SMART_MONEY_BASE}{path}", json=body)
            elif method == "DELETE":
                resp = await client.delete(f"{SMART_MONEY_BASE}{path}")
            else:
                return {"error": f"Unsupported method: {method}"}
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"error": f"Invalid JSON from smart-money API: {resp.text[:200]}"}
    except (httpx.ConnectError, httpx.TimeoutException):
        return {"error": "Smart Money API not running (port 8002)", "offline": True}
    except Exception as e:
        return {"error": str(e)}


@router.get("/health")
async def api_onchain_health():
    """Check if smart-money API is reachable."""
    return await _sm_proxy("GET", "/health")


@router.get("/wallets")
async def api_onchain_wallets():
    """List all analyzed wallet profiles."""
    return await _sm_proxy("GET", "/wallets/")


@router.get("/wallets/smart-money")
async def api_onchain_smart_money():
    """List wallets identified as smart money."""
    return await _sm_proxy("GET", "/wallets/smart-money")


@router.get("/wallets/{address}")
async def api_onchain_wallet_detail(address: str):
    """Get profile for a specific wallet address."""
    return await _sm_proxy("GET", f"/wallets/{address}")


@router.post("/wallets/track/{address}")
async def api_onchain_track_wallet(address: str):
    """Start tracking a wallet address."""
    return await _sm_proxy("POST", f"/wallets/track/{address}")


@router.delete("/wallets/track/{address}")
async def api_onchain_untrack_wallet(address: str):
    """Stop tracking a wallet."""
    return await _sm_proxy("DELETE", f"/wallets/track/{address}")


@router.get("/signals")
async def api_onchain_signals(limit: int = Query(50, ge=1, le=200)):
    """Get latest trading signals from smart money analysis."""
    return await _sm_proxy("GET", f"/signals/?limit={limit}")


@router.get("/signals/recommendations")
async def api_onchain_recommendations(limit: int = Query(20, ge=1, le=200)):
    """Get latest recommendations."""
    return await _sm_proxy("GET", f"/signals/recommendations?limit={limit}")


@router.get("/signals/params")
async def api_onchain_get_params():
    """Get current analysis parameters."""
    return await _sm_proxy("GET", "/signals/params")


@router.put("/signals/params")
async def api_onchain_update_params(request: Request):
    """Update analysis parameters."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON body"}
    return await _sm_proxy("PUT", "/signals/params", body)


@router.get("/token/analyze/{token_address}")
async def api_onchain_token_analyze(
    token_address: str,
    network: str = Query("solana"),
    pool: str = Query(None),
):
    """Run full smart-money analysis on a live token."""
    path = f"/token/analyze/{token_address}?network={network}"
    if pool:
        path += f"&pool={pool}"
    return await _sm_proxy("GET", path)


@router.get("/validator/backtest")
async def api_onchain_backtest():
    """Get latest backtest result."""
    return await _sm_proxy("GET", "/validator/backtest")


@router.get("/validator/high-confidence-wallets")
async def api_onchain_high_confidence():
    """Get wallets with high win rate."""
    return await _sm_proxy("GET", "/validator/high-confidence-wallets")
