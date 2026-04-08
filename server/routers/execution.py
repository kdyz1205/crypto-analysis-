"""
Execution routes: OKX API keys and connection status.
Phase 3 adds: order ticket, position management, slippage estimation.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.dependencies import get_agent

router = APIRouter(prefix="/api/agent", tags=["execution"])


class OKXKeysRequest(BaseModel):
    api_key: str
    secret: str
    passphrase: str


@router.post("/okx-keys")
async def api_agent_okx_keys(req: OKXKeysRequest):
    """Set OKX API keys for live trading. Stored in memory only (not persisted)."""
    agent = get_agent()
    agent.trader.set_api_keys(req.api_key, req.secret, req.passphrase)
    balance = await agent.trader.get_account_balance()
    if balance.get("ok"):
        return {
            "ok": True,
            "message": "OKX API keys verified successfully",
            "balance": balance,
            "has_keys": True,
        }
    return {
        "ok": False,
        "reason": f"Keys set but verification failed: {balance.get('reason', 'unknown')}",
        "has_keys": True,
    }


@router.get("/okx-status")
async def api_agent_okx_status():
    """Check OKX connection status and account balance."""
    agent = get_agent()
    has_keys = agent.trader.has_api_keys()
    if not has_keys:
        return {"ok": True, "has_keys": False, "mode": agent.trader.state.mode}
    balance = await agent.trader.get_account_balance()
    return {
        "ok": True,
        "has_keys": True,
        "mode": agent.trader.state.mode,
        "balance": balance if balance.get("ok") else None,
        "error": balance.get("reason") if not balance.get("ok") else None,
    }
