"""
Execution routes: OKX API keys, connection status, order ticket preview,
exposure breakdown.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.dependencies import get_agent

router = APIRouter(prefix="/api/agent", tags=["execution"])

# OKX taker fee is 0.05% (5 bps) for swap contracts
FEE_BPS = 5


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


# ── Order Ticket ──────────────────────────────────────────────────────────

class OrderTicketRequest(BaseModel):
    symbol: str
    side: str  # long | short
    size_pct: float | None = None   # percent of equity (e.g. 5)
    size_usd: float | None = None    # absolute usd
    entry: float | None = None       # optional hint, default = current price
    stop: float
    target: float


def _estimate_slippage_bps(size_usd: float) -> float:
    """
    Very simple slippage model: bigger orders leak more.
    Real-world this would use order book depth / 24h volume ratio.
    """
    if size_usd < 1000: return 2.0
    if size_usd < 10000: return 5.0
    if size_usd < 100000: return 10.0
    return 20.0


@router.post("/order-ticket/preview")
async def api_order_ticket_preview(req: OrderTicketRequest):
    """
    Preview an order before submission. Returns expected fill, fees, slippage,
    R-multiple, liquidation distance, and validation flags.
    """
    agent = get_agent()
    s = agent.trader.state
    r = agent.trader.risk
    errors: list[str] = []
    warnings: list[str] = []

    side = req.side.lower()
    if side not in ("long", "short"):
        errors.append(f"Invalid side: {req.side}")

    # Resolve entry price
    entry = req.entry
    if entry is None or entry <= 0:
        try:
            entry = await agent.trader.get_price(req.symbol)
        except Exception:
            entry = None
    if entry is None or entry <= 0:
        errors.append("Could not determine entry price")
        return {"ok": False, "errors": errors}

    # Resolve size
    if req.size_usd and req.size_usd > 0:
        size_usd = req.size_usd
    elif req.size_pct and req.size_pct > 0:
        size_usd = (req.size_pct / 100.0) * s.equity
    else:
        # Default = max_position_pct
        size_usd = r.max_position_pct * s.equity
        warnings.append(f"Using default position size = {r.max_position_pct * 100:.1f}% of equity")

    # Slippage
    slippage_bps = _estimate_slippage_bps(size_usd)
    slippage_mult = slippage_bps / 10000
    if side == "long":
        expected_entry = entry * (1 + slippage_mult)
    else:
        expected_entry = entry * (1 - slippage_mult)

    # Risk / reward
    risk_distance = abs(expected_entry - req.stop)
    reward_distance = abs(req.target - expected_entry)
    risk_pct = (risk_distance / expected_entry) * 100 if expected_entry else 0
    reward_pct = (reward_distance / expected_entry) * 100 if expected_entry else 0
    risk_usd = size_usd * (risk_pct / 100)
    reward_usd = size_usd * (reward_pct / 100)
    rr = reward_usd / max(risk_usd, 0.01)

    # Fees (entry + exit, both market orders)
    fees_usd = size_usd * (FEE_BPS / 10000) * 2

    # Validation: SL direction
    if side == "long" and req.stop >= expected_entry:
        errors.append(f"Stop {req.stop} must be below entry {expected_entry:.4f} for long")
    if side == "short" and req.stop <= expected_entry:
        errors.append(f"Stop {req.stop} must be above entry {expected_entry:.4f} for short")
    if side == "long" and req.target <= expected_entry:
        errors.append(f"Target {req.target} must be above entry for long")
    if side == "short" and req.target >= expected_entry:
        errors.append(f"Target {req.target} must be below entry for short")

    # Risk limit checks
    size_pct = (size_usd / max(s.equity, 1)) * 100
    if size_pct > r.max_position_pct * 100:
        warnings.append(f"Size {size_pct:.2f}% exceeds max_position_pct {r.max_position_pct * 100:.1f}%")

    current_exposure = sum(abs(p.size) for p in s.positions.values())
    new_exposure_pct = ((current_exposure + size_usd) / max(s.equity, 1)) * 100
    if new_exposure_pct > r.max_total_exposure_pct * 100:
        warnings.append(f"Total exposure after fill {new_exposure_pct:.1f}% exceeds limit")

    if len(s.positions) >= r.max_positions:
        errors.append(f"Already at max_positions ({r.max_positions})")

    if risk_pct > 10:
        warnings.append(f"Stop distance {risk_pct:.1f}% is very wide (>10%)")
    if risk_pct < 0.3:
        warnings.append(f"Stop distance {risk_pct:.2f}% is very tight (<0.3%)")

    return {
        "ok": len(errors) == 0,
        "symbol": req.symbol,
        "side": side,
        "expected_entry": round(expected_entry, 6),
        "hint_price": round(entry, 6),
        "slippage_bps": round(slippage_bps, 1),
        "size_usd": round(size_usd, 2),
        "size_pct_of_equity": round(size_pct, 2),
        "stop": req.stop,
        "target": req.target,
        "risk_usd": round(risk_usd, 2),
        "risk_pct": round(risk_pct, 3),
        "reward_usd": round(reward_usd, 2),
        "reward_pct": round(reward_pct, 3),
        "rr_estimate": round(rr, 2),
        "fees_usd": round(fees_usd, 2),
        "expected_net_profit_if_tp": round(reward_usd - fees_usd, 2),
        "expected_net_loss_if_sl": round(-(risk_usd + fees_usd), 2),
        "exposure_after_fill_pct": round(new_exposure_pct, 2),
        "exposure_limit_pct": round(r.max_total_exposure_pct * 100, 2),
        "errors": errors,
        "warnings": warnings,
    }


@router.get("/exposure-breakdown")
async def api_exposure_breakdown():
    """Per-symbol and per-side exposure breakdown."""
    agent = get_agent()
    s = agent.trader.state
    total_exposure = sum(abs(p.size) for p in s.positions.values()) or 1
    items = []
    for sym, pos in s.positions.items():
        items.append({
            "symbol": sym,
            "side": pos.side,
            "size_usd": round(abs(pos.size), 2),
            "entry_price": pos.entry_price,
            "unrealized_pnl": round(pos.unrealized_pnl, 2),
            "exposure_pct": round(abs(pos.size) / total_exposure * 100, 2),
            "portfolio_weight": round(abs(pos.size) / max(s.equity, 1) * 100, 2),
        })
    return {
        "total_exposure_usd": round(total_exposure if s.positions else 0, 2),
        "total_exposure_pct": round(total_exposure / max(s.equity, 1) * 100, 2),
        "positions": items,
    }
