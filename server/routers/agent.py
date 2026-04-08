"""
Agent routes: lifecycle (start/stop/revive), config, strategy params/presets,
signals, audit log, lessons.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from ..core.config import PROJECT_ROOT
from ..core.dependencies import get_agent

router = APIRouter(prefix="/api/agent", tags=["agent"])

STRATEGY_PRESETS_FILE = PROJECT_ROOT / "strategy_presets.json"


# ── Preset helpers ───────────────────────────────────────────────────────

def _load_presets() -> dict:
    if STRATEGY_PRESETS_FILE.exists():
        try:
            return json.loads(STRATEGY_PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "V6 Default": {
            "ma5_len": 5, "ma8_len": 8, "ema21_len": 21, "ma55_len": 55,
            "bb_length": 21, "bb_std_dev": 2.5,
            "dist_ma5_ma8": 1.5, "dist_ma8_ema21": 2.5, "dist_ema21_ma55": 4.0,
            "slope_len": 3, "slope_threshold": 0.1, "atr_period": 14,
        },
        "Momentum": {
            "ma5_len": 3, "ma8_len": 6, "ema21_len": 15, "ma55_len": 40,
            "bb_length": 18, "bb_std_dev": 2.0,
            "dist_ma5_ma8": 2.0, "dist_ma8_ema21": 3.5, "dist_ema21_ma55": 5.0,
            "slope_len": 2, "slope_threshold": 0.15, "atr_period": 10,
        },
        "Conservative": {
            "ma5_len": 5, "ma8_len": 10, "ema21_len": 25, "ma55_len": 60,
            "bb_length": 25, "bb_std_dev": 3.0,
            "dist_ma5_ma8": 1.0, "dist_ma8_ema21": 2.0, "dist_ema21_ma55": 3.0,
            "slope_len": 4, "slope_threshold": 0.05, "atr_period": 18,
        },
        "Scalper": {
            "ma5_len": 3, "ma8_len": 5, "ema21_len": 13, "ma55_len": 34,
            "bb_length": 15, "bb_std_dev": 2.0,
            "dist_ma5_ma8": 2.5, "dist_ma8_ema21": 4.0, "dist_ema21_ma55": 6.0,
            "slope_len": 2, "slope_threshold": 0.2, "atr_period": 7,
        },
    }


def _save_presets(presets: dict):
    STRATEGY_PRESETS_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")


# ── Pydantic models ──────────────────────────────────────────────────────

class StrategyConfigRequest(BaseModel):
    timeframe: str | None = None
    symbols: list[str] | None = None
    top_volume: int | None = None
    tick_interval: int | None = None
    max_position_pct: float | None = None
    max_positions: int | None = None


# ── Lifecycle ────────────────────────────────────────────────────────────

@router.get("/status")
async def api_agent_status():
    """Get current agent status: equity, positions, trades, generation, etc."""
    return get_agent().get_status()


@router.get("/summary")
async def api_agent_summary():
    """
    Condensed runtime summary for Execution Overview / Glassbox.
    Returns human-readable explanations, not raw state.
    """
    import time
    agent = get_agent()
    s = agent.trader.state
    from ..agent_brain import WATCH_SYMBOLS, TICK_INTERVAL_SEC

    signals = agent._last_signals or {}
    blocked_signals = [sig for sig in signals.values() if sig.get("blocked")]
    pending = [sig for sig in signals.values() if not sig.get("blocked") and sig.get("action") in ("long", "short")]

    last_scan_at = max(
        (sig.get("_ts", 0) for sig in signals.values()),
        default=0,
    )
    now = time.time()
    age_sec = int(now - last_scan_at) if last_scan_at else None
    next_scan_eta = max(0, TICK_INTERVAL_SEC - (age_sec or 0)) if last_scan_at else TICK_INTERVAL_SEC

    # Determine last action + last block reason
    last_action = None
    if s.trade_history:
        last_t = s.trade_history[-1]
        last_action = {
            "symbol": last_t.symbol,
            "side": last_t.side,
            "pnl_pct": round(last_t.pnl_pct, 2),
            "reason": last_t.reason,
        }
    last_block_reason = None
    if blocked_signals:
        b = blocked_signals[-1]
        last_block_reason = "; ".join(b.get("block_reasons", [])[:2])

    # Health score: simple composite
    health = 100
    if not s.is_alive: health -= 50
    if not agent._running: health -= 20
    if s.daily_pnl < 0 and abs(s.daily_pnl) > s.equity * 0.01: health -= 10
    if (s.peak_equity - s.equity) / max(s.peak_equity, 1) > 0.03: health -= 10
    health = max(0, min(100, health))

    return {
        "mode": s.mode,
        "runtime_state": "RUNNING" if agent._running else ("SHUTDOWN" if not s.is_alive else "STOPPED"),
        "current_phase": agent._cycle_phase,
        "current_regime": agent.lessons.market_regime,
        "regime_confidence": round(agent.lessons.regime_confidence, 2),
        "watch_symbols_count": len(WATCH_SYMBOLS),
        "active_positions_count": len(s.positions),
        "pending_signals_count": len(pending),
        "blocked_signals_count": len(blocked_signals),
        "last_scan_age_sec": age_sec,
        "next_scan_eta_sec": int(next_scan_eta),
        "last_action": last_action,
        "last_block_reason": last_block_reason,
        "equity": round(s.equity, 2),
        "daily_pnl": round(s.daily_pnl, 2),
        "total_trades": s.total_trades,
        "win_rate": round(s.win_count / max(s.total_trades, 1) * 100, 1),
        "generation": s.generation,
        "health_score": health,
    }


@router.get("/risk-state")
async def api_agent_risk_state():
    """
    Current risk snapshot with state machine label, meters, and block reasons.
    Used by v2 Risk sub-tab + Decision Rail Risk Gate card.
    """
    agent = get_agent()
    s = agent.trader.state
    r = agent.trader.risk

    # Current usage (pct)
    exposure_usd = sum(abs(p.size) for p in s.positions.values())
    exposure_pct = (exposure_usd / max(s.equity, 1)) * 100
    daily_loss_pct = max(0, -s.daily_pnl / max(s.equity, 1) * 100)
    dd_pct = max(0, (s.peak_equity - s.equity) / max(s.peak_equity, 1) * 100)
    positions_used = len(s.positions)

    # Convert limits decimals to percentages
    max_pos_pct = r.max_position_pct * 100
    max_exp_pct = r.max_total_exposure_pct * 100
    max_daily_loss = r.max_daily_loss_pct * 100
    max_dd = r.max_drawdown_pct * 100

    # State machine: NORMAL / WATCH / COOLDOWN / HALTED
    state = "NORMAL"
    reason = None
    if not s.is_alive:
        state = "HALTED"
        reason = s.shutdown_reason or "Emergency shutdown"
    elif dd_pct >= max_dd * 0.8:
        state = "WATCH"
        reason = f"Drawdown {dd_pct:.1f}% approaching limit {max_dd:.1f}%"
    elif daily_loss_pct >= max_daily_loss * 0.7:
        state = "WATCH"
        reason = f"Daily loss {daily_loss_pct:.2f}% approaching limit {max_daily_loss:.2f}%"

    # Cooldown check
    import time
    now = time.time()
    cooldown_remaining = 0
    for sym, last_ts in s.last_trade_time.items():
        remaining = r.cooldown_seconds - (now - last_ts)
        if remaining > cooldown_remaining:
            cooldown_remaining = remaining
    if cooldown_remaining > 0 and state == "NORMAL":
        state = "COOLDOWN"
        reason = f"Cooldown {int(cooldown_remaining)}s remaining"

    # Consecutive loss check
    recent = s.trade_history[-3:]
    consec_losses = 0
    for t in reversed(recent):
        if t.pnl_usd < 0: consec_losses += 1
        else: break

    return {
        "state": state,
        "state_reason": reason,
        "meters": {
            "exposure": {"current": round(exposure_pct, 2), "max": round(max_exp_pct, 2), "unit": "%"},
            "daily_loss": {"current": round(daily_loss_pct, 2), "max": round(max_daily_loss, 2), "unit": "%"},
            "drawdown": {"current": round(dd_pct, 2), "max": round(max_dd, 2), "unit": "%"},
            "positions": {"current": positions_used, "max": r.max_positions, "unit": ""},
        },
        "max_position_pct": round(max_pos_pct, 2),
        "cooldown_remaining_sec": int(max(0, cooldown_remaining)),
        "consecutive_losses": consec_losses,
        "is_alive": s.is_alive,
        "kill_switch_armed": not s.is_alive,
    }


@router.get("/risk-blocks/recent")
async def api_agent_risk_blocks_recent(limit: int = 20):
    """Return recent signals that were blocked, with reasons."""
    agent = get_agent()
    blocks = []
    for symbol, sig in agent._last_signals.items():
        if sig.get("blocked"):
            blocks.append({
                "symbol": symbol,
                "side": sig.get("action"),
                "confidence": sig.get("confidence"),
                "block_reasons": sig.get("block_reasons", []),
                "reason_codes": [r.split(":")[0].strip() for r in sig.get("block_reasons", [])],
                "market_regime": sig.get("market_regime"),
                "price": sig.get("price"),
                "sl": sig.get("sl"),
                "tp": sig.get("tp"),
            })
    return {"count": len(blocks), "blocks": blocks[:limit]}


@router.get("/signal-candidates")
async def api_agent_signal_candidates():
    """
    Standardized signal candidate pool with scores, states, and risk.
    Consumed by Execution → Execution sub-tab and Decision Rail Trade Candidate card.
    """
    agent = get_agent()
    s = agent.trader.state
    candidates = []

    for symbol, sig in agent._last_signals.items():
        if not sig or sig.get("action") not in ("long", "short"):
            continue

        price = sig.get("price", 0)
        sl = sig.get("sl", 0)
        tp = sig.get("tp", 0)
        confidence = sig.get("confidence", 0)

        # Setup score = confidence * 100
        setup_score = round(confidence * 100)

        # Execution score: penalize blocked, low SL distance
        exec_score = 100 if not sig.get("blocked") else 0
        if price and sl:
            sl_pct = abs(price - sl) / price * 100
            if sl_pct < 0.5 or sl_pct > 8:
                exec_score -= 20

        # RR estimate
        rr = None
        if price and sl and tp:
            risk = abs(price - sl)
            reward = abs(tp - price)
            if risk > 0:
                rr = round(reward / risk, 2)

        # Risk state (uses current agent state)
        risk_state = "OK"
        if sig.get("blocked"):
            risk_state = "BLOCKED"
        elif symbol in s.positions:
            risk_state = "POSITION_EXISTS"

        candidates.append({
            "symbol": symbol,
            "side": sig.get("action"),
            "signal_state": "BLOCKED" if sig.get("blocked") else "READY",
            "setup_score": setup_score,
            "execution_score": max(0, exec_score),
            "confidence": confidence,
            "risk_state": risk_state,
            "block_reason": "; ".join(sig.get("block_reasons", [])[:2]) or None,
            "entry": price,
            "stop": sl,
            "target": tp,
            "rr_estimate": rr,
            "reason": sig.get("reason"),
            "market_regime": sig.get("market_regime"),
            "source_strategy": "V6",
        })

    # Sort by setup_score descending
    candidates.sort(key=lambda x: -x["setup_score"])
    return {"count": len(candidates), "candidates": candidates}


@router.post("/start")
async def api_agent_start():
    """Start the agent background loop."""
    agent = get_agent()
    if agent._running:
        return {"ok": True, "message": "Agent already running"}
    agent.start()
    return {"ok": True, "message": "Agent started"}


@router.post("/stop")
async def api_agent_stop():
    """Stop the agent background loop."""
    agent = get_agent()
    agent.stop()
    return {"ok": True, "message": "Agent stopped"}


@router.post("/revive")
async def api_agent_revive():
    """Revive the agent after emergency shutdown."""
    agent = get_agent()
    agent.trader.revive()
    agent._save_state()
    return {"ok": True, "message": "Agent revived", "equity": agent.trader.state.equity}


@router.post("/config")
async def api_agent_config(
    request: Request,
    mode: str | None = Query(None, description="paper or live"),
    equity: float | None = Query(None, description="Set paper equity"),
):
    """Update agent config (mode, equity). Accepts both query params and JSON body."""
    agent = get_agent()
    body_mode, body_equity = None, None
    try:
        body = await request.json()
        body_mode = body.get("mode")
        body_equity = body.get("equity")
    except Exception:
        pass

    final_mode = body_mode or mode
    final_equity = body_equity or equity

    if final_mode and final_mode in ("paper", "live"):
        if final_mode == "live" and not agent.trader.has_api_keys():
            return {"ok": False, "reason": "Cannot switch to live: OKX API keys not configured. Set keys first via /api/agent/okx-keys"}
        agent.trader.state.mode = final_mode
        print(f"[Agent] Mode switched to {final_mode.upper()}")
    if final_equity is not None and final_equity > 0:
        agent.trader.state.equity = float(final_equity)
        agent.trader.state.peak_equity = max(agent.trader.state.peak_equity, float(final_equity))
        agent.trader.state.cash = float(final_equity)
    agent._save_state()
    return {"ok": True, "state": agent.get_status()}


# ── Strategy config ──────────────────────────────────────────────────────

@router.post("/strategy-config")
async def api_agent_strategy_config(req: StrategyConfigRequest):
    """Update strategy runtime config (timeframe, symbols, tick interval, risk)."""
    from .. import agent_brain
    agent = get_agent()
    changes = []

    if req.timeframe and req.timeframe in ("5m", "15m", "1h", "4h", "1d"):
        agent_brain.SIGNAL_INTERVAL = req.timeframe
        changes.append(f"timeframe={req.timeframe}")

    if req.top_volume and 1 <= req.top_volume <= 50:
        from ..data_service import get_top_volume_symbols
        top_syms = await get_top_volume_symbols(req.top_volume)
        if top_syms:
            agent_brain.WATCH_SYMBOLS = top_syms
            changes.append(f"top_{req.top_volume}_vol={top_syms[:5]}...")
    elif req.symbols and len(req.symbols) > 0:
        clean = [s.upper().replace("/", "").strip() for s in req.symbols if s.strip()]
        if clean:
            agent_brain.WATCH_SYMBOLS = clean
            changes.append(f"symbols={clean}")

    if req.tick_interval and 10 <= req.tick_interval <= 600:
        agent_brain.TICK_INTERVAL_SEC = req.tick_interval
        changes.append(f"tick={req.tick_interval}s")

    if req.max_position_pct and 0.5 <= req.max_position_pct <= 25:
        agent.trader.risk.max_position_pct = req.max_position_pct / 100.0
        changes.append(f"pos_pct={req.max_position_pct}%")

    if req.max_positions and 1 <= req.max_positions <= 10:
        agent.trader.risk.max_positions = req.max_positions
        changes.append(f"max_pos={req.max_positions}")

    if changes:
        print(f"[Agent] Config updated: {', '.join(changes)}")
        return {"ok": True, "changes": changes, "watch_symbols": agent_brain.WATCH_SYMBOLS}
    return {"ok": True, "changes": [], "message": "No changes"}


@router.post("/strategy-params")
async def api_agent_strategy_params(request: Request):
    """Update V6 strategy parameters. Accepts partial dict of param key:value."""
    agent = get_agent()
    try:
        req = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON body"}
    params = agent.trader.state.strategy_params
    changes = []
    valid_keys = set(params.keys())

    for key, value in req.items():
        if key not in valid_keys:
            continue
        try:
            if isinstance(params[key], int):
                params[key] = int(value)
            else:
                params[key] = float(value)
            changes.append(f"{key}={params[key]}")
        except (ValueError, TypeError):
            pass

    if changes:
        agent._save_state()
        print(f"[Agent] Strategy params updated: {', '.join(changes)}")
    return {"ok": True, "changes": changes, "params": params}


# ── Presets ───────────────────────────────────────────────────────────────

@router.get("/strategy-presets")
async def api_get_presets():
    """List all saved strategy presets."""
    return {"presets": _load_presets()}


@router.post("/strategy-presets/save")
async def api_save_preset(request: Request):
    """Save current strategy params as a named preset."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON body"}
    name = body.get("name", "").strip()
    if not name:
        return {"ok": False, "reason": "Preset name is required"}
    agent = get_agent()
    presets = _load_presets()
    presets[name] = dict(agent.trader.state.strategy_params)
    _save_presets(presets)
    print(f"[Agent] Strategy preset saved: {name}")
    return {"ok": True, "name": name, "presets": presets}


@router.post("/strategy-presets/load")
async def api_load_preset(request: Request):
    """Load a named preset into the active strategy."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON body"}
    name = body.get("name", "").strip()
    presets = _load_presets()
    if name not in presets:
        return {"ok": False, "reason": f"Preset '{name}' not found"}
    agent = get_agent()
    agent.trader.state.strategy_params.update(presets[name])
    agent._save_state()
    print(f"[Agent] Loaded strategy preset: {name}")
    return {"ok": True, "name": name, "params": agent.trader.state.strategy_params}


@router.post("/strategy-presets/delete")
async def api_delete_preset(request: Request):
    """Delete a named preset."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON body"}
    name = body.get("name", "").strip()
    presets = _load_presets()
    if name in presets:
        del presets[name]
        _save_presets(presets)
        return {"ok": True, "name": name}
    return {"ok": False, "reason": f"Preset '{name}' not found"}


# ── Signals, audit, lessons ──────────────────────────────────────────────

@router.get("/signals")
async def api_agent_signals():
    """Run signal check on all watched symbols and return current signals."""
    agent = get_agent()
    signals = {}
    for symbol in agent._last_signals:
        signals[symbol] = agent._last_signals[symbol]
    from ..agent_brain import WATCH_SYMBOLS

    async def _gen(sym: str):
        try:
            return sym, await agent.generate_signal(sym)
        except Exception as e:
            logging.warning(f"Signal generation failed for {sym}: {e}")
            return sym, None

    results = await asyncio.gather(*[_gen(sym) for sym in WATCH_SYMBOLS])
    for sym, sig in results:
        if sig:
            signals[sym] = sig
    return {"signals": signals}


@router.get("/audit-log")
async def api_agent_audit_log(limit: int = Query(50, ge=1, le=500)):
    """Get recent entries from the trade audit log."""
    from ..agent_brain import TRADE_AUDIT_LOG
    if not TRADE_AUDIT_LOG.exists():
        return {"entries": []}
    try:
        raw = TRADE_AUDIT_LOG.read_text(encoding="utf-8").strip()
        if not raw:
            return {"entries": []}
        lines = raw.split("\n")
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return {"entries": entries}
    except Exception as e:
        logging.warning(f"Failed to read trade log: {e}")
        return {"entries": []}


@router.get("/lessons")
async def api_agent_lessons():
    """Get the agent's lessons ledger."""
    agent = get_agent()
    return agent.lessons.get_summary()
