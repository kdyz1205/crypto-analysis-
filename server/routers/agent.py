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
