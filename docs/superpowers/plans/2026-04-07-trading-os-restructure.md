# Trading OS Restructure — Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the monolithic crypto trading dashboard into a role-separated, event-driven Trading Operating System with 4 layers: Market Workbench, Execution Center, Research Lab, Control Bus.

**Architecture:** Backend splits from a single `app.py` into 8 domain routers. Frontend restructures from 6 equal-weight tabs into a layered workspace with progressive disclosure. An event bus connects signal lifecycle (DETECTED -> FILLED -> CLOSED) across all modules. The Telegram bot becomes an event subscriber, not a UI replacement.

**Tech Stack:** Python/FastAPI (backend), Vanilla JS + lightweight-charts (frontend), httpx (async HTTP), existing OKX/Anthropic integrations.

---

## Phasing Strategy

This plan is split into 4 independent phases. Each phase produces working software and can be shipped alone.

| Phase | Name | Risk | Impact | Estimated Tasks |
|-------|------|------|--------|-----------------|
| 1 | Backend Router Split | Low (refactor only) | Foundation for all future work | 10 |
| 2 | Frontend 4-Layer Restructure | Medium (UI rewrite) | Solves cognitive friction | 15 |
| 3 | Event Bus + State Machine | Medium (new infra) | Enables glassbox + bot integration | 8 |
| 4 | UX Polish (Command Palette, Combat Mode, Glassbox) | Low (additive) | Professional trading experience | 10 |

**Execution order:** Phase 1 -> Phase 2 -> Phase 3 -> Phase 4. Each phase is a separate plan document.

---

## Phase 1: Backend Router Split

> Detailed plan below. Lowest risk, highest leverage.

## Phase 2: Frontend 4-Layer Restructure

> Separate plan: `2026-04-07-frontend-4-layer.md`

## Phase 3: Event Bus + State Machine

> Separate plan: `2026-04-07-event-bus.md`

## Phase 4: UX Polish

> Separate plan: `2026-04-07-ux-polish.md`

---

# Phase 1: Backend Router Split

**Goal:** Split `server/app.py` (1495 lines, 59 routes) into 8 domain routers without changing any behavior or API contracts.

**Architecture:** Each domain becomes a FastAPI `APIRouter` in its own file. `app.py` shrinks to ~80 lines: app creation, middleware, singleton lifecycle, router registration. All route paths, request/response formats, and behavior stay identical.

**Tech Stack:** FastAPI APIRouter, existing modules unchanged.

---

### File Structure

```
server/
  app.py                    # MODIFY: Slim down to app shell + router registration (~80 lines)
  routers/
    __init__.py             # CREATE: Empty
    market_data.py          # CREATE: /api/symbols, /api/symbol-info, /api/chart, /api/ohlcv, /api/top-volume, /api/data-info
    research.py             # CREATE: /api/backtest, /api/backtest/optimize, /api/patterns, /api/pattern-stats/*, /api/ma-ribbon*
    signal.py               # CREATE: /api/agent/strategy-*, /api/agent/signals, /api/agent/audit-log, /api/agent/lessons
    execution.py            # CREATE: /api/agent/status, /api/agent/start, /api/agent/stop, /api/agent/revive, /api/agent/config, /api/agent/okx-*
    risk.py                 # CREATE: /api/agent/risk-limits
    ops.py                  # CREATE: /api/health, /api/agent/telegram-config, /api/agent/logs, /api/healer/*, static files
    chat.py                 # CREATE: /api/chat, /api/chat/models, /api/chat/history, /api/chat/clear
    onchain.py              # CREATE: /api/onchain/* (all 13 proxy routes)
  agent_brain.py            # NO CHANGE
  okx_trader.py             # NO CHANGE
  data_service.py           # NO CHANGE
  backtest_service.py       # NO CHANGE
  pattern_service.py        # NO CHANGE
  pattern_features.py       # NO CHANGE
  ma_ribbon_service.py      # NO CHANGE
  ai_chat.py                # NO CHANGE
  self_healer.py            # NO CHANGE
```

---

### Task 1: Create router directory and shared dependencies

**Files:**
- Create: `server/routers/__init__.py`
- Create: `server/routers/_deps.py`

The shared deps module provides singleton getters so all routers access the same AgentBrain, AIChatEngine, and SelfHealer instances without circular imports.

- [ ] **Step 1: Create router package**

```python
# server/routers/__init__.py
```

- [ ] **Step 2: Create shared dependencies module**

```python
# server/routers/_deps.py
"""
Shared singleton accessors for all routers.
Singletons are initialized once by app.py startup event.
Routers import getters from here — never instantiate directly.
"""

from ..agent_brain import AgentBrain
from ..ai_chat import AIChatEngine
from ..self_healer import SelfHealer

_agent: AgentBrain | None = None
_chat: AIChatEngine | None = None
_healer: SelfHealer | None = None


def init_agent() -> AgentBrain:
    global _agent
    if _agent is None:
        _agent = AgentBrain()
    return _agent


def init_chat() -> AIChatEngine:
    global _chat
    if _chat is None:
        _chat = AIChatEngine()
    return _chat


def init_healer() -> SelfHealer:
    global _healer
    if _healer is None:
        _healer = SelfHealer()
    return _healer


def get_agent() -> AgentBrain:
    return init_agent()


def get_chat() -> AIChatEngine:
    return init_chat()


def get_healer() -> SelfHealer:
    return init_healer()
```

- [ ] **Step 3: Commit**

```bash
git add server/routers/__init__.py server/routers/_deps.py
git commit -m "feat: create router package with shared singleton deps"
```

---

### Task 2: Extract market_data router

**Files:**
- Create: `server/routers/market_data.py`
- Reference: `server/app.py:206-340` (symbols, symbol-info, chart, ohlcv) + `server/app.py:1212-1240` (top-volume, data-info)

- [ ] **Step 1: Create market_data router**

Move the 6 market data routes. All imports come from `data_service`. No singleton dependencies.

```python
# server/routers/market_data.py
"""Market data routes: symbols, OHLCV, chart overlays."""

from fastapi import APIRouter, Query
from ..data_service import (
    load_symbols, load_okx_swap_symbols, get_ohlcv, get_ohlcv_with_df, API_ONLY,
)
from ..pattern_service import get_patterns, DEFAULT_PARAMS
from ..backtest_service import _sma, _ema, _atr, _bb_upper

router = APIRouter(prefix="/api", tags=["market_data"])


@router.get("/symbols")
async def get_symbols_route(source: str = Query("csv")):
    if source == "okx":
        return await load_okx_swap_symbols()
    return load_symbols()


@router.get("/symbol-info")
async def get_symbol_info(symbol: str = Query(...)):
    try:
        from ..data_service import get_ohlcv_with_df
        df, _ = await get_ohlcv_with_df(symbol, "1d", days=7)
        if df is None or df.is_empty():
            return {"symbol": symbol, "error": "No data"}
        close = df["close"].to_list()
        vol = df["volume"].to_list() if "volume" in df.columns else []
        last_price = close[-1] if close else 0
        prev_price = close[-2] if len(close) > 1 else last_price
        change_pct = ((last_price - prev_price) / prev_price * 100) if prev_price else 0
        avg_vol = sum(vol[-7:]) / max(len(vol[-7:]), 1) if vol else 0
        return {
            "symbol": symbol,
            "price": last_price,
            "change_pct": round(change_pct, 2),
            "avg_volume_7d": round(avg_vol, 2),
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


@router.get("/chart")
async def api_chart(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    sr: bool = Query(True),
    max_lines: int = Query(8),
    mode: str = Query("recognize"),
):
    ohlcv = await get_ohlcv(symbol, interval, days=days)
    result = {"ohlcv": ohlcv}
    if sr:
        params = dict(DEFAULT_PARAMS)
        params["max_lines"] = max_lines
        patterns = get_patterns(symbol, interval, days=days, params=params, mode=mode)
        result["patterns"] = patterns
    return result


@router.get("/ohlcv")
async def api_ohlcv(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    end: int | None = Query(None),
):
    kw = {}
    if end:
        kw["end_ts"] = end
    ohlcv = await get_ohlcv(symbol, interval, days=days, **kw)
    return {"ohlcv": ohlcv, "symbol": symbol, "interval": interval}


@router.get("/top-volume")
async def api_top_volume(limit: int = Query(20, ge=1, le=100)):
    syms = await load_okx_swap_symbols()
    return {"symbols": syms[:limit]}


@router.get("/data-info")
async def api_data_info(symbol: str = Query("BTCUSDT")):
    from ..data_service import _find_csv
    info = {}
    for iv in ["5m", "15m", "1h", "4h", "1d"]:
        path = _find_csv(symbol, iv)
        if path:
            import os
            stat = os.stat(path)
            info[iv] = {"path": str(path), "size_kb": round(stat.st_size / 1024, 1)}
    return {"symbol": symbol, "intervals": info}
```

- [ ] **Step 2: Verify routes match original paths**

Confirm: `/api/symbols`, `/api/symbol-info`, `/api/chart`, `/api/ohlcv`, `/api/top-volume`, `/api/data-info` — all match.

- [ ] **Step 3: Commit**

```bash
git add server/routers/market_data.py
git commit -m "feat: extract market_data router (6 routes)"
```

---

### Task 3: Extract research router

**Files:**
- Create: `server/routers/research.py`
- Reference: `server/app.py:343-760` (backtest, optimize, patterns, pattern-stats/*, ma-ribbon*)

- [ ] **Step 1: Create research router**

Move the 9 research routes. These depend on backtest_service, pattern_service, pattern_features, ma_ribbon_service, and data_service.

```python
# server/routers/research.py
"""Research routes: backtest, optimize, patterns, MA ribbon."""

import traceback
from fastapi import APIRouter, Query, Request

from ..data_service import get_ohlcv_with_df
from ..pattern_service import get_patterns, get_patterns_from_df, DEFAULT_PARAMS
from ..backtest_service import run_backtest, load_df_from_csv, BacktestParams, optimize_backtest
from ..ma_ribbon_service import get_current_ribbon, run_ribbon_backtest, RibbonBacktestConfig
from ..pattern_features import (
    run_trendline_backtest, extract_features, similarity,
    is_same_class, current_vs_history, TrendLineFeatures,
    PatternBacktestConfig, user_line_to_features, _signal_to_features,
)

router = APIRouter(prefix="/api", tags=["research"])


def _load_df_for_analysis(symbol: str, interval: str, days: int):
    """Load DataFrame: try CSV first, return (df, source)."""
    df = load_df_from_csv(symbol, interval)
    if df is not None and len(df) > 0:
        return df, "csv"
    return None, None


@router.get("/backtest")
async def api_backtest(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    mfi_period: int = Query(14),
    mfi_upper: float = Query(80),
    mfi_lower: float = Query(20),
    ma_fast: int = Query(10),
    ma_slow: int = Query(30),
    atr_period: int = Query(14),
    atr_mult: float = Query(2.0),
    rsi_period: int = Query(14),
    rsi_upper: float = Query(70),
    rsi_lower: float = Query(30),
):
    try:
        df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
        if df is None:
            return {"error": "No data"}
        params = BacktestParams(
            mfi_period=mfi_period, mfi_upper=mfi_upper, mfi_lower=mfi_lower,
            ma_fast=ma_fast, ma_slow=ma_slow,
            atr_period=atr_period, atr_mult=atr_mult,
            rsi_period=rsi_period, rsi_upper=rsi_upper, rsi_lower=rsi_lower,
        )
        result = run_backtest(df, params)
        return result
    except Exception as e:
        return {"error": str(e)}


@router.post("/backtest/optimize")
async def api_backtest_optimize(request: Request):
    try:
        body = await request.json()
        symbol = body.get("symbol", "BTCUSDT")
        interval = body.get("interval", "4h")
        days = body.get("days", 365)
        df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
        if df is None:
            return {"error": "No data"}
        result = optimize_backtest(df, iterations=body.get("iterations", 50))
        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/patterns")
async def api_patterns(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    max_lines: int = Query(8),
    mode: str = Query("recognize"),
    end: int | None = Query(None),
):
    try:
        params = dict(DEFAULT_PARAMS)
        params["max_lines"] = max_lines
        kw = {}
        if end:
            kw["end_ts"] = end
        patterns = get_patterns(symbol, interval, days=days, params=params, mode=mode, **kw)
        return patterns
    except Exception as e:
        return {"error": str(e), "supports": [], "resists": [], "trendlines": [], "zones": []}


@router.get("/pattern-stats/backtest")
async def api_pattern_stats_backtest(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    line_price: float = Query(...),
    line_type: str = Query("support"),
    slope: float = Query(0.0),
    start_idx: int = Query(0),
    end_idx: int = Query(0),
):
    try:
        df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
        if df is None:
            return {"error": "No data"}
        config = PatternBacktestConfig(
            target_price=line_price,
            pattern_type=line_type,
            slope=slope,
            start_bar=start_idx,
            end_bar=end_idx,
        )
        result = run_trendline_backtest(df, config)
        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/pattern-stats/features")
async def api_pattern_stats_features(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    line_price: float = Query(...),
    line_type: str = Query("support"),
    slope: float = Query(0.0),
    start_idx: int = Query(0),
    end_idx: int = Query(0),
    start_time: int | None = Query(None),
    end_time: int | None = Query(None),
):
    try:
        df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
        if df is None:
            return {"error": "No data"}
        feats = user_line_to_features(
            df, line_price, line_type, slope,
            start_idx, end_idx,
            start_time=start_time, end_time=end_time,
        )
        return feats.to_dict() if feats else {"error": "Could not extract features"}
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.get("/pattern-stats/current-vs-history")
async def api_pattern_stats_current_vs_history(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    line_price: float = Query(...),
    line_type: str = Query("support"),
    slope: float = Query(0.0),
    start_idx: int = Query(0),
    end_idx: int = Query(0),
    start_time: int | None = Query(None),
    end_time: int | None = Query(None),
    max_lines: int = Query(8),
    mode: str = Query("recognize"),
):
    try:
        df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
        if df is None:
            return {"error": "No data"}
        result = current_vs_history(
            df=df, symbol=symbol, interval=interval, days=days,
            line_price=line_price, line_type=line_type,
            slope=slope, start_idx=start_idx, end_idx=end_idx,
            start_time=start_time, end_time=end_time,
            max_lines=max_lines, mode=mode,
        )
        return result
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.get("/pattern-stats/line-similar")
async def api_pattern_stats_line_similar(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    line_price: float = Query(...),
    line_type: str = Query("support"),
    slope: float = Query(0.0),
    start_idx: int = Query(0),
    end_idx: int = Query(0),
    start_time: int | None = Query(None),
    end_time: int | None = Query(None),
    max_lines: int = Query(8),
    mode: str = Query("recognize"),
    top_n: int = Query(5),
):
    try:
        df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
        if df is None:
            return {"error": "No data"}
        user_feats = user_line_to_features(
            df, line_price, line_type, slope,
            start_idx, end_idx,
            start_time=start_time, end_time=end_time,
        )
        if user_feats is None:
            return {"similar": [], "error": "Cannot extract features for this line"}

        params = dict(DEFAULT_PARAMS)
        params["max_lines"] = max_lines
        patterns = get_patterns_from_df(df, symbol, interval, params=params, mode=mode)

        all_lines = []
        for kind in ("supports", "resists", "trendlines"):
            for line in patterns.get(kind, []):
                all_lines.append(line)

        scored = []
        for line in all_lines:
            sig_feats = _signal_to_features(line, df)
            if sig_feats is None:
                continue
            if not is_same_class(user_feats, sig_feats):
                continue
            sim = similarity(user_feats, sig_feats)
            scored.append({**line, "_similarity": round(sim, 4)})

        scored.sort(key=lambda x: x["_similarity"], reverse=True)
        return {"similar": scored[:top_n], "query_features": user_feats.to_dict()}
    except Exception as e:
        return {"error": str(e), "similar": []}


@router.get("/ma-ribbon")
async def api_ma_ribbon(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
):
    try:
        df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
        if df is None:
            return {"error": "No data"}
        return get_current_ribbon(df)
    except Exception as e:
        return {"error": str(e)}


@router.get("/ma-ribbon/backtest")
async def api_ma_ribbon_backtest(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
):
    try:
        df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
        if df is None:
            return {"error": "No data"}
        config = RibbonBacktestConfig()
        return run_ribbon_backtest(df, config)
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 2: Commit**

```bash
git add server/routers/research.py
git commit -m "feat: extract research router (9 routes: backtest, patterns, ma-ribbon)"
```

---

### Task 4: Extract execution router

**Files:**
- Create: `server/routers/execution.py`
- Reference: `server/app.py:801-900` (agent status/start/stop/revive/config, okx-keys, okx-status)

- [ ] **Step 1: Create execution router**

```python
# server/routers/execution.py
"""Execution routes: agent lifecycle, OKX connectivity, position management."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from ._deps import get_agent

router = APIRouter(prefix="/api/agent", tags=["execution"])


class OKXKeysRequest(BaseModel):
    api_key: str
    secret: str
    passphrase: str


@router.get("/status")
async def api_agent_status():
    return get_agent().get_status()


@router.post("/start")
async def api_agent_start():
    agent = get_agent()
    if agent._running:
        return {"ok": True, "message": "Agent already running"}
    agent.start()
    return {"ok": True, "message": "Agent started"}


@router.post("/stop")
async def api_agent_stop():
    agent = get_agent()
    agent.stop()
    return {"ok": True, "message": "Agent stopped"}


@router.post("/revive")
async def api_agent_revive():
    agent = get_agent()
    agent.trader.state.is_alive = True
    agent.trader.state.shutdown_reason = ""
    agent.trader.state.daily_pnl = 0.0
    agent._save_state()
    return {"ok": True, "message": "Agent revived"}


@router.post("/config")
async def api_agent_config(request: Request):
    agent = get_agent()
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON"}

    mode = body.get("mode")
    if mode in ("paper", "live"):
        old = agent.trader.state.mode
        agent.trader.state.mode = mode
        if old != mode:
            print(f"[Agent] Mode switched to {mode.upper()}")
        agent._save_state()

    return {"ok": True, "mode": agent.trader.state.mode}


@router.post("/okx-keys")
async def api_agent_okx_keys(req: OKXKeysRequest):
    agent = get_agent()
    import os
    os.environ["OKX_API_KEY"] = req.api_key
    os.environ["OKX_SECRET"] = req.secret
    os.environ["OKX_PASSPHRASE"] = req.passphrase
    agent.trader._load_keys()
    has_keys = agent.trader._has_keys()
    return {"ok": True, "has_keys": has_keys}


@router.get("/okx-status")
async def api_agent_okx_status():
    agent = get_agent()
    if not agent.trader._has_keys():
        return {"connected": False, "reason": "No API keys configured"}
    try:
        bal = await agent.trader.get_balance()
        return {"connected": True, "balance": bal}
    except Exception as e:
        return {"connected": False, "reason": str(e)}
```

- [ ] **Step 2: Commit**

```bash
git add server/routers/execution.py
git commit -m "feat: extract execution router (7 routes: agent lifecycle, OKX)"
```

---

### Task 5: Extract signal router

**Files:**
- Create: `server/routers/signal.py`
- Reference: `server/app.py:920-1090` (strategy config/params/presets) + `server/app.py:1182-1280` (audit-log, lessons, signals)

- [ ] **Step 1: Create signal router**

```python
# server/routers/signal.py
"""Signal routes: strategy config, params, presets, audit log, lessons."""

import json
from pathlib import Path
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from ._deps import get_agent

router = APIRouter(prefix="/api/agent", tags=["signal"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRESETS_FILE = PROJECT_ROOT / "strategy_presets.json"

# ── Built-in presets ──

_BUILTIN_PRESETS = {
    "conservative": {
        "label": "Conservative",
        "description": "Tight filters, fewer trades, higher confidence required",
        "params": {
            "ma5_len": 5, "ma8_len": 8, "ema21_len": 21, "ma55_len": 55,
            "bb_length": 21, "bb_std_dev": 3.0,
            "dist_ma5_ma8": 1.0, "dist_ma8_ema21": 2.0, "dist_ema21_ma55": 3.0,
            "slope_len": 3, "slope_threshold": 0.15, "atr_period": 14,
        },
    },
    "aggressive": {
        "label": "Aggressive",
        "description": "Wider filters, more trades, lower confidence threshold",
        "params": {
            "ma5_len": 5, "ma8_len": 8, "ema21_len": 21, "ma55_len": 55,
            "bb_length": 21, "bb_std_dev": 2.0,
            "dist_ma5_ma8": 2.0, "dist_ma8_ema21": 3.5, "dist_ema21_ma55": 5.5,
            "slope_len": 3, "slope_threshold": 0.05, "atr_period": 14,
        },
    },
}


def _load_presets() -> dict:
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(_BUILTIN_PRESETS)


def _save_presets(presets: dict):
    PRESETS_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")


class StrategyConfigRequest(BaseModel):
    timeframe: str | None = None
    symbols: list[str] | None = None
    top_volume: int | None = None
    tick_interval: int | None = None
    max_position_pct: float | None = None
    max_positions: int | None = None


@router.post("/strategy-config")
async def api_agent_strategy_config(req: StrategyConfigRequest):
    agent = get_agent()
    from ..agent_brain import WATCH_SYMBOLS, SIGNAL_INTERVAL, TICK_INTERVAL_SEC
    import server.agent_brain as ab

    if req.timeframe:
        ab.SIGNAL_INTERVAL = req.timeframe
    if req.symbols is not None:
        ab.WATCH_SYMBOLS = req.symbols
    elif req.top_volume and req.top_volume > 0:
        from ..data_service import load_okx_swap_symbols
        top = await load_okx_swap_symbols()
        ab.WATCH_SYMBOLS = top[:req.top_volume]
    if req.tick_interval and req.tick_interval >= 10:
        ab.TICK_INTERVAL_SEC = req.tick_interval
    if req.max_position_pct is not None:
        agent.trader.risk.max_position_pct = req.max_position_pct
    if req.max_positions is not None:
        agent.trader.risk.max_positions = req.max_positions

    agent._save_state()
    return {
        "ok": True,
        "timeframe": ab.SIGNAL_INTERVAL,
        "symbols": ab.WATCH_SYMBOLS,
        "tick_interval": ab.TICK_INTERVAL_SEC,
    }


@router.post("/strategy-params")
async def api_agent_strategy_params(request: Request):
    agent = get_agent()
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON"}

    params = body.get("params", {})
    if not params:
        return {"ok": False, "reason": "No params provided"}

    current = agent.trader.state.strategy_params
    for key, val in params.items():
        if key in current:
            expected_type = type(current[key])
            try:
                current[key] = expected_type(val)
            except (ValueError, TypeError):
                pass

    agent._save_state()
    return {"ok": True, "params": current}


@router.get("/strategy-presets")
async def api_get_presets():
    return _load_presets()


@router.post("/strategy-presets/save")
async def api_save_preset(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON"}
    name = body.get("name", "").strip()
    if not name:
        return {"ok": False, "reason": "Name required"}
    presets = _load_presets()
    presets[name] = {
        "label": body.get("label", name),
        "description": body.get("description", ""),
        "params": body.get("params", get_agent().trader.state.strategy_params),
    }
    _save_presets(presets)
    return {"ok": True}


@router.post("/strategy-presets/load")
async def api_load_preset(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON"}
    name = body.get("name", "")
    presets = _load_presets()
    if name not in presets:
        return {"ok": False, "reason": f"Preset '{name}' not found"}
    preset = presets[name]
    agent = get_agent()
    agent.trader.state.strategy_params.update(preset.get("params", {}))
    agent._save_state()
    return {"ok": True, "params": agent.trader.state.strategy_params}


@router.post("/strategy-presets/delete")
async def api_delete_preset(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON"}
    name = body.get("name", "")
    presets = _load_presets()
    if name in presets:
        del presets[name]
        _save_presets(presets)
    return {"ok": True}


@router.get("/audit-log")
async def api_agent_audit_log(limit: int = Query(50, ge=1, le=500)):
    from ..agent_brain import TRADE_AUDIT_LOG
    if not TRADE_AUDIT_LOG.exists():
        return {"entries": []}
    try:
        raw = TRADE_AUDIT_LOG.read_text(encoding="utf-8").strip()
        lines = raw.split("\n") if raw else []
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        entries.reverse()
        return {"entries": entries}
    except Exception as e:
        return {"entries": [], "error": str(e)}


@router.get("/lessons")
async def api_agent_lessons():
    agent = get_agent()
    return agent.lessons.get_summary()


@router.get("/signals")
async def api_agent_signals(limit: int = Query(20, ge=1, le=100)):
    agent = get_agent()
    signals = agent._signal_history[-limit:]
    signals.reverse()
    return {"signals": signals}
```

- [ ] **Step 2: Commit**

```bash
git add server/routers/signal.py
git commit -m "feat: extract signal router (10 routes: strategy config, presets, audit)"
```

---

### Task 6: Extract risk router

**Files:**
- Create: `server/routers/risk.py`
- Reference: `server/app.py:1100-1134`

- [ ] **Step 1: Create risk router**

```python
# server/routers/risk.py
"""Risk management routes."""

from fastapi import APIRouter
from pydantic import BaseModel
from ._deps import get_agent

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
    agent = get_agent()
    r = agent.trader.risk
    if req.max_position_pct is not None:
        r.max_position_pct = req.max_position_pct / 100 if req.max_position_pct > 1 else req.max_position_pct
    if req.max_total_exposure_pct is not None:
        r.max_total_exposure_pct = req.max_total_exposure_pct / 100 if req.max_total_exposure_pct > 1 else req.max_total_exposure_pct
    if req.max_daily_loss_pct is not None:
        r.max_daily_loss_pct = req.max_daily_loss_pct / 100 if req.max_daily_loss_pct > 1 else req.max_daily_loss_pct
    if req.max_drawdown_pct is not None:
        r.max_drawdown_pct = req.max_drawdown_pct / 100 if req.max_drawdown_pct > 1 else req.max_drawdown_pct
    if req.max_positions is not None:
        r.max_positions = req.max_positions
    if req.cooldown_seconds is not None:
        r.cooldown_seconds = req.cooldown_seconds

    return {"ok": True, "risk_limits": {
        "max_position_pct": r.max_position_pct * 100,
        "max_total_exposure_pct": r.max_total_exposure_pct * 100,
        "max_daily_loss_pct": r.max_daily_loss_pct * 100,
        "max_drawdown_pct": r.max_drawdown_pct * 100,
        "max_positions": r.max_positions,
        "cooldown_seconds": r.cooldown_seconds,
    }}
```

- [ ] **Step 2: Commit**

```bash
git add server/routers/risk.py
git commit -m "feat: extract risk router (1 route)"
```

---

### Task 7: Extract ops router

**Files:**
- Create: `server/routers/ops.py`
- Reference: `server/app.py:151-157` (health), `server/app.py:1137-1180` (telegram), `server/app.py:1288-1300` (logs), `server/app.py:1338-1372` (healer, static files)

- [ ] **Step 1: Create ops router**

```python
# server/routers/ops.py
"""Ops routes: health, telegram, logs, healer, static files."""

import builtins
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from ._deps import get_agent, get_healer

router = APIRouter(tags=["ops"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@router.get("/api/health")
async def health_check():
    return {"status": "ok"}


@router.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@router.post("/api/agent/telegram-config")
async def api_agent_telegram_config(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON"}

    bot_token = body.get("bot_token", "").strip()
    chat_id = body.get("chat_id", "").strip()
    if not bot_token or not chat_id:
        return {"ok": False, "reason": "bot_token and chat_id required"}

    agent = get_agent()
    agent._telegram_config = {
        "bot_token": bot_token,
        "chat_id": chat_id,
        "notify_signals": body.get("notify_signals", True),
        "notify_fills": body.get("notify_fills", True),
        "notify_errors": body.get("notify_errors", False),
        "notify_daily": body.get("notify_daily", False),
    }

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": "Crypto Agent connected! Notifications enabled.", "parse_mode": "HTML"},
            )
            if resp.status_code == 200:
                print(f"[Telegram] Config saved, test message sent to {chat_id}")
                return {"ok": True}
            else:
                try:
                    err = resp.json().get("description", resp.text)
                except Exception:
                    err = resp.text
                return {"ok": False, "reason": f"Telegram API error: {err}"}
    except Exception as e:
        return {"ok": False, "reason": f"Network error: {e}"}


@router.get("/api/agent/logs")
async def api_agent_logs():
    from ..app import _LOG_BUFFER
    return {"logs": list(_LOG_BUFFER)}


@router.get("/api/healer/status")
async def api_healer_status():
    h = get_healer()
    return h.get_status()


@router.post("/api/healer/trigger")
async def api_healer_trigger():
    h = get_healer()
    result = await h.run_check()
    return {"ok": True, "result": result}


@router.post("/api/healer/stop")
async def api_healer_stop():
    h = get_healer()
    h.stop()
    return {"ok": True}


@router.post("/api/healer/start")
async def api_healer_start():
    h = get_healer()
    h.start()
    return {"ok": True}


@router.get("/style.css")
async def serve_css():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@router.get("/app.js")
async def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")
```

- [ ] **Step 2: Commit**

```bash
git add server/routers/ops.py
git commit -m "feat: extract ops router (9 routes: health, telegram, healer, static)"
```

---

### Task 8: Extract chat router

**Files:**
- Create: `server/routers/chat.py`
- Reference: `server/app.py:1302-1335`

- [ ] **Step 1: Create chat router**

```python
# server/routers/chat.py
"""Chat routes: AI conversation with Claude."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ._deps import get_chat

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str | None = None


@router.post("")
async def api_chat(req: ChatRequest):
    chat = get_chat()
    gen = chat.stream_response(req.message, session_id=req.session_id, model=req.model)
    return StreamingResponse(gen, media_type="text/plain")


@router.get("/models")
async def api_chat_models():
    return get_chat().get_models()


@router.get("/history")
async def api_chat_history(session_id: str = "default"):
    return get_chat().get_history(session_id)


@router.post("/clear")
async def api_chat_clear(session_id: str = "default"):
    get_chat().clear_history(session_id)
    return {"ok": True}
```

- [ ] **Step 2: Commit**

```bash
git add server/routers/chat.py
git commit -m "feat: extract chat router (4 routes)"
```

---

### Task 9: Extract onchain router

**Files:**
- Create: `server/routers/onchain.py`
- Reference: `server/app.py:1377-1495` (all 13 proxy routes)

- [ ] **Step 1: Create onchain router**

```python
# server/routers/onchain.py
"""On-chain / Smart Money proxy routes (proxied to port 8002)."""

import httpx
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/onchain", tags=["onchain"])

SM_BASE = "http://127.0.0.1:8002"


async def _sm_proxy(request: Request, path: str):
    """Forward request to Smart Money service on port 8002."""
    url = f"{SM_BASE}{path}"
    params = dict(request.query_params)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if request.method == "GET":
                resp = await client.get(url, params=params)
            elif request.method == "POST":
                body = await request.body()
                resp = await client.post(url, content=body, params=params,
                                         headers={"content-type": "application/json"})
            elif request.method == "PUT":
                body = await request.body()
                resp = await client.put(url, content=body, params=params,
                                        headers={"content-type": "application/json"})
            elif request.method == "DELETE":
                resp = await client.delete(url, params=params)
            else:
                return {"error": f"Unsupported method: {request.method}"}

            if resp.status_code == 200:
                return resp.json()
            return {"error": f"Smart Money service returned {resp.status_code}", "detail": resp.text[:500]}
    except httpx.ConnectError:
        return {"error": "Smart Money service offline", "status": "offline"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/health")
async def api_onchain_health(request: Request):
    return await _sm_proxy(request, "/api/health")


@router.get("/wallets")
async def api_onchain_wallets(request: Request):
    return await _sm_proxy(request, "/api/wallets")


@router.get("/wallets/smart-money")
async def api_onchain_smart_money(request: Request):
    return await _sm_proxy(request, "/api/wallets/smart-money")


@router.get("/wallets/{address}")
async def api_onchain_wallet_detail(address: str, request: Request):
    return await _sm_proxy(request, f"/api/wallets/{address}")


@router.post("/wallets/track/{address}")
async def api_onchain_track_wallet(address: str, request: Request):
    return await _sm_proxy(request, f"/api/wallets/track/{address}")


@router.delete("/wallets/track/{address}")
async def api_onchain_untrack_wallet(address: str, request: Request):
    return await _sm_proxy(request, f"/api/wallets/track/{address}")


@router.get("/signals")
async def api_onchain_signals(request: Request):
    return await _sm_proxy(request, "/api/signals")


@router.get("/signals/recommendations")
async def api_onchain_recommendations(request: Request):
    return await _sm_proxy(request, "/api/signals/recommendations")


@router.get("/signals/params")
async def api_onchain_get_params(request: Request):
    return await _sm_proxy(request, "/api/signals/params")


@router.put("/signals/params")
async def api_onchain_update_params(request: Request):
    return await _sm_proxy(request, "/api/signals/params")


@router.get("/token/analyze/{token_address}")
async def api_onchain_token_analyze(token_address: str, request: Request):
    return await _sm_proxy(request, f"/api/token/analyze/{token_address}")


@router.get("/validator/backtest")
async def api_onchain_backtest(request: Request):
    return await _sm_proxy(request, "/api/validator/backtest")


@router.get("/validator/high-confidence-wallets")
async def api_onchain_high_confidence(request: Request):
    return await _sm_proxy(request, "/api/validator/high-confidence-wallets")
```

- [ ] **Step 2: Commit**

```bash
git add server/routers/onchain.py
git commit -m "feat: extract onchain proxy router (13 routes)"
```

---

### Task 10: Slim down app.py to shell + register routers

**Files:**
- Modify: `server/app.py` (replace ~1495 lines with ~90 lines)

This is the final step. Replace the entire app.py with a slim shell that only does: create app, register middleware, register routers, lifecycle events.

- [ ] **Step 1: Rewrite app.py**

```python
# server/app.py
"""
Crypto Trading OS — FastAPI application shell.

All route handlers live in server/routers/*.py.
This file handles: app creation, CORS, singleton lifecycle, router registration, logging.
"""

import builtins
import logging
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers._deps import init_agent, init_chat, init_healer, get_agent, get_healer
from .routers import market_data, research, signal, execution, risk, ops, chat, onchain

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Centralized log capture ──────────────────────────────────────────────
_LOG_BUFFER: deque = deque(maxlen=200)


class _AgentLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            _LOG_BUFFER.append(msg)
        except Exception:
            pass


_handler = _AgentLogHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_handler)
logging.getLogger().setLevel(logging.INFO)

_original_print = builtins.print

def _capturing_print(*args, **kwargs):
    _original_print(*args, **kwargs)
    try:
        msg = " ".join(str(a) for a in args)
        _LOG_BUFFER.append(msg)
    except Exception:
        pass

builtins.print = _capturing_print


# ── App lifecycle ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"[env] Loaded {env_path}")
    except ImportError:
        pass

    init_agent()
    init_chat()
    healer = init_healer()
    healer.start()
    print("[App] All services initialized")

    yield

    # Shutdown
    try:
        get_agent().stop()
    except Exception:
        pass
    try:
        get_healer().stop()
    except Exception:
        pass


# ── App creation ─────────────────────────────────────────────────────────

app = FastAPI(title="Crypto Trading OS", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────────

app.include_router(market_data.router)
app.include_router(research.router)
app.include_router(signal.router)
app.include_router(execution.router)
app.include_router(risk.router)
app.include_router(chat.router)
app.include_router(onchain.router)
app.include_router(ops.router)  # ops last — serves / and static files

# Static files mount (fallback for direct file access)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")
```

- [ ] **Step 2: Verify all 59 routes still respond**

```bash
# Start server
python run.py &
sleep 5

# Test each domain
curl -s http://127.0.0.1:8001/api/health | python -m json.tool
curl -s http://127.0.0.1:8001/api/symbols | python -m json.tool | head -5
curl -s http://127.0.0.1:8001/api/agent/status | python -m json.tool | head -5
curl -s http://127.0.0.1:8001/api/agent/strategy-presets | python -m json.tool | head -5
curl -s http://127.0.0.1:8001/api/agent/risk-limits -X POST -H "Content-Type: application/json" -d '{}' | python -m json.tool
curl -s http://127.0.0.1:8001/api/chat/models | python -m json.tool
curl -s http://127.0.0.1:8001/api/healer/status | python -m json.tool
curl -s http://127.0.0.1:8001/api/onchain/health | python -m json.tool
curl -s http://127.0.0.1:8001/ -o /dev/null -w "%{http_code}"  # should be 200
```

- [ ] **Step 3: Commit**

```bash
git add server/app.py
git commit -m "refactor: slim app.py to shell — all routes in domain routers

Moved 59 routes from monolithic app.py (~1495 lines) into 8 domain routers:
- market_data (6), research (9), signal (10), execution (7)
- risk (1), ops (9), chat (4), onchain (13)

app.py is now ~90 lines: app creation, CORS, lifecycle, router registration.
No API contract changes — all paths and behaviors preserved."
```

---

## Verification Checklist

After all 10 tasks complete:

- [ ] `python run.py` starts without errors
- [ ] All 6 tabs in frontend still work (Recognize, Draw, Assist, Agent, On-Chain, Chat)
- [ ] Agent can start/stop
- [ ] Backtest runs
- [ ] Telegram config saves
- [ ] OKX status checks
- [ ] On-chain panel shows offline (expected without port 8002)
- [ ] AI Chat responds
- [ ] No broken API calls in browser console

---

## Route Count Verification

| Router | Routes | Prefix |
|--------|--------|--------|
| market_data | 6 | /api |
| research | 9 | /api |
| signal | 10 | /api/agent |
| execution | 7 | /api/agent |
| risk | 1 | /api/agent |
| ops | 9 | /api + /api/agent |
| chat | 4 | /api/chat |
| onchain | 13 | /api/onchain |
| **Total** | **59** | |
