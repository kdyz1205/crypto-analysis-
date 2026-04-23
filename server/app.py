"""
Crypto Trading OS — FastAPI application shell.

All route handlers live in server/routers/*.py.
This file handles: app creation, CORS, singleton lifecycle, router registration.

Log capture is installed by importing server/core/log_buffer.py (side effect on import).
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Import log_buffer FIRST to install log capture handlers before any other module
from .core import log_buffer  # noqa: F401 (side effect: installs handlers)
from .core.config import PROJECT_ROOT
from .core.dependencies import init_agent, init_chat, init_healer, get_agent, get_healer

# Re-export _LOG_BUFFER for any module that imports it from server.app (backward compat)
from .core.log_buffer import _LOG_BUFFER  # noqa: F401

from .routers import (
    market, patterns, research,
    agent, risk, execution,
    ops, chat, onchain,
    stream, memory, schedule,
    strategy, paper_execution, live_execution, drawings, runtime,
    tools_api, conditionals,
    mar_bb_runner, orderbook, line_alerts,
)
from .subscribers import audit as audit_sub, telegram as telegram_sub, sse_broadcast as sse_sub
from .core import scheduler as sched_core

# Force UTF-8 on stdout/stderr so non-ASCII chars in print() statements
# don't raise UnicodeEncodeError on Windows cp1252. User 2026-04-22: HTTP 500
# on place-line-order triggered by `\u2192` right arrow in SL-cap log. Root-cause
# fix: bypass the encoding rather than hunt every char.
import sys as _sys_encoding_fix
try:
    _sys_encoding_fix.stdout.reconfigure(encoding='utf-8', errors='replace')
    _sys_encoding_fix.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


app = FastAPI(title="Crypto TA")


# ── Lifecycle ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    # Load .env FIRST so all modules can see BITGET_API_KEY etc.
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=True)
    except ImportError:
        pass

    agent_inst = init_agent()
    chat_inst = init_chat()
    healer_inst = init_healer()
    healer_inst.start()
    print(f"[Agent] Initialized. Mode={agent_inst.trader.state.mode} Gen={agent_inst.trader.state.generation}")
    print(f"[AI Chat] Ready. API key={'set' if chat_inst._anthropic_client else 'NOT set'}")
    print(f"[Healer] Self-healing active. AI={'enabled' if healer_inst._client else 'disabled'}")

    # Register Phase 3 event bus subscribers
    audit_sub.register()
    telegram_sub.register()
    sse_sub.register()
    print("[EventBus] Phase 3 subscribers registered")

    # Start conditional-order watcher (polls pending virtual orders)
    try:
        from .conditionals.watcher import start_watcher as _start_cond_watcher
        _start_cond_watcher()
    except Exception as _e:
        print(f"[ConditionalWatcher] failed to start: {_e}")

    # Auto-start the MA Ribbon + BB Exit live runner.
    # Runs in background 24/7 scanning top-100 Bitget USDT perps and
    # firing real orders when V1+V3 crossover signals appear. Safe to
    # enable because: multi-position gate (max 5 concurrent), skips
    # already-held symbols, $12 notional per position.
    # Disable by setting MAR_BB_AUTOSTART=0 in .env.
    import os as _os
    if _os.environ.get("MAR_BB_AUTOSTART", "1") != "0":
        try:
            from .strategy.mar_bb_runner import start_runner as _start_mar_bb, DEFAULT_RUNNER_CFG
            _start_mar_bb({**DEFAULT_RUNNER_CFG})
            print(f"[mar_bb] auto-started: top_n={DEFAULT_RUNNER_CFG['top_n']} "
                  f"tf={DEFAULT_RUNNER_CFG['timeframe']} "
                  f"max_pos={DEFAULT_RUNNER_CFG['max_concurrent_positions']} "
                  f"notional=${DEFAULT_RUNNER_CFG['notional_usd']}")
        except Exception as _e:
            print(f"[mar_bb] auto-start failed: {_e}")

    # Always-on trailing-SL maintenance. Runs REGARDLESS of MAR_BB_AUTOSTART
    # so manually-placed line orders (via Trade Plan modal) keep their SL
    # moving with the drawn line even when the scanner is off. The loop
    # only touches existing positions — never opens new trades.
    try:
        from .strategy.mar_bb_runner import start_maintenance_only as _start_trendline_maint
        _start_trendline_maint()
        print("[trendline_maintenance] always-on trailing loop started")
    except Exception as _e:
        print(f"[trendline_maintenance] start failed: {_e}")

    # Auto-start orderbook websocket service for real-time L2 features.
    # Tracks top symbols for microstructure signals (imbalance, cancel pressure, etc.)
    try:
        from .hft.orderbook_service import start_service as _start_ob
        await _start_ob(["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "DOGEUSDT",
                         "XRPUSDT", "BNBUSDT", "NEARUSDT", "SUIUSDT", "PEPEUSDT"])
        print("[orderbook] L2 websocket service started")
    except Exception as _e:
        print(f"[orderbook] failed to start: {_e}")

    # Hermes-inspired: register default scheduler handlers + start loop
    from .core import memory as mem_core
    from .core.events import bus as _bus, make_summary_daily

    async def _handler_daily_summary(params: dict) -> str:
        s = agent_inst.trader.state
        positions = {sym: {"side": p.side, "pnl": p.unrealized_pnl} for sym, p in s.positions.items()}
        await _bus.publish(make_summary_daily(
            equity=s.equity, daily_pnl=s.daily_pnl,
            total_trades=s.total_trades,
            win_rate=(s.win_count / max(s.total_trades, 1)) * 100,
            positions=positions,
        ))
        return f"Daily summary published: equity=${s.equity:.2f}"

    async def _handler_agent_scan(params: dict) -> str:
        try:
            from .agent_brain import WATCH_SYMBOLS
            count = 0
            for sym in WATCH_SYMBOLS:
                sig = await agent_inst.generate_signal(sym)
                if sig:
                    count += 1
            return f"Scanned {len(WATCH_SYMBOLS)} symbols, found {count} signals"
        except Exception as e:
            return f"Scan failed: {e}"

    async def _handler_memory_note(params: dict) -> str:
        ns = params.get("namespace", "scheduled")
        key = params.get("key", f"note_{int(asyncio.get_event_loop().time())}")
        content = params.get("content", "scheduled memory note")
        mem_core.save_memory(ns, key, content, params.get("metadata"))
        return f"Saved memory {ns}/{key}"

    async def _handler_pattern_rebuild(params: dict) -> str:
        """Scheduled pattern database rebuild for top symbols."""
        from tools.pattern_batch import start_batch_build, is_running
        if is_running():
            return "batch build already in progress"
        symbols = params.get("symbols") or ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","HYPEUSDT","DOGEUSDT","PEPEUSDT","ADAUSDT"]
        timeframes = params.get("timeframes") or ["15m","1h","4h","1d"]
        days = params.get("days", 730)
        result = await start_batch_build(symbols, timeframes, days=days)
        return f"pattern rebuild started: {len(symbols)}x{len(timeframes)} jobs"

    async def _handler_pattern_recompute(params: dict) -> str:
        """Scheduled pattern stats recompute from live outcome flags."""
        from tools.pattern_writeback import process_recompute_flags
        result = process_recompute_flags()
        return f"recomputed {result['processed']} symbol_timeframes"

    async def _handler_health_digest(params: dict) -> str:
        """Scheduled closed-loop health digest — automatic snapshot to disk."""
        from tools.health_digest import build_digest, save_digest
        digest = build_digest(hours=params.get("hours", 24))
        path = save_digest(digest)
        return f"digest saved: {digest['summary_line']}"

    sched_core.register_handler("daily_summary", _handler_daily_summary)
    sched_core.register_handler("agent_scan", _handler_agent_scan)
    sched_core.register_handler("memory_note", _handler_memory_note)
    sched_core.register_handler("pattern_rebuild", _handler_pattern_rebuild)
    sched_core.register_handler("pattern_recompute", _handler_pattern_recompute)
    sched_core.register_handler("health_digest", _handler_health_digest)
    sched_core.start_scheduler()
    print("[Scheduler] Started with 6 default handlers")
    await runtime.runtime_manager.startup()
    print("[Runtime] Subaccount runtime manager ready")

    # Evolution engine disabled on startup to avoid API congestion.
    # Start manually via POST /api/runtime/leaderboard/start when needed.
    print("[Evolution] Engine available but not auto-started (avoids API congestion)")


@app.on_event("shutdown")
async def _shutdown():
    try:
        get_agent().stop()
    except Exception:
        pass
    try:
        get_healer().stop()
    except Exception:
        pass
    try:
        await runtime.runtime_manager.shutdown()
    except Exception:
        pass


# ── Middleware ───────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Fixed: credentials=True conflicts with origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)
# Gzip everything >= 1 KB. Major impact on /api/ohlcv (3 MB JSON → ~400 KB
# over the wire). minimum_size keeps tiny endpoints (health, status) bare.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)


# Serve frontend static files at /static/ (legacy path)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")


# ── JS no-cache middleware (Bug B cache-bust — 2026-04-22) ──────────────
# StaticFiles subclass approach failed: headers set by override didn't
# reach the wire (GZip middleware and/or internal Response finalization
# stripped them). Middleware runs AFTER response build for every path,
# so it reliably mutates Cache-Control on /static/**/*.js.
# Verified via: curl -I http://localhost:8000/static/js/.../chart_drawing.js
from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTPMiddleware


class _StaticJsNoCacheMiddleware(_BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") and path.endswith(".js"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(_StaticJsNoCacheMiddleware)


# ── Register routers ─────────────────────────────────────────────────────
# Order matters: specific prefixes before ops (which owns "/" and "/style.css")

app.include_router(market.router)       # /api/symbols, /api/ohlcv, /api/chart, etc.
app.include_router(patterns.router)     # /api/patterns, /api/pattern-stats/*
app.include_router(strategy.router)     # /api/strategy/config, snapshot, replay
app.include_router(drawings.router)     # /api/drawings/*
app.include_router(paper_execution.router)  # /api/paper-execution/*
app.include_router(live_execution.router)  # /api/live-execution/*
app.include_router(runtime.router)      # /api/runtime/*
app.include_router(research.router)     # /api/backtest, /api/ma-ribbon*
app.include_router(agent.router)        # /api/agent/status, start, stop, config, strategy*, signals, audit-log, lessons
app.include_router(risk.router)         # /api/agent/risk-limits
app.include_router(execution.router)    # /api/agent/okx-keys, okx-status
app.include_router(chat.router)         # /api/chat/*
app.include_router(onchain.router)      # /api/onchain/* (proxy to 8002)
app.include_router(stream.router)       # /api/stream, /api/stream/status, /api/events/history
app.include_router(memory.router)       # /api/memory/*  (Hermes persistent memory)
app.include_router(schedule.router)     # /api/schedule/* (Hermes natural-language scheduler)
app.include_router(tools_api.router)    # /api/tools/* (research leaderboard, audit, failures, factors)
app.include_router(conditionals.router) # /api/conditionals/* + /api/drawings/manual/analyze
app.include_router(mar_bb_runner.router) # /api/mar-bb/* — live MA ribbon + BB exit scanner
app.include_router(orderbook.router)    # /api/orderbook/* — real-time L2 features
app.include_router(line_alerts.router)  # /api/alerts/* — trendline price alerts → Telegram
app.include_router(ops.router)          # LAST: /api/health, /, /style.css, /app.js, telegram, logs, healer
# reload trigger 1776885655
