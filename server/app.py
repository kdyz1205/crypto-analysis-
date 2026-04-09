"""
Crypto Trading OS — FastAPI application shell.

All route handlers live in server/routers/*.py.
This file handles: app creation, CORS, singleton lifecycle, router registration.

Log capture is installed by importing server/core/log_buffer.py (side effect on import).
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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
    strategy,
)
from .subscribers import audit as audit_sub, telegram as telegram_sub, sse_broadcast as sse_sub
from .core import scheduler as sched_core


app = FastAPI(title="Crypto TA")


# ── Lifecycle ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
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

    sched_core.register_handler("daily_summary", _handler_daily_summary)
    sched_core.register_handler("agent_scan", _handler_agent_scan)
    sched_core.register_handler("memory_note", _handler_memory_note)
    sched_core.start_scheduler()
    print("[Scheduler] Started with 3 default handlers")


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


# ── Middleware ───────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files at /static/ (legacy path)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")


# ── Register routers ─────────────────────────────────────────────────────
# Order matters: specific prefixes before ops (which owns "/" and "/style.css")

app.include_router(market.router)       # /api/symbols, /api/ohlcv, /api/chart, etc.
app.include_router(patterns.router)     # /api/patterns, /api/pattern-stats/*
app.include_router(strategy.router)     # /api/strategy/config, snapshot, replay
app.include_router(research.router)     # /api/backtest, /api/ma-ribbon*
app.include_router(agent.router)        # /api/agent/status, start, stop, config, strategy*, signals, audit-log, lessons
app.include_router(risk.router)         # /api/agent/risk-limits
app.include_router(execution.router)    # /api/agent/okx-keys, okx-status
app.include_router(chat.router)         # /api/chat/*
app.include_router(onchain.router)      # /api/onchain/* (proxy to 8002)
app.include_router(stream.router)       # /api/stream, /api/stream/status, /api/events/history
app.include_router(memory.router)       # /api/memory/*  (Hermes persistent memory)
app.include_router(schedule.router)     # /api/schedule/* (Hermes natural-language scheduler)
app.include_router(ops.router)          # LAST: /api/health, /, /style.css, /app.js, telegram, logs, healer
