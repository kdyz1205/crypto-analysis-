# Architecture Boundary Spec — Trading OS

> This document constrains Phase 1 router boundaries and prevents rework in Phase 2-4.
> It is NOT a detailed implementation plan — only boundary definitions.

---

## 1. Frontend Four Layers -> Backend Domain Mapping

### Layer 1: Market Workbench (default view — "what's happening?")

**Backend domains serving this layer:**
- `market` — symbol list, OHLCV data, chart overlays, top volume, data info
- `patterns` — S/R detection, trendlines, zones, pattern features, similarity

**Routes:**
- `/api/symbols`, `/api/symbol-info`, `/api/chart`, `/api/ohlcv`
- `/api/top-volume`, `/api/data-info`
- `/api/patterns`
- `/api/pattern-stats/backtest`, `/api/pattern-stats/features`
- `/api/pattern-stats/current-vs-history`, `/api/pattern-stats/line-similar`

### Layer 2: Execution Center ("do" and "control")

**Backend domains:**
- `agent` — lifecycle (start/stop/revive), config, strategy params/presets, signals, audit log, lessons
- `risk` — risk limits, block reasons, exposure (future: risk events, exposure endpoints)
- `execution` — OKX keys, OKX status, order routing (future: order ticket, position management)

**Routes:**
- `/api/agent/status`, `/api/agent/start`, `/api/agent/stop`, `/api/agent/revive`
- `/api/agent/config`, `/api/agent/strategy-config`, `/api/agent/strategy-params`
- `/api/agent/strategy-presets`, `/api/agent/strategy-presets/save`, `/api/agent/strategy-presets/load`, `/api/agent/strategy-presets/delete`
- `/api/agent/signals`, `/api/agent/audit-log`, `/api/agent/lessons`
- `/api/agent/risk-limits`
- `/api/agent/okx-keys`, `/api/agent/okx-status`

**Why agent/risk/execution are separate routers:**
- `agent` = the autonomous brain (start, stop, configure, inspect)
- `risk` = guard rails (limits, blocks, exposure). Currently only 1 route, but Phase 3 adds risk events and exposure meters. Splitting now prevents moving it later.
- `execution` = OKX connectivity and order routing. Currently just keys/status, but Phase 3 adds order ticket, slippage estimation, position management. Splitting now makes Phase 3 additive.

### Layer 3: Research Lab ("verify" and "evolve")

**Backend domains:**
- `research` — backtest, optimize, MA ribbon analysis/backtest

**Routes:**
- `/api/backtest`, `/api/backtest/optimize`
- `/api/ma-ribbon`, `/api/ma-ribbon/backtest`

**Note:** Pattern stats (backtest, features, similarity) serve BOTH Workbench and Research. They live in `patterns` router because they're tightly coupled to pattern detection data, not to the research workflow.

### Layer 4: Control Bus (ops + external integration)

**Backend domains:**
- `ops` — health, static files, telegram config, agent logs, healer lifecycle
- `onchain` — smart money proxy (port 8002)
- `chat` — AI conversation

**Routes:**
- `/api/health`, `/`, `/style.css`, `/app.js`
- `/api/agent/telegram-config`, `/api/agent/logs`
- `/api/healer/status`, `/api/healer/trigger`, `/api/healer/stop`, `/api/healer/start`
- `/api/onchain/*` (13 proxy routes)
- `/api/chat`, `/api/chat/models`, `/api/chat/history`, `/api/chat/clear`

---

## 2. Event Bus — Core Event Names (Phase 3)

These events don't exist yet. This list constrains Phase 3 design and informs Phase 1 router naming.

### Signal Lifecycle
```
signal.detected        -> origin: agent_brain.generate_signal()
signal.validated       -> origin: PreTradeChecklist.validate() passes
signal.blocked         -> origin: PreTradeChecklist or lessons ledger blocks
signal.expired         -> origin: dedup window elapsed without execution
```

### Order Lifecycle
```
order.intent.created   -> origin: agent tick() decides to act
order.submitted        -> origin: okx_trader.open_position() called
order.filled           -> origin: okx_trader confirms fill (paper or live)
order.rejected         -> origin: exchange rejects or execution fails
order.cancelled        -> origin: manual cancel or timeout
```

### Position Lifecycle
```
position.opened        -> origin: order.filled triggers position creation
position.updated       -> origin: mark-to-market on each tick
position.sl_hit        -> origin: price breaks stop loss level
position.tp_hit        -> origin: price reaches take profit level
position.closed        -> origin: exit order filled
```

### Risk Events
```
risk.limit.hit         -> origin: daily loss, drawdown, or exposure threshold
risk.cooldown.start    -> origin: consecutive loss protection triggered
risk.cooldown.end      -> origin: cooldown timer expired
risk.shutdown          -> origin: max drawdown breached
```

### System Events
```
agent.started          -> origin: agent.start()
agent.stopped          -> origin: agent.stop()
agent.evolved          -> origin: parameter mutation
agent.regime.changed   -> origin: market regime detection shift
ops.healer.triggered   -> origin: self-healer detects error
ops.error              -> origin: any unhandled exception in tick loop
summary.daily          -> origin: daily summary timer fires
```

### Event Consumers (Phase 4)
```
Telegram bot     -> subscribes to: signal.blocked, order.filled, position.closed, risk.*, summary.daily
Frontend UI      -> subscribes to: ALL (via SSE or WebSocket)
Audit log        -> subscribes to: ALL (append to trade_audit.jsonl)
Lessons ledger   -> subscribes to: position.closed, risk.*, agent.evolved
```

---

## 3. State Machine Boundaries

### Signal State Machine
```
                   ┌──────────┐
                   │ DETECTED │
                   └────┬─────┘
                        │
                   ┌────▼─────┐
              ┌────│ VALIDATED │────┐
              │    └──────────┘    │
              │                    │
        ┌─────▼──────┐    ┌───────▼────────┐
        │ RISK_CHECKED│    │ BLOCKED (reason)│
        └─────┬──────┘    └────────────────┘
              │
        ┌─────▼──┐
        │ READY  │
        └─────┬──┘
              │
        ┌─────▼─────┐
        │ SUBMITTED │
        └─────┬─────┘
              │
     ┌────────┼────────┐
     │        │        │
┌────▼───┐ ┌──▼───┐ ┌──▼──────┐
│ FILLED │ │REJECT│ │CANCELLED│
└────┬───┘ └──────┘ └─────────┘
     │
┌────▼────┐
│ MANAGED │ (position open, being monitored)
└────┬────┘
     │
┌────▼────┐
│ CLOSED  │
└─────────┘
```

**Router ownership of state transitions:**
- DETECTED, VALIDATED: `agent` router (signal generation)
- BLOCKED: `risk` router (risk checks) or `agent` router (lessons block)
- READY, SUBMITTED, FILLED, REJECTED, CANCELLED: `execution` router
- MANAGED, CLOSED: `agent` router (position management loop)

### Risk State Machine
```
NORMAL -> CAUTION (daily loss > 1%) -> DANGER (daily loss > 1.5%) -> SHUTDOWN (drawdown > max)
NORMAL -> COOLDOWN (3 consecutive losses) -> NORMAL (after cooldown expires or win)
```

---

## 4. Target Directory Structure (Post Phase 1)

```
server/
├── app.py                      # ~80 lines: FastAPI app, CORS, lifespan, router registration
│
├── core/                       # Shared infrastructure (NOT business logic)
│   ├── __init__.py
│   ├── dependencies.py         # get_agent(), get_chat(), get_healer() singletons
│   ├── log_buffer.py           # _LOG_BUFFER, _AgentLogHandler, _capturing_print
│   └── config.py               # PROJECT_ROOT, env loading, shared constants
│
├── routers/                    # HTTP route handlers only (thin — delegate to services)
│   ├── __init__.py
│   ├── market.py               # /api/symbols, /api/ohlcv, /api/chart, etc.
│   ├── patterns.py             # /api/patterns, /api/pattern-stats/*
│   ├── research.py             # /api/backtest, /api/ma-ribbon*
│   ├── agent.py                # /api/agent/status, start, stop, config, strategy, signals, audit
│   ├── risk.py                 # /api/agent/risk-limits (+ future risk events)
│   ├── execution.py            # /api/agent/okx-keys, okx-status (+ future order ticket)
│   ├── ops.py                  # /api/health, /, static, telegram, logs, healer
│   ├── chat.py                 # /api/chat/*
│   └── onchain.py              # /api/onchain/* (proxy)
│
├── services/                   # Business logic (UNCHANGED in Phase 1)
│   ├── agent_brain.py          # Renamed from server/agent_brain.py (Phase 2+ move)
│   ├── okx_trader.py
│   ├── data_service.py
│   ├── backtest_service.py
│   ├── pattern_service.py
│   ├── pattern_features.py
│   ├── ma_ribbon_service.py
│   ├── ai_chat.py
│   └── self_healer.py
│
└── __init__.py
```

**Phase 1 scope:** Create `core/` and `routers/`. Do NOT move service files — they stay at `server/*.py`. Moving to `server/services/` is Phase 2 work (would break all imports simultaneously).

**Phase 1 import rule:** Routers import from `core.dependencies` for singletons and from `server.*_service` for business logic. No router imports another router. No service imports a router.

---

## 5. Shared Dependency Ownership

| Dependency | Location (Phase 1) | Used by |
|---|---|---|
| `get_agent()` | `core/dependencies.py` | agent, risk, execution, ops, signal routers |
| `get_chat()` | `core/dependencies.py` | chat router |
| `get_healer()` | `core/dependencies.py` | ops router |
| `_LOG_BUFFER` | `core/log_buffer.py` | ops router (read), app.py (init) |
| `_load_df_for_analysis()` | stays in `app.py` or moves to `core/config.py` | market, patterns routers |
| `_sm_proxy()` | `routers/onchain.py` (private to that router) | onchain router only |
| `PROJECT_ROOT` | `core/config.py` | ops router (static files), multiple services |
| `PRESETS_FILE` | `routers/agent.py` (private) | agent router only |
| Pydantic models | defined in the router that uses them | each router owns its models |

---

## 6. Route Count Per Router (59 total)

| Router | Count | Prefix |
|--------|-------|--------|
| `market` | 6 | `/api` |
| `patterns` | 4 | `/api` |
| `research` | 3 | `/api` |
| `agent` | 12 | `/api/agent` |
| `risk` | 1 | `/api/agent` |
| `execution` | 2 | `/api/agent` |
| `ops` | 9 | mixed |
| `chat` | 4 | `/api/chat` |
| `onchain` | 13 | `/api/onchain` |
| **Total** | **54** | |

Note: Original count was 59 but some were misallocated. Precise count: 54 unique route handlers + 5 from ops (health, root, css, js, + telegram overlap). Will verify during implementation.

---

## 7. Phase 1 Execution Rules

1. **No behavior changes.** Every route returns identical JSON before and after.
2. **No renamed endpoints.** All URL paths preserved exactly.
3. **No new features.** No added fields, no changed defaults, no improved error handling.
4. **No service refactoring.** `agent_brain.py`, `okx_trader.py`, etc. stay untouched.
5. **No import chain changes in services.** Services don't know routers exist.
6. **Smoke test after every router extraction.** Server starts + key routes respond.
7. **One router at a time.** Never extract two routers in parallel.
