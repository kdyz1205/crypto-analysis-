# Phase 3: Event Bus + Signal State Machine — Implementation Plan

> **Execution mode:** Inline. Event infrastructure is load-bearing across the agent loop, trader, healer, and (future) bot. Single-threaded changes with smoke tests after each milestone.

**Goal:** Introduce an in-process event bus and explicit state machines for Signal → Risk → Order → Position lifecycle. Make the system observable, replayable, and ready for the Telegram bot to subscribe as a consumer instead of owning frontend state.

**Architecture:** Python in-process async event bus (no external broker yet). Events are dataclasses published by the agent brain / trader / risk checks. Subscribers include: audit log writer (existing `trade_audit.jsonl`), Telegram notifier, SSE stream to frontend (for glassbox UI), and future Bot integration.

**Tech Stack:** asyncio, dataclasses. No new external dependencies.

**Prerequisites:**
- Phase 1 complete (routers split)
- Phase 2 complete (frontend layered) — needed so Decision Rail and Glassbox can consume SSE stream

**Non-goals for Phase 3:**
- No external message broker (Kafka, Redis, etc.)
- No persistent event store (events are emitted and consumed live; audit log remains the source of truth for history)
- No replay engine (Phase 4+)

---

## 1. Target File Structure

```
server/
├── core/
│   ├── events.py               # NEW: Event dataclasses, EventBus class, global bus instance
│   ├── state_machines.py       # NEW: SignalState, OrderState, PositionState enums + transitions
│   ├── event_types.py          # EXISTS (Phase 1 scaffold) — upgrade to canonical source
│   └── ...
│
├── subscribers/                # NEW: Long-running event consumers
│   ├── __init__.py
│   ├── audit.py                # Writes events to trade_audit.jsonl
│   ├── telegram.py             # Pushes to Telegram (replaces direct _notify_telegram in agent_brain)
│   └── sse_broadcast.py        # Queues events for frontend SSE stream
│
├── routers/
│   ├── stream.py               # NEW: GET /api/stream (SSE endpoint consumed by frontend Decision Rail + Glassbox)
│   └── ...
│
└── agent_brain.py              # MODIFIED: emits events at each lifecycle point instead of direct print+audit
```

---

## 2. Canonical Event Names

Already drafted in `server/core/event_types.py`. Phase 3 promotes these from constants-only to fully-typed event dataclasses.

### Signal lifecycle
- `signal.detected` — Agent generated a raw signal (before validation)
- `signal.validated` — Pre-trade checklist passed
- `signal.blocked` — Risk check or lessons ledger blocked
- `signal.expired` — Dedup window passed without execution

### Order lifecycle
- `order.submitted` — Sent to exchange (or paper trader)
- `order.filled` — Confirmed fill
- `order.rejected` — Exchange rejected
- `order.cancelled` — Manual cancel or timeout

### Position lifecycle
- `position.opened` — From order.filled on entry
- `position.updated` — Unrealized P&L tick
- `position.sl_hit` — Stop loss breached
- `position.tp_hit` — Take profit reached
- `position.closed` — Exit order filled

### Risk events
- `risk.check.passed`
- `risk.check.failed`
- `risk.limit.hit` — Exposure, daily loss, or position count limit
- `risk.cooldown.started` / `risk.cooldown.ended`
- `risk.kill_switch.armed` / `risk.kill_switch.released`

### Agent events
- `agent.started` / `agent.stopped` / `agent.revived`
- `agent.mode.changed`
- `agent.config.updated`
- `agent.error.raised`
- `agent.regime.changed`

### Ops events
- `ops.healer.triggered` / `ops.healer.completed`
- `ops.integration.failed`
- `summary.daily`

---

## 3. Signal State Machine

```
DETECTED
    ↓ (PreTradeChecklist.validate passes)
VALIDATED
    ↓ (risk check passes)
RISK_CHECKED
    ↓
READY
    ↓ (execute called)
SUBMITTED
    ↓
┌───────────┬───────────┬────────────┐
FILLED   REJECTED  CANCELLED   EXPIRED
    ↓
MANAGED (position open, tracked each tick)
    ↓
┌─────────┬─────────┐
TP_HIT   SL_HIT   MANUAL_CLOSE
    ↓
CLOSED
```

**Forbidden transitions:**
- Cannot go back from FILLED to READY
- Cannot transition from BLOCKED to any state (terminal)
- Cannot transition from CLOSED to any state (terminal)

**Transition validator:** `SignalState.can_transition(current, next) -> bool`

---

## 4. Event Bus Design

### Core types

```python
# server/core/events.py
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import asyncio
import json
import uuid


@dataclass(slots=True)
class Event:
    """Base event structure. All domain events inherit this."""
    type: str
    payload: dict
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    correlation_id: str | None = None  # Links related events (e.g., signal→order→position)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


Subscriber = "callable[[Event], Awaitable[None]]"  # async callable


class EventBus:
    """
    In-process async pub/sub.

    Subscribers are async functions. Publishing fires all matching handlers
    concurrently via asyncio.gather. Wildcards: subscribe('signal.*') matches
    signal.detected, signal.blocked, etc.
    """

    def __init__(self):
        self._subscribers: dict[str, list] = {}   # event_type -> [callables]
        self._wildcard: list = []                  # ('prefix', callable)
        self._history: list[Event] = []            # last N events for debugging
        self._history_limit: int = 500

    def subscribe(self, event_type: str, handler):
        if event_type.endswith('.*'):
            prefix = event_type[:-2]  # e.g., 'signal' from 'signal.*'
            self._wildcard.append((prefix, handler))
        else:
            self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler):
        if event_type.endswith('.*'):
            prefix = event_type[:-2]
            self._wildcard = [(p, h) for (p, h) in self._wildcard if not (p == prefix and h == handler)]
        else:
            self._subscribers.get(event_type, []).remove(handler)

    async def publish(self, event: Event):
        """Fan out event to all matching subscribers concurrently."""
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        handlers = list(self._subscribers.get(event.type, []))
        for (prefix, handler) in self._wildcard:
            if event.type.startswith(prefix + '.') or event.type == prefix:
                handlers.append(handler)

        if not handlers:
            return

        results = await asyncio.gather(
            *(self._safe_invoke(h, event) for h in handlers),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                # Log but don't propagate — one bad subscriber shouldn't kill the bus
                print(f"[EventBus] Subscriber error for {event.type}: {r}")

    async def _safe_invoke(self, handler, event):
        try:
            await handler(event)
        except Exception as e:
            raise e

    def get_recent(self, limit: int = 50) -> list[Event]:
        return self._history[-limit:]


# Global bus instance (singleton)
bus = EventBus()


# ── Typed event helpers ──

def make_signal_detected(symbol, side, confidence, reason, price, sl, tp, regime, correlation_id=None):
    return Event(
        type='signal.detected',
        payload={
            'symbol': symbol,
            'side': side,
            'confidence': confidence,
            'reason': reason,
            'price': price,
            'sl': sl,
            'tp': tp,
            'regime': regime,
        },
        correlation_id=correlation_id,
    )

def make_signal_blocked(symbol, side, block_reasons, correlation_id=None):
    return Event(
        type='signal.blocked',
        payload={'symbol': symbol, 'side': side, 'block_reasons': block_reasons},
        correlation_id=correlation_id,
    )

def make_order_submitted(symbol, side, size, price, mode, correlation_id=None):
    return Event(
        type='order.submitted',
        payload={'symbol': symbol, 'side': side, 'size': size, 'price': price, 'mode': mode},
        correlation_id=correlation_id,
    )

def make_order_filled(symbol, side, size, fill_price, pnl=None, correlation_id=None):
    return Event(
        type='order.filled',
        payload={'symbol': symbol, 'side': side, 'size': size, 'fill_price': fill_price, 'pnl': pnl},
        correlation_id=correlation_id,
    )

def make_position_opened(symbol, side, size, entry_price, sl, tp, correlation_id=None):
    return Event(
        type='position.opened',
        payload={'symbol': symbol, 'side': side, 'size': size, 'entry_price': entry_price, 'sl': sl, 'tp': tp},
        correlation_id=correlation_id,
    )

def make_position_closed(symbol, side, entry_price, exit_price, pnl_pct, pnl_usd, reason, correlation_id=None):
    return Event(
        type='position.closed',
        payload={
            'symbol': symbol, 'side': side,
            'entry_price': entry_price, 'exit_price': exit_price,
            'pnl_pct': pnl_pct, 'pnl_usd': pnl_usd, 'reason': reason,
        },
        correlation_id=correlation_id,
    )

def make_risk_limit_hit(limit_name, current_value, max_value, symbol=None):
    return Event(
        type='risk.limit.hit',
        payload={'limit': limit_name, 'current': current_value, 'max': max_value, 'symbol': symbol},
    )

def make_agent_started(mode, equity, generation):
    return Event(
        type='agent.started',
        payload={'mode': mode, 'equity': equity, 'generation': generation},
    )

def make_agent_regime_changed(from_regime, to_regime, confidence):
    return Event(
        type='agent.regime.changed',
        payload={'from': from_regime, 'to': to_regime, 'confidence': confidence},
    )

def make_summary_daily(equity, daily_pnl, total_trades, win_rate, positions):
    return Event(
        type='summary.daily',
        payload={
            'equity': equity, 'daily_pnl': daily_pnl,
            'total_trades': total_trades, 'win_rate': win_rate,
            'positions': positions,
        },
    )
```

---

## 5. Tasks

### Task 1: Create event bus core

**Files:**
- Create: `server/core/events.py` (full module as shown above)

- [ ] **Step 1:** Write `server/core/events.py` with `Event`, `EventBus`, `bus` singleton, and all `make_*` helpers

- [ ] **Step 2:** Write unit smoke test in a temp file

```python
# tmp_test_bus.py
import asyncio
from server.core.events import bus, make_signal_detected, Event

received = []

async def handler(ev: Event):
    received.append(ev)

async def main():
    bus.subscribe('signal.*', handler)
    ev = make_signal_detected('BTCUSDT', 'long', 0.7, 'test', 50000, 49000, 51000, 'trending')
    await bus.publish(ev)
    assert len(received) == 1
    assert received[0].type == 'signal.detected'
    assert received[0].payload['symbol'] == 'BTCUSDT'
    print('OK bus publish/subscribe works')

asyncio.run(main())
```

Run: `python tmp_test_bus.py`
Expected: `OK bus publish/subscribe works`
Then delete the temp file.

- [ ] **Step 3:** Commit

```bash
git add server/core/events.py
git commit -m "feat(phase3): add EventBus core with async pub/sub + event factories"
```

---

### Task 2: Create state machines module

**Files:**
- Create: `server/core/state_machines.py`

- [ ] **Step 1:** Implement state machines

```python
# server/core/state_machines.py
from enum import Enum
from dataclasses import dataclass


class SignalState(Enum):
    DETECTED = 'detected'
    VALIDATED = 'validated'
    RISK_CHECKED = 'risk_checked'
    READY = 'ready'
    SUBMITTED = 'submitted'
    FILLED = 'filled'
    REJECTED = 'rejected'
    CANCELLED = 'cancelled'
    EXPIRED = 'expired'
    BLOCKED = 'blocked'
    MANAGED = 'managed'
    CLOSED = 'closed'


# Allowed transitions: current -> set of allowed next
_SIGNAL_TRANSITIONS = {
    SignalState.DETECTED: {SignalState.VALIDATED, SignalState.BLOCKED, SignalState.EXPIRED},
    SignalState.VALIDATED: {SignalState.RISK_CHECKED, SignalState.BLOCKED},
    SignalState.RISK_CHECKED: {SignalState.READY, SignalState.BLOCKED},
    SignalState.READY: {SignalState.SUBMITTED, SignalState.EXPIRED, SignalState.CANCELLED},
    SignalState.SUBMITTED: {SignalState.FILLED, SignalState.REJECTED, SignalState.CANCELLED},
    SignalState.FILLED: {SignalState.MANAGED},
    SignalState.MANAGED: {SignalState.CLOSED},
    # Terminal states:
    SignalState.REJECTED: set(),
    SignalState.CANCELLED: set(),
    SignalState.EXPIRED: set(),
    SignalState.BLOCKED: set(),
    SignalState.CLOSED: set(),
}


def can_transition(current: SignalState, target: SignalState) -> bool:
    return target in _SIGNAL_TRANSITIONS.get(current, set())


def validate_transition(current: SignalState, target: SignalState):
    if not can_transition(current, target):
        raise ValueError(f"Invalid signal transition: {current.value} -> {target.value}")


@dataclass
class SignalLifecycle:
    """Tracks a single signal from detection to close."""
    signal_id: str
    symbol: str
    side: str
    state: SignalState = SignalState.DETECTED
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = [(self.state, None)]

    def transition(self, target: SignalState, reason: str = None):
        validate_transition(self.state, target)
        self.state = target
        self.history.append((target, reason))
        return self
```

- [ ] **Step 2:** Smoke test

```python
# tmp_test_sm.py
from server.core.state_machines import SignalLifecycle, SignalState

lc = SignalLifecycle(signal_id='abc', symbol='BTCUSDT', side='long')
lc.transition(SignalState.VALIDATED)
lc.transition(SignalState.RISK_CHECKED)
lc.transition(SignalState.READY)
lc.transition(SignalState.SUBMITTED)
lc.transition(SignalState.FILLED)
lc.transition(SignalState.MANAGED)
lc.transition(SignalState.CLOSED)

try:
    lc.transition(SignalState.READY)  # should fail — closed is terminal
    raise AssertionError('should have rejected')
except ValueError:
    print('OK — terminal CLOSED blocks re-entry')

lc2 = SignalLifecycle('def', 'ETHUSDT', 'short')
try:
    lc2.transition(SignalState.FILLED)  # skip states
    raise AssertionError('should reject')
except ValueError:
    print('OK — cannot skip from DETECTED to FILLED')
```

- [ ] **Step 3:** Commit

```bash
git add server/core/state_machines.py
git commit -m "feat(phase3): add SignalLifecycle state machine"
```

---

### Task 3: Wire event bus into agent_brain.py

**Files:**
- Modify: `server/agent_brain.py`

Goal: every time the agent generates/blocks/opens/closes a signal, publish an event. Keep existing behavior (audit log, Telegram, prints) working — events are ADDITIVE.

- [ ] **Step 1:** Add bus import at top of agent_brain.py

```python
from .core.events import (
    bus,
    make_signal_detected, make_signal_blocked,
    make_position_opened, make_position_closed,
    make_agent_started, make_agent_regime_changed,
    make_risk_limit_hit, make_summary_daily,
)
```

- [ ] **Step 2:** Add publish calls at key points

In `tick()` when a signal is generated (after Layer 5 confidence check):
```python
await bus.publish(make_signal_detected(
    symbol=symbol, side=action, confidence=confidence,
    reason=reason, price=price, sl=sl, tp=tp,
    regime=self.market_regime,
))
```

In `PreTradeChecklist.validate` caller when blocked:
```python
await bus.publish(make_signal_blocked(symbol, action, failures))
```

In position open (after successful `open_position`):
```python
await bus.publish(make_position_opened(symbol, side, size, entry_price, sl, tp))
```

In position close (after successful `close_position`):
```python
await bus.publish(make_position_closed(symbol, side, entry_price, exit_price, pnl_pct, pnl_usd, reason))
```

In `start()`:
```python
await bus.publish(make_agent_started(self.state.mode, self.state.equity, self.state.generation))
```

When regime changes in `LessonsLedger.detect_regime`:
```python
if regime != self.market_regime:
    await bus.publish(make_agent_regime_changed(self.market_regime, regime, confidence))
```

- [ ] **Step 3:** Smoke test by subscribing a test handler before agent tick and verifying events arrive

- [ ] **Step 4:** Commit

```bash
git add server/agent_brain.py
git commit -m "feat(phase3): publish lifecycle events from agent brain"
```

---

### Task 4: Create subscribers/ package

**Files:**
- Create: `server/subscribers/__init__.py`
- Create: `server/subscribers/audit.py`
- Create: `server/subscribers/telegram.py`
- Create: `server/subscribers/sse_broadcast.py`

Each subscriber is an async function registered at app startup.

- [ ] **Step 1:** Implement `subscribers/audit.py`

```python
# server/subscribers/audit.py
"""Writes all events to trade_audit.jsonl (replaces scattered audit logging)."""

import json
from pathlib import Path
from ..core.config import PROJECT_ROOT
from ..core.events import bus, Event

AUDIT_LOG = PROJECT_ROOT / "trade_audit.jsonl"


async def audit_writer(event: Event):
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, 'a', encoding='utf-8') as f:
            f.write(event.to_json() + '\n')
    except Exception as e:
        print(f"[Audit] Write failed: {e}")


def register():
    bus.subscribe('*', audit_writer)  # NOTE: wildcard '*' is bus-level — see events.py for match semantics
```

**Fix:** The wildcard `*` alone isn't currently supported. Use explicit prefixes:

```python
def register():
    for prefix in ['signal.*', 'order.*', 'position.*', 'risk.*', 'agent.*', 'ops.*', 'summary.*']:
        bus.subscribe(prefix, audit_writer)
```

- [ ] **Step 2:** Implement `subscribers/telegram.py`

```python
# server/subscribers/telegram.py
"""Sends events to Telegram (replaces _notify_telegram scattered calls)."""

import os
import httpx
from ..core.events import bus, Event

_cfg = None


def _load_config():
    global _cfg
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
    if token and chat_id:
        _cfg = {'token': token, 'chat_id': chat_id}


async def _send(text: str):
    if not _cfg:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{_cfg['token']}/sendMessage",
                json={'chat_id': _cfg['chat_id'], 'text': text, 'parse_mode': 'HTML'},
            )
    except Exception as e:
        print(f"[Telegram] send failed: {e}")


async def on_signal_blocked(event: Event):
    p = event.payload
    await _send(f"🚫 Signal blocked {p['symbol']} {p['side']}\nReasons: {', '.join(p['block_reasons'][:3])}")


async def on_position_opened(event: Event):
    p = event.payload
    arrow = '📈' if p['side'] == 'long' else '📉'
    await _send(
        f"{arrow} <b>{p['side'].upper()} {p['symbol']}</b>\n"
        f"Size: ${p['size']:.0f} | Entry: ${p['entry_price']}\n"
        f"SL: ${p['sl']} | TP: ${p['tp']}"
    )


async def on_position_closed(event: Event):
    p = event.payload
    emoji = '🟢' if p['pnl_usd'] >= 0 else '🔴'
    await _send(
        f"{emoji} <b>Closed {p['symbol']}</b>\n"
        f"P&L: {p['pnl_pct']:+.2f}% (${p['pnl_usd']:+.2f})\n"
        f"Reason: {p['reason']}"
    )


async def on_risk_limit_hit(event: Event):
    p = event.payload
    await _send(f"⚠️ Risk limit hit: {p['limit']} ({p['current']}/{p['max']})")


async def on_agent_started(event: Event):
    p = event.payload
    await _send(
        f"🚀 Agent started\n"
        f"Mode: {p['mode']} | Equity: ${p['equity']:.2f} | Gen: {p['generation']}"
    )


async def on_summary_daily(event: Event):
    p = event.payload
    await _send(
        f"📊 Daily Summary\n"
        f"Equity: ${p['equity']:.2f}\n"
        f"Daily PnL: ${p['daily_pnl']:+.2f}\n"
        f"Trades: {p['total_trades']} | Win rate: {p['win_rate']:.0f}%"
    )


def register():
    _load_config()
    if not _cfg:
        print("[Telegram subscriber] No config, skipping registration")
        return
    bus.subscribe('signal.blocked', on_signal_blocked)
    bus.subscribe('position.opened', on_position_opened)
    bus.subscribe('position.closed', on_position_closed)
    bus.subscribe('risk.limit.hit', on_risk_limit_hit)
    bus.subscribe('agent.started', on_agent_started)
    bus.subscribe('summary.daily', on_summary_daily)
```

- [ ] **Step 3:** Implement `subscribers/sse_broadcast.py`

```python
# server/subscribers/sse_broadcast.py
"""Queues events for SSE delivery to frontend. Each connected client has a queue."""

import asyncio
from ..core.events import bus, Event

# Set of asyncio.Queue — one per connected SSE client
_client_queues: set = set()


def add_client() -> asyncio.Queue:
    q = asyncio.Queue(maxsize=200)
    _client_queues.add(q)
    return q


def remove_client(q: asyncio.Queue):
    _client_queues.discard(q)


async def _broadcast(event: Event):
    dead = []
    for q in _client_queues:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _client_queues.discard(q)


def register():
    for prefix in ['signal.*', 'order.*', 'position.*', 'risk.*', 'agent.*', 'ops.*', 'summary.*']:
        bus.subscribe(prefix, _broadcast)
```

- [ ] **Step 4:** Commit

```bash
git add server/subscribers/
git commit -m "feat(phase3): add audit, telegram, and SSE broadcast subscribers"
```

---

### Task 5: Register subscribers at app startup

**Files:**
- Modify: `server/app.py`

- [ ] **Step 1:** Add subscriber registration in the `_startup` lifecycle hook

```python
from .subscribers import audit, telegram, sse_broadcast

@app.on_event("startup")
async def _startup():
    # ... existing init ...
    audit.register()
    telegram.register()
    sse_broadcast.register()
    print("[EventBus] Subscribers registered")
```

- [ ] **Step 2:** Commit

---

### Task 6: Create SSE stream router

**Files:**
- Create: `server/routers/stream.py`
- Modify: `server/app.py` (register new router)

- [ ] **Step 1:** Implement stream router

```python
# server/routers/stream.py
"""Server-Sent Events stream of all events — consumed by frontend Decision Rail and Glassbox."""

import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..subscribers.sse_broadcast import add_client, remove_client

router = APIRouter(tags=["stream"])


async def _event_generator(queue: asyncio.Queue):
    try:
        # Send initial ping
        yield f"event: ping\ndata: {{}}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                data = json.dumps(event.to_dict(), default=str)
                yield f"event: {event.type}\ndata: {data}\n\n"
            except asyncio.TimeoutError:
                yield f"event: ping\ndata: {{}}\n\n"
    finally:
        remove_client(queue)


@router.get("/api/stream")
async def api_stream():
    """Server-Sent Events stream of all agent events."""
    queue = add_client()
    return StreamingResponse(
        _event_generator(queue),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )
```

- [ ] **Step 2:** Register in `app.py`

```python
from .routers import stream
app.include_router(stream.router)
```

- [ ] **Step 3:** Smoke test

```bash
# In one terminal:
curl -N http://127.0.0.1:8003/api/stream

# In another, trigger an agent event (e.g., start agent)
curl -X POST http://127.0.0.1:8003/api/agent/start

# First terminal should receive SSE events in real time
```

- [ ] **Step 4:** Commit

---

### Task 7: Remove scattered Telegram / audit calls from agent_brain.py

Now that events go through subscribers, clean up:

- Remove direct `_notify_telegram` calls (subscribers handle this)
- Remove direct `_audit_log` calls for events already emitted (subscribers handle this)
- Keep `print()` calls for console visibility

- [ ] **Step 1:** Delete `_notify_telegram` method and `_telegram_config` logic from agent_brain.py
- [ ] **Step 2:** Delete `_audit_log` call sites that are now covered by events
- [ ] **Step 3:** Verify via smoke test: start agent, force a signal, confirm Telegram still sends and audit log still writes (but now via subscribers)
- [ ] **Step 4:** Commit

---

### Task 8: Frontend SSE consumer (in Phase 2 modules)

**Depends on:** Phase 2 complete

**Files:**
- Modify: `frontend/js/state/ui.js` — add `lastEvents` ring buffer
- Create: `frontend/js/services/stream.js` — EventSource wrapper
- Modify: `frontend/js/workbench/decision_rail.js` — consume `signal.*` and `agent.regime.*` events
- Modify: `frontend/js/execution/execution_tab.js` — consume `signal.blocked` → blocked reasons feed
- Modify: `frontend/js/execution/risk_tab.js` — consume `risk.*` → live meters

- [ ] **Step 1:** Implement `services/stream.js`

```javascript
// frontend/js/services/stream.js
import { publish } from '../util/events.js';

let source = null;

export function connectStream() {
  if (source) return source;
  source = new EventSource('/api/stream');

  // Forward every event type to the UI pub/sub
  const types = [
    'signal.detected', 'signal.validated', 'signal.blocked', 'signal.expired',
    'order.submitted', 'order.filled', 'order.rejected',
    'position.opened', 'position.closed', 'position.sl_hit', 'position.tp_hit',
    'risk.limit.hit', 'risk.cooldown.started', 'risk.cooldown.ended',
    'agent.started', 'agent.stopped', 'agent.regime.changed',
    'summary.daily',
    'ping',
  ];
  for (const t of types) {
    source.addEventListener(t, (ev) => {
      try { publish(t, JSON.parse(ev.data)); } catch {}
    });
  }

  source.onerror = () => {
    console.warn('[stream] disconnected, will retry');
    setTimeout(() => { source = null; connectStream(); }, 3000);
  };

  return source;
}
```

- [ ] **Step 2:** Wire into `main.js` boot

```javascript
import { connectStream } from './services/stream.js';
connectStream();
```

- [ ] **Step 3:** Decision Rail subscribes to signal events

```javascript
// frontend/js/workbench/decision_rail.js
import { subscribe } from '../util/events.js';

let currentState = {
  regime: null,
  setup_score: null,
  trigger: null,
  stop: null,
  target: null,
  block_reason: null,
};

function render() { /* render to DOM */ }

export function initDecisionRail() {
  subscribe('signal.detected', (e) => {
    currentState.trigger = e.payload.reason;
    currentState.stop = e.payload.sl;
    currentState.target = e.payload.tp;
    currentState.block_reason = null;
    render();
  });
  subscribe('signal.blocked', (e) => {
    currentState.block_reason = e.payload.block_reasons.join(', ');
    render();
  });
  subscribe('agent.regime.changed', (e) => {
    currentState.regime = e.payload.to;
    render();
  });
  render();
}
```

- [ ] **Step 4:** Commit

---

### Task 9: Risk meter live updates

**Files:**
- Modify: `frontend/js/execution/risk_tab.js`

Add live exposure meters that consume `risk.*` events. When a `risk.limit.hit` fires, flash the meter red.

- [ ] **Step 1:** Implement meter rendering + subscription
- [ ] **Step 2:** Smoke test
- [ ] **Step 3:** Commit

---

### Task 10: Final integration smoke test

- [ ] Agent start → Telegram receives startup message (via subscriber)
- [ ] Agent generates signal → frontend Decision Rail updates via SSE
- [ ] Signal blocked → Execution tab blocked feed updates, Telegram notifies
- [ ] Position opens → Telegram notifies, positions table updates
- [ ] Position closes → Telegram notifies with P&L
- [ ] Regime changes → Decision Rail regime badge updates
- [ ] Daily summary → Telegram receives summary

Commit final tag: `git tag phase3-complete`

---

## 6. Phase 3 Success Criteria

- All signal/order/position lifecycle events are published through the bus
- No direct Telegram calls in agent_brain.py (only through subscribers)
- SSE stream works — frontend receives events in < 500ms of publication
- Old audit log format preserved (trade_audit.jsonl still valid)
- Existing UI still works (no regression)
- Signal state machine rejects invalid transitions
- Bot (Phase 4) can subscribe to the bus as a consumer
