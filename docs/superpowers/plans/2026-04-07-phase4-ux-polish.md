# Phase 4: UX Polish & Control Bus Integration — Implementation Plan

> **Execution mode:** Inline. All Phase 4 tasks are additive (no destructive refactors). Each task produces a testable feature.

**Goal:** Turn the layered Trading OS from Phase 2/3 into a professional trader workstation with (1) Cmd+K command palette, (2) glassbox Agent visualization, (3) combat mode full-screen view, (4) slippage/R-multiple preview, (5) context-aware chat, (6) bot subscriber integration, (7) scenario-based dashboard presets.

**Architecture:** No architecture changes — Phase 4 layers features on top of the Phase 2 modules and Phase 3 event bus. The Telegram bot is refactored to consume events via the same mechanism the frontend uses (either polling `/api/agent/audit-log` or subscribing to `/api/stream`).

**Tech Stack:** Same as Phase 2/3. No new dependencies except `fzf.js`-style fuzzy matcher (implemented inline, no npm package).

**Prerequisites:**
- Phase 1 complete (routers split)
- Phase 2 complete (frontend 4 layers)
- Phase 3 complete (event bus + SSE)

**Non-goals:**
- Phase 4 does NOT replace the Telegram bot project — it makes the bot subscribe to events instead of duplicating UI logic
- No mobile-responsive redesign (already exists)
- No theme switching (dark theme only)

---

## 1. Feature Overview

| Feature | Description | Priority |
|---|---|---|
| A. Command Palette (Cmd+K) | Fuzzy search to jump between symbols, switch timeframes, open panels, run actions | High |
| B. Glassbox Agent | Timeline UI showing every tick's reasoning (regime → signal → validation → decision) | High |
| C. Combat Mode | Full-screen chart with minimal chrome, for focused discretionary trading | Medium |
| D. Order Ticket Preview | Show expected P&L, fees, slippage, R-multiple before submitting | High |
| E. Context-Aware Chat | AI chat auto-injects current symbol/timeframe/regime/positions into every message | Medium |
| F. Bot as Event Subscriber | Refactor claude-tg-bot project to consume `/api/stream` instead of owning frontend state | High |
| G. Dashboard Scenarios | Save/load UI presets (e.g., "day trading", "swing watching", "research mode") | Low |
| H. Risk Dashboard Meters | Real-time exposure/drawdown/cooldown bars (depends on Phase 3 risk events) | High |
| I. Blocked Reasons Feed | Live feed of why signals were rejected (exposed from Phase 3 `signal.blocked` events) | Medium |
| J. Session Replay | Scrub through the last N minutes of events to understand what happened | Low |

---

## 2. Target File Additions

```
frontend/
├── js/
│   ├── command_palette/
│   │   ├── palette.js            # NEW: Cmd+K modal UI
│   │   ├── commands.js           # NEW: Command registry (declarative list of actions)
│   │   └── fuzzy.js              # NEW: Simple fuzzy matcher (no dependency)
│   │
│   ├── workbench/
│   │   └── combat_mode.js        # NEW: Full-screen chart toggle
│   │
│   ├── execution/
│   │   ├── order_ticket.js       # NEW: Order intent preview (symbol, side, size, SL/TP, fees, R, slippage)
│   │   ├── risk_meters.js        # NEW: Live bars (exposure, daily loss, drawdown, positions, cooldown)
│   │   └── blocked_feed.js       # NEW: Stream of recent signal.blocked events
│   │
│   ├── control/
│   │   ├── chat_context.js       # NEW: Context injector — wraps sendChat with current state snapshot
│   │   └── glassbox.js           # NEW: Agent glassbox timeline view
│   │
│   ├── scenarios/
│   │   ├── manager.js            # NEW: Save/load dashboard presets to localStorage
│   │   └── presets.js            # NEW: Built-in default scenarios
│   │
│   └── session_replay/
│       └── replay.js             # NEW: Event history scrubber

server/
├── core/
│   └── slippage.py               # NEW: Expected slippage estimation based on volume/liquidity
│
└── routers/
    └── execution.py              # MODIFIED: Add POST /api/execution/ticket-preview endpoint
```

---

## 3. Tasks

### Task 1: Command Palette (Cmd+K)

**Files:**
- Create: `frontend/js/command_palette/palette.js`, `commands.js`, `fuzzy.js`
- Modify: `frontend/js/main.js` (mount palette + register Ctrl+K keybinding)

- [ ] **Step 1:** Implement `fuzzy.js` (simple subsequence matcher)

```javascript
// frontend/js/command_palette/fuzzy.js
export function fuzzyMatch(query, text) {
  if (!query) return { score: 0, indexes: [] };
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  let score = 0;
  let qi = 0;
  const indexes = [];
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      indexes.push(ti);
      score += 10;
      if (indexes.length > 1 && indexes[indexes.length - 1] === indexes[indexes.length - 2] + 1) {
        score += 5;  // bonus for consecutive
      }
      qi++;
    }
  }
  return qi === q.length ? { score, indexes } : null;
}

export function fuzzySort(query, items, textFn = (x) => x) {
  return items
    .map(item => ({ item, match: fuzzyMatch(query, textFn(item)) }))
    .filter(x => x.match)
    .sort((a, b) => b.match.score - a.match.score);
}
```

- [ ] **Step 2:** Implement `commands.js`

```javascript
// frontend/js/command_palette/commands.js
import { marketState, setSymbol, setInterval } from '../state/market.js';
import { uiState } from '../state/ui.js';
import * as agentService from '../services/agent.js';

export function buildCommands() {
  const cmds = [
    // Symbol jumps (dynamic — one per symbol)
    ...marketState.allSymbols.map(sym => ({
      id: `symbol:${sym}`,
      label: `Switch to ${sym}`,
      category: 'Symbol',
      action: () => setSymbol(sym),
    })),
    // Timeframes
    ...['5m', '15m', '1h', '4h', '1d'].map(tf => ({
      id: `tf:${tf}`,
      label: `Timeframe ${tf}`,
      category: 'Timeframe',
      action: () => setInterval(tf),
    })),
    // Panels
    { id: 'panel:execution', label: 'Open Execution Center', category: 'Panel',
      action: () => { uiState.activeLayer = 'execution'; /* trigger open */ } },
    { id: 'panel:research', label: 'Open Research Drawer', category: 'Panel',
      action: () => { uiState.researchDrawerOpen = true; } },
    { id: 'panel:chat', label: 'Open Chat Dock', category: 'Panel',
      action: () => { uiState.chatDockOpen = true; } },
    // Actions
    { id: 'agent:start', label: 'Start Agent', category: 'Agent', action: () => agentService.start() },
    { id: 'agent:stop', label: 'Stop Agent', category: 'Agent', action: () => agentService.stop() },
    { id: 'agent:revive', label: 'Revive Agent', category: 'Agent', action: () => agentService.revive() },
    { id: 'agent:scan', label: 'Scan Now', category: 'Agent', action: () => agentService.getSignals() },
    // Combat mode
    { id: 'view:combat', label: 'Enter Combat Mode', category: 'View',
      action: () => document.body.classList.toggle('combat-mode') },
  ];
  return cmds;
}
```

- [ ] **Step 3:** Implement `palette.js`

```javascript
// frontend/js/command_palette/palette.js
import { $ } from '../util/dom.js';
import { fuzzySort } from './fuzzy.js';
import { buildCommands } from './commands.js';

let overlayEl = null;
let inputEl = null;
let listEl = null;
let open = false;

function build() {
  overlayEl = document.createElement('div');
  overlayEl.id = 'cmdk-overlay';
  overlayEl.className = 'cmdk-overlay hidden';
  overlayEl.innerHTML = `
    <div class="cmdk-modal">
      <input class="cmdk-input" type="text" placeholder="Type a command or symbol..." autofocus>
      <ul class="cmdk-list"></ul>
    </div>
  `;
  document.body.appendChild(overlayEl);
  inputEl = overlayEl.querySelector('.cmdk-input');
  listEl = overlayEl.querySelector('.cmdk-list');

  overlayEl.addEventListener('click', (e) => { if (e.target === overlayEl) close(); });
  inputEl.addEventListener('input', render);
  inputEl.addEventListener('keydown', onKey);
}

let selectedIdx = 0;
let results = [];

function render() {
  const q = inputEl.value;
  const cmds = buildCommands();
  results = q ? fuzzySort(q, cmds, c => c.label) : cmds.slice(0, 20).map(c => ({ item: c }));
  selectedIdx = 0;

  listEl.innerHTML = results.map((r, i) => `
    <li class="cmdk-item ${i === selectedIdx ? 'selected' : ''}" data-idx="${i}">
      <span class="cmdk-label">${r.item.label}</span>
      <span class="cmdk-category">${r.item.category}</span>
    </li>
  `).join('');

  Array.from(listEl.children).forEach((el, i) => {
    el.addEventListener('click', () => { selectedIdx = i; execute(); });
  });
}

function onKey(e) {
  if (e.key === 'Escape') { close(); return; }
  if (e.key === 'ArrowDown') { e.preventDefault(); selectedIdx = Math.min(selectedIdx + 1, results.length - 1); updateSelection(); return; }
  if (e.key === 'ArrowUp')   { e.preventDefault(); selectedIdx = Math.max(selectedIdx - 1, 0); updateSelection(); return; }
  if (e.key === 'Enter')     { e.preventDefault(); execute(); return; }
}

function updateSelection() {
  Array.from(listEl.children).forEach((el, i) =>
    el.classList.toggle('selected', i === selectedIdx)
  );
  listEl.children[selectedIdx]?.scrollIntoView({ block: 'nearest' });
}

function execute() {
  const entry = results[selectedIdx];
  if (!entry) return;
  try { entry.item.action(); } catch (err) { console.error('[cmdk] action failed', err); }
  close();
}

export function openPalette() {
  if (!overlayEl) build();
  open = true;
  overlayEl.classList.remove('hidden');
  inputEl.value = '';
  inputEl.focus();
  render();
}

export function close() {
  if (!overlayEl) return;
  open = false;
  overlayEl.classList.add('hidden');
}

export function initCommandPalette() {
  document.addEventListener('keydown', (e) => {
    const ctrlK = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k';
    if (ctrlK) {
      e.preventDefault();
      open ? close() : openPalette();
    }
  });
}
```

- [ ] **Step 4:** Add CSS for palette

```css
/* Add to frontend/style.css */
.cmdk-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 9999; display: flex; justify-content: center; padding-top: 15vh; }
.cmdk-overlay.hidden { display: none; }
.cmdk-modal { background: #1a2035; border-radius: 12px; width: 580px; max-height: 60vh; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.5); }
.cmdk-input { width: 100%; padding: 16px 20px; font-size: 16px; background: transparent; border: none; color: #e0e6ed; border-bottom: 1px solid #2a3548; outline: none; }
.cmdk-list { list-style: none; margin: 0; padding: 8px 0; max-height: 45vh; overflow-y: auto; }
.cmdk-item { display: flex; justify-content: space-between; padding: 10px 20px; cursor: pointer; color: #e0e6ed; }
.cmdk-item.selected, .cmdk-item:hover { background: #2a3548; }
.cmdk-category { color: #6b7280; font-size: 12px; }
```

- [ ] **Step 5:** Wire into main.js

```javascript
import { initCommandPalette } from './command_palette/palette.js';
initCommandPalette();
```

- [ ] **Step 6:** Smoke test: press Ctrl+K → palette opens → type "BTC" → select → chart switches

- [ ] **Step 7:** Commit

---

### Task 2: Risk Dashboard Meters

**Files:**
- Create: `frontend/js/execution/risk_meters.js`
- Modify: `frontend/js/execution/risk_tab.js` (mount meters)

- [ ] **Step 1:** Implement meter component

```javascript
// frontend/js/execution/risk_meters.js
import { $, setHtml } from '../util/dom.js';
import { subscribe } from '../util/events.js';
import * as agentService from '../services/agent.js';

const meters = [
  { key: 'daily_loss', label: 'Daily Loss', format: v => `${v.toFixed(2)}%` },
  { key: 'drawdown', label: 'Drawdown', format: v => `${v.toFixed(2)}%` },
  { key: 'exposure', label: 'Exposure', format: v => `${v.toFixed(1)}%` },
  { key: 'positions', label: 'Positions', format: v => `${v.toFixed(0)}` },
];

function meterHtml(label, current, max, format, color) {
  const pct = max > 0 ? Math.min(100, (current / max) * 100) : 0;
  return `
    <div class="meter ${color}">
      <div class="meter-label">${label}</div>
      <div class="meter-bar"><div class="meter-fill" style="width:${pct}%"></div></div>
      <div class="meter-value">${format(current)} / ${format(max)}</div>
    </div>
  `;
}

function colorForPct(pct) {
  if (pct < 50) return 'green';
  if (pct < 80) return 'yellow';
  return 'red';
}

export async function renderRiskMeters() {
  const container = $('#risk-meters');
  if (!container) return;

  const status = await agentService.getStatus();
  const risk = status.risk_limits || {};
  const state = status;

  const daily_loss_cur = Math.max(0, -state.daily_pnl / state.equity * 100);
  const daily_loss_max = risk.max_daily_loss_pct;
  const dd_cur = Math.max(0, (state.peak_equity - state.equity) / state.peak_equity * 100);
  const dd_max = risk.max_drawdown_pct;
  const exposure_cur = Object.values(state.positions || {}).reduce((s, p) => s + (p.size_usd || 0), 0) / state.equity * 100;
  const exposure_max = risk.max_total_exposure_pct;
  const pos_cur = Object.keys(state.positions || {}).length;
  const pos_max = risk.max_positions;

  setHtml(container, `
    ${meterHtml('Daily Loss', daily_loss_cur, daily_loss_max, v => `${v.toFixed(2)}%`, colorForPct(daily_loss_cur / daily_loss_max * 100))}
    ${meterHtml('Drawdown', dd_cur, dd_max, v => `${v.toFixed(2)}%`, colorForPct(dd_cur / dd_max * 100))}
    ${meterHtml('Exposure', exposure_cur, exposure_max, v => `${v.toFixed(1)}%`, colorForPct(exposure_cur / exposure_max * 100))}
    ${meterHtml('Positions', pos_cur, pos_max, v => `${v}`, colorForPct(pos_cur / pos_max * 100))}
  `);
}

export function initRiskMeters() {
  renderRiskMeters();
  // Refresh on risk events
  subscribe('risk.limit.hit', renderRiskMeters);
  subscribe('position.opened', renderRiskMeters);
  subscribe('position.closed', renderRiskMeters);
  // And on a timer
  setInterval(renderRiskMeters, 5000);
}
```

- [ ] **Step 2:** Add CSS for meters

```css
.meter { margin-bottom: 12px; }
.meter-label { font-size: 12px; color: #9ca3af; margin-bottom: 4px; }
.meter-bar { height: 8px; background: #2a3548; border-radius: 4px; overflow: hidden; }
.meter-fill { height: 100%; transition: width 0.3s ease, background 0.3s ease; }
.meter.green .meter-fill { background: #00e676; }
.meter.yellow .meter-fill { background: #ffc107; }
.meter.red .meter-fill { background: #ff1744; }
.meter-value { font-size: 11px; color: #e0e6ed; margin-top: 2px; text-align: right; }
```

- [ ] **Step 3:** Commit

---

### Task 3: Blocked Reasons Feed

**Files:**
- Create: `frontend/js/execution/blocked_feed.js`

- [ ] **Step 1:** Implement feed subscribing to `signal.blocked`

```javascript
// frontend/js/execution/blocked_feed.js
import { $, setHtml } from '../util/dom.js';
import { subscribe } from '../util/events.js';

const maxItems = 20;
const items = [];

function render() {
  const container = $('#blocked-feed');
  if (!container) return;
  if (items.length === 0) {
    setHtml(container, '<div class="empty">No blocked signals yet</div>');
    return;
  }
  setHtml(container, items.map(item => `
    <div class="blocked-item">
      <div class="blocked-header">
        <span class="blocked-symbol">${item.symbol}</span>
        <span class="blocked-side ${item.side}">${item.side.toUpperCase()}</span>
        <span class="blocked-time">${new Date(item.ts).toLocaleTimeString()}</span>
      </div>
      <ul class="blocked-reasons">
        ${item.reasons.map(r => `<li>${r}</li>`).join('')}
      </ul>
    </div>
  `).join(''));
}

export function initBlockedFeed() {
  subscribe('signal.blocked', (payload) => {
    items.unshift({
      symbol: payload.symbol,
      side: payload.side,
      reasons: payload.block_reasons,
      ts: Date.now(),
    });
    if (items.length > maxItems) items.length = maxItems;
    render();
  });
  render();
}
```

- [ ] **Step 2:** Commit

---

### Task 4: Order Ticket Preview

**Files:**
- Create: `server/core/slippage.py`
- Modify: `server/routers/execution.py` (add `/api/execution/ticket-preview`)
- Create: `frontend/js/execution/order_ticket.js`

- [ ] **Step 1:** Backend slippage estimator

```python
# server/core/slippage.py
"""Estimate expected slippage based on order size and symbol liquidity."""

FEE_BPS = 5  # 0.05% taker fee on OKX swap


def estimate_slippage_bps(symbol: str, size_usd: float, avg_volume_24h_usd: float) -> float:
    """Simple linear model: bigger orders relative to daily volume = more slippage."""
    if avg_volume_24h_usd <= 0:
        return 20.0  # 20 bps fallback
    ratio = size_usd / avg_volume_24h_usd
    return min(50.0, 5.0 + ratio * 1000)  # cap at 50 bps


def calculate_ticket(price: float, size_usd: float, sl: float, tp: float,
                      side: str, avg_volume: float, fee_bps: float = FEE_BPS):
    slippage_bps = estimate_slippage_bps("", size_usd, avg_volume)

    # Expected entry = price adjusted for slippage
    if side == "long":
        expected_entry = price * (1 + slippage_bps / 10000)
    else:
        expected_entry = price * (1 - slippage_bps / 10000)

    # Risk (stop distance)
    risk_pct = abs(expected_entry - sl) / expected_entry * 100
    risk_usd = size_usd * risk_pct / 100

    # Reward (target distance)
    reward_pct = abs(tp - expected_entry) / expected_entry * 100
    reward_usd = size_usd * reward_pct / 100

    # R-multiple
    r_multiple = reward_usd / max(risk_usd, 0.01)

    # Fees (entry + exit)
    fees = size_usd * fee_bps / 10000 * 2

    return {
        "expected_entry": round(expected_entry, 6),
        "slippage_bps": round(slippage_bps, 1),
        "risk_pct": round(risk_pct, 2),
        "risk_usd": round(risk_usd, 2),
        "reward_pct": round(reward_pct, 2),
        "reward_usd": round(reward_usd, 2),
        "r_multiple": round(r_multiple, 2),
        "fees_usd": round(fees, 2),
        "expected_net_profit": round(reward_usd - fees, 2),
        "expected_net_loss": round(-risk_usd - fees, 2),
    }
```

- [ ] **Step 2:** Add preview endpoint to `server/routers/execution.py`

```python
from pydantic import BaseModel
from ..core.slippage import calculate_ticket

class TicketPreviewRequest(BaseModel):
    symbol: str
    side: str
    size_usd: float
    price: float
    sl: float
    tp: float

@router.post("/ticket-preview")
async def api_ticket_preview(req: TicketPreviewRequest):
    """Preview expected fill, slippage, R-multiple, fees for a potential order."""
    # Fetch avg volume — simplified, use top-volume service
    from ..data_service import get_top_volume_symbols
    # ... get volume for req.symbol ...
    avg_volume = 1_000_000  # placeholder — TODO: fetch real 24h volume
    return calculate_ticket(req.price, req.size_usd, req.sl, req.tp, req.side, avg_volume)
```

- [ ] **Step 3:** Frontend ticket preview component

```javascript
// frontend/js/execution/order_ticket.js
import { fetchJson } from '../util/fetch.js';
import { $, setHtml } from '../util/dom.js';

export async function previewTicket({ symbol, side, size_usd, price, sl, tp }) {
  const result = await fetchJson('/api/agent/ticket-preview', {
    method: 'POST',
    body: { symbol, side, size_usd, price, sl, tp },
  });

  const container = $('#ticket-preview');
  if (!container) return;

  setHtml(container, `
    <div class="ticket">
      <h4>Order Preview</h4>
      <div class="ticket-row"><span>Expected Entry:</span><span>$${result.expected_entry}</span></div>
      <div class="ticket-row"><span>Slippage:</span><span>${result.slippage_bps} bps</span></div>
      <div class="ticket-row pnl-neg"><span>Risk:</span><span>-$${result.risk_usd} (-${result.risk_pct}%)</span></div>
      <div class="ticket-row pnl-pos"><span>Reward:</span><span>+$${result.reward_usd} (+${result.reward_pct}%)</span></div>
      <div class="ticket-row"><span>R-Multiple:</span><span><strong>${result.r_multiple}R</strong></span></div>
      <div class="ticket-row"><span>Fees:</span><span>$${result.fees_usd}</span></div>
      <div class="ticket-row pnl-pos"><span>Net profit if TP:</span><span>+$${result.expected_net_profit}</span></div>
      <div class="ticket-row pnl-neg"><span>Net loss if SL:</span><span>$${result.expected_net_loss}</span></div>
    </div>
  `);
}
```

- [ ] **Step 4:** Commit

---

### Task 5: Glassbox Agent Timeline

**Files:**
- Create: `frontend/js/control/glassbox.js`

- [ ] **Step 1:** Implement timeline subscribing to all agent events

```javascript
// frontend/js/control/glassbox.js
import { $, setHtml } from '../util/dom.js';
import { subscribe } from '../util/events.js';

const timeline = [];
const MAX = 100;

const ICONS = {
  'signal.detected': '🎯',
  'signal.validated': '✅',
  'signal.blocked': '🚫',
  'signal.expired': '⏰',
  'order.submitted': '📤',
  'order.filled': '✔️',
  'order.rejected': '❌',
  'position.opened': '🟢',
  'position.closed': '⚫',
  'risk.limit.hit': '⚠️',
  'agent.regime.changed': '🔄',
  'agent.started': '🚀',
  'agent.stopped': '🛑',
};

function formatEventLine(evType, payload, ts) {
  const icon = ICONS[evType] || '•';
  const time = new Date(ts).toLocaleTimeString();
  let text = evType;
  if (evType === 'signal.detected') text = `${payload.symbol} ${payload.side} @ ${payload.confidence} — ${payload.reason}`;
  else if (evType === 'signal.blocked') text = `${payload.symbol} ${payload.side} blocked: ${payload.block_reasons?.join('; ')}`;
  else if (evType === 'position.opened') text = `Opened ${payload.symbol} ${payload.side} size=$${payload.size}`;
  else if (evType === 'position.closed') text = `Closed ${payload.symbol} PnL ${payload.pnl_pct}%`;
  else if (evType === 'agent.regime.changed') text = `Regime ${payload.from} → ${payload.to}`;
  return `<div class="glass-row"><span class="glass-icon">${icon}</span><span class="glass-time">${time}</span><span class="glass-text">${text}</span></div>`;
}

function render() {
  const container = $('#glassbox-timeline');
  if (!container) return;
  setHtml(container, timeline.slice().reverse().map(e => formatEventLine(e.type, e.payload, e.ts)).join(''));
}

function handle(evType) {
  return (payload) => {
    timeline.push({ type: evType, payload, ts: Date.now() });
    if (timeline.length > MAX) timeline.shift();
    render();
  };
}

export function initGlassbox() {
  const types = Object.keys(ICONS);
  for (const t of types) subscribe(t, handle(t));
  render();
}
```

- [ ] **Step 2:** Commit

---

### Task 6: Context-Aware Chat

**Files:**
- Create: `frontend/js/control/chat_context.js`
- Modify: `frontend/js/control/chat.js` (use context injector)

- [ ] **Step 1:** Implement context injector

```javascript
// frontend/js/control/chat_context.js
import { marketState } from '../state/market.js';
import * as agentService from '../services/agent.js';

export async function buildContext() {
  let agentStatus = null;
  try { agentStatus = await agentService.getStatus(); } catch {}
  const positions = agentStatus?.positions || {};
  const posText = Object.values(positions).map(p =>
    `${p.symbol} ${p.side} size=$${p.size_usd} uPnL=${p.unrealized_pnl_pct?.toFixed(2)}%`
  ).join('; ') || 'none';

  return `[Context]
Symbol: ${marketState.currentSymbol}
Timeframe: ${marketState.currentInterval}
Agent mode: ${agentStatus?.mode || 'unknown'}
Agent equity: $${agentStatus?.equity || 0}
Market regime: ${agentStatus?.harness?.market_regime || 'unknown'}
Open positions: ${posText}
[End context]

User question: `;
}

export async function sendWithContext(userMessage, sendFn) {
  const ctx = await buildContext();
  return sendFn(ctx + userMessage);
}
```

- [ ] **Step 2:** Commit

---

### Task 7: Combat Mode

**Files:**
- Create: `frontend/js/workbench/combat_mode.js`

- [ ] **Step 1:** Implement full-screen toggle

```javascript
// frontend/js/workbench/combat_mode.js
let active = false;

export function toggleCombatMode() {
  active = !active;
  document.body.classList.toggle('combat-mode', active);
  window.dispatchEvent(new Event('resize'));  // Trigger chart resize
}

export function initCombatMode() {
  document.addEventListener('keydown', (e) => {
    if (e.key === 'F11' || (e.ctrlKey && e.key === '.')) {
      e.preventDefault();
      toggleCombatMode();
    }
  });
}
```

CSS:
```css
body.combat-mode #toolbar { display: none; }
body.combat-mode .chart-left { display: none; }
body.combat-mode #agent-panel,
body.combat-mode #onchain-panel,
body.combat-mode #chat-panel { display: none; }
body.combat-mode #chart-wrapper { position: fixed; inset: 0; z-index: 1000; }
```

- [ ] **Step 2:** Commit

---

### Task 8: Dashboard Scenarios

**Files:**
- Create: `frontend/js/scenarios/manager.js`, `presets.js`

- [ ] **Step 1:** Built-in scenarios

```javascript
// frontend/js/scenarios/presets.js
export const PRESETS = {
  day_trading: {
    name: 'Day Trading',
    layout: { activeLayer: 'workbench', researchDrawerOpen: false, chatDockOpen: false },
    market: { currentInterval: '5m' },
    chart: { maOverlayVisible: true, srVisible: true },
  },
  swing_watching: {
    name: 'Swing Watching',
    layout: { activeLayer: 'workbench', researchDrawerOpen: true },
    market: { currentInterval: '4h' },
    chart: { maOverlayVisible: true, srVisible: true },
  },
  research: {
    name: 'Research Mode',
    layout: { activeLayer: 'workbench', researchDrawerOpen: true, chatDockOpen: true },
    market: { currentInterval: '1d' },
  },
  combat: {
    name: 'Combat Mode',
    combatMode: true,
  },
};
```

- [ ] **Step 2:** Manager

```javascript
// frontend/js/scenarios/manager.js
import { PRESETS } from './presets.js';
import { marketState, setSymbol, setInterval } from '../state/market.js';
import { uiState } from '../state/ui.js';
import { loadJSON, saveJSON } from '../util/storage.js';

export function applyScenario(scenarioKey) {
  const p = PRESETS[scenarioKey] || loadJSON(`scenario:${scenarioKey}`);
  if (!p) return;
  if (p.market?.currentInterval) setInterval(p.market.currentInterval);
  if (p.market?.currentSymbol) setSymbol(p.market.currentSymbol);
  Object.assign(uiState, p.layout || {});
  // Trigger re-render
  window.dispatchEvent(new CustomEvent('scenario-applied', { detail: p }));
}

export function saveScenario(name) {
  saveJSON(`scenario:${name}`, {
    name,
    layout: { ...uiState },
    market: { currentSymbol: marketState.currentSymbol, currentInterval: marketState.currentInterval },
  });
}

export function listScenarios() {
  return Object.entries(PRESETS).map(([k, v]) => ({ key: k, name: v.name, builtin: true }));
}
```

- [ ] **Step 3:** Commit

---

### Task 9: Bot Event Subscriber Integration

**Files:**
- Modify: `claude-tg-bot` project (referenced externally — must be in its own repo)
- Create: Documentation `docs/bot-integration.md`

This task refactors the claude-tg-bot project so it subscribes to `/api/stream` instead of polling state.

- [ ] **Step 1:** Write integration doc `docs/bot-integration.md`

```markdown
# Bot Integration via Event Stream

The Telegram bot should consume `/api/stream` (SSE) from the main crypto-analysis server.

## Connection
```python
import httpx
import asyncio
import json

async def stream_events():
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream('GET', 'http://crypto-analysis:8001/api/stream') as resp:
            async for line in resp.aiter_lines():
                if line.startswith('event: '):
                    event_type = line[7:]
                elif line.startswith('data: '):
                    data = json.loads(line[6:])
                    await handle_event(event_type, data)
```

## Events to handle

| Event | Telegram Action |
|---|---|
| `signal.blocked` | Forward reason to user |
| `position.opened` | Send fill confirmation |
| `position.closed` | Send P&L summary |
| `risk.limit.hit` | High-priority alert |
| `agent.stopped` | Warning notification |
| `summary.daily` | Daily report |

## Commands bot can send back

Bot receives events but can also send commands via the existing HTTP API:
- `POST /api/agent/start` / `stop` / `revive`
- `POST /api/agent/config` (mode switch)
- `GET /api/agent/status` (on-demand status query)
```

- [ ] **Step 2:** Commit doc

Note: actual claude-tg-bot code changes happen in the bot's own repo.

---

### Task 10: Session Replay

**Files:**
- Create: `frontend/js/session_replay/replay.js`
- Modify: `server/routers/stream.py` (add `/api/events/history` endpoint)

- [ ] **Step 1:** Backend: expose event history from bus

Modify `server/routers/stream.py`:

```python
from ..core.events import bus

@router.get("/api/events/history")
async def api_events_history(limit: int = 100):
    events = bus.get_recent(limit)
    return {"events": [e.to_dict() for e in events]}
```

- [ ] **Step 2:** Frontend replay scrubber

```javascript
// frontend/js/session_replay/replay.js
import { fetchJson } from '../util/fetch.js';
import { publish } from '../util/events.js';

export async function loadHistory(limit = 100) {
  const { events } = await fetchJson(`/api/events/history?limit=${limit}`);
  return events;
}

export async function replayLast(minutes = 10) {
  const events = await loadHistory(200);
  const cutoff = Date.now() - minutes * 60000;
  const filtered = events.filter(e => new Date(e.ts).getTime() > cutoff);

  // Re-emit each event at its original relative time (accelerated 10x)
  const startWall = Date.now();
  const startOrig = new Date(filtered[0]?.ts).getTime();
  for (const ev of filtered) {
    const origDelta = new Date(ev.ts).getTime() - startOrig;
    const waitMs = origDelta / 10;  // 10x speedup
    const elapsed = Date.now() - startWall;
    if (waitMs > elapsed) await new Promise(r => setTimeout(r, waitMs - elapsed));
    publish(ev.type, ev.payload);
  }
}
```

- [ ] **Step 3:** Commit

---

## 4. Phase 4 Success Criteria

- [ ] Ctrl+K opens command palette, fuzzy searches work, all actions execute
- [ ] Risk meters update in real time when positions open/close
- [ ] Blocked reasons feed populates when signals are blocked
- [ ] Order ticket preview shows expected P&L and R-multiple before submission
- [ ] Glassbox timeline shows last N events with icons
- [ ] Chat auto-injects context (symbol/timeframe/positions) into every message
- [ ] Combat mode (F11 or Ctrl+.) hides all chrome, chart takes full screen
- [ ] Scenarios save/load/apply dashboard state
- [ ] Telegram bot (in its own repo) subscribes to `/api/stream` and forwards events
- [ ] Session replay can scrub through last 10 minutes of events

---

## 5. Final Deliverable

A professional trading workstation where:
1. Charts load fast with decision rail
2. Every agent action is visible in the glassbox
3. Command palette lets traders jump between symbols/timeframes/panels in under 1 second
4. Risk is visualized, not buried in config
5. The bot is an event consumer, not a parallel UI
6. Users can enter combat mode for discretionary trading without distractions
