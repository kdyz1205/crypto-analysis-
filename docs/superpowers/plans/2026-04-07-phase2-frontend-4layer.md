# Phase 2: Frontend 4-Layer Restructure — Implementation Plan

> **For agentic workers:** This plan is executed inline (single-threaded), not by subagents. The frontend is a single monolithic app.js (3366 lines) + index.html — splitting it requires careful sequencing, not parallelism.

**Goal:** Transform the single-page dashboard with 6 equal-weight tabs into a layered Trading OS workspace: Workbench (primary chart view), Execution Center (agent/risk panel), Research Lab (drawer), Control Bus (dock assistant + ops drawer).

**Architecture:** Preserve the vanilla-JS no-build-tool setup. Split `app.js` into ES modules loaded via `<script type="module">`. Split `index.html` into semantic regions. Preserve all current functionality — migration is additive until the final switch.

**Tech Stack:** Vanilla JS (ES6 modules), LightweightCharts, no build tool.

**Prerequisites:**
- Phase 1 complete and pushed (commit `e6177f0`)
- Backend routers stable

**Non-goals for Phase 2:**
- No backend changes
- No new features — only reorganization
- No CSS framework migration (keep hand-written CSS)

---

## 1. Target File Structure

```
frontend/
├── index.html                  # Slimmed: minimal shell, loads module entry
├── style.css                   # Keep as-is initially (split in later task)
├── app.js                      # DEPRECATED — kept until final switchover
│
├── js/
│   ├── main.js                 # Module entry point: boot, imports, init sequence
│   ├── state/
│   │   ├── market.js           # { currentSymbol, currentInterval, allSymbols, lastCandles, pricePrecision, currentScale, magnetMode }
│   │   ├── agent.js            # { agentPanelOpen, agentPollTimer, _lastOKXBalanceFetch, agentStatus }
│   │   ├── patterns.js         # { toolMode, drawingMode, userDrawnLines, rawPatternData, srVisible, maxSRLines, selectedLineIndex, patternResponseCache }
│   │   ├── ui.js               # { chatPanelOpen, onchainPanelOpen, researchDrawerOpen, activeLayer }
│   │   └── index.js            # Re-exports + global state snapshot helper
│   │
│   ├── services/               # Thin API clients (one function per endpoint)
│   │   ├── market.js           # getSymbols, getOhlcv, getSymbolInfo, getTopVolume, getDataInfo
│   │   ├── patterns.js         # getPatterns, getPatternStats, getPatternFeatures, getLineSimilar, getCurrentVsHistory
│   │   ├── research.js         # runBacktest, runOptimize, getMaRibbon, getMaRibbonBacktest
│   │   ├── agent.js            # getStatus, start, stop, revive, setConfig, getSignals, getAuditLog, getLessons, getStrategyConfig/Params/Presets (+save/load/delete)
│   │   ├── risk.js             # setRiskLimits
│   │   ├── execution.js        # setOkxKeys, getOkxStatus
│   │   ├── ops.js              # setTelegramConfig, getLogs, getHealerStatus, triggerHealer
│   │   ├── chat.js             # sendChat, getModels, getHistory, clearHistory
│   │   └── onchain.js          # onchainHealth, getSmartMoneyWallets, trackWallet, getActivity, tokenAnalyze
│   │
│   ├── workbench/              # LAYER 1: Market observation
│   │   ├── chart.js            # initChart, loadData, startLiveUpdates, crosshair handlers, scale/magnet
│   │   ├── ticker.js           # renderTickerList, selectTicker, toolbar price strip
│   │   ├── timeframe.js        # timeframe button group handlers
│   │   ├── analysis.js         # UNIFIED recognize/draw/assist (replaces 3 separate tabs)
│   │   ├── drawing.js          # draw tool logic (trend line, horizontal), userDrawnLines, magnet snap
│   │   ├── patterns.js         # drawAllPatterns, drawTrendlines, drawConsolidationZones, filterByProximity
│   │   ├── ma_overlay.js       # drawMAOverlays, toggleMAOverlays, MA colors, MA legend
│   │   ├── replay.js           # replay controls, getReplayEndTime
│   │   ├── legends.js          # updateOHLCLegend, updateChartHeader, updateToolbarPrice, updateTrendIndicator
│   │   └── decision_rail.js    # NEW: Decision Card (regime, setup score, trigger, stop, target, grade)
│   │
│   ├── execution/              # LAYER 2: Trading control
│   │   ├── panel.js            # Execution Center panel toggle, sub-tab switcher
│   │   ├── overview.js         # Sub-tab: equity hero, PnL, positions, recent trades, regime
│   │   ├── execution_tab.js    # Sub-tab: current signals, fills, blocked reasons, scan now
│   │   ├── risk_tab.js         # Sub-tab: risk limits, exposure display, cooldown, kill switch
│   │   ├── ops_tab.js          # Sub-tab: OKX status, Telegram config, healer, logs
│   │   ├── strategy_form.js    # strategy-config + strategy-params + presets (moved from main agent panel)
│   │   └── controls.js         # Start/Stop/Revive buttons, mode switch paper/live
│   │
│   ├── research/               # LAYER 3: Verification/backtest drawer
│   │   ├── drawer.js           # Research drawer toggle + sub-tab switcher (Backtest / Replay / Similar / Ribbon)
│   │   ├── backtest.js         # runBacktest, render results table
│   │   ├── pattern_stats.js    # fetchPatternStats, renderSimilarLinesList, current-vs-history
│   │   └── ma_ribbon.js        # MA ribbon multi-timeframe view + backtest
│   │
│   ├── control/                # LAYER 4: Ops + external integrations
│   │   ├── chat.js             # Dock assistant: toggle, send message, model selector, context-aware
│   │   ├── onchain.js          # On-chain panel: wallets, tokens, signals, whale alerts
│   │   └── ops_drawer.js       # Unified ops drawer: healer, logs, healthcheck
│   │
│   └── util/
│       ├── dom.js              # $(sel), $$(sel), on(el, ev, fn), show/hide, toggleClass
│       ├── format.js           # formatPrice, inferPrecision, formatPct, formatUsd
│       ├── fetch.js            # wrapFetch with timeout + abort + JSON parse
│       ├── events.js           # Lightweight pub/sub — subscribe(ev, fn), publish(ev, data) — NOT Phase 3 bus, just UI-local
│       └── storage.js          # localStorage wrappers
```

**Module loading strategy:**
- `index.html` loads `<script type="module" src="/js/main.js"></script>` only
- `main.js` imports all necessary modules and kicks off boot sequence
- Services are imported where needed; state modules export plain objects + setters
- No bundler — browser ES modules handle it natively

**File path routing:** Add server routes in `server/routers/ops.py` to serve `/js/*` as static files (matching existing `/style.css`, `/app.js` pattern).

---

## 2. Layer Responsibility Matrix

### Layer 1 — Workbench (Market Observation)

**Purpose:** Answer "what is happening in this market right now?" First screen a user sees.

**UI elements owned:**
- Chart (candles, volume, overlays)
- Symbol/timeframe toolbar (top)
- Analysis workspace (left sidebar): unified recognize/draw/assist mode picker
- Drawing tools (trend line, horizontal line, undo, clear)
- S/R lines visibility + max count
- MA overlay toggle + legend
- Replay controls
- Scale/magnet controls
- OHLC legend (crosshair)
- Trend indicator badge
- **NEW: Decision Rail (right side)** — market state, setup score, trigger, invalidator, stop, target, risk grade, block reason

**Backend domains consumed:**
- `market` (symbols, ohlcv, chart overlays)
- `patterns` (SR detection, pattern stats for selected line)

**State owned:**
- `state/market.js` — symbol, interval, candles, precision, scale, magnet
- `state/patterns.js` — tool mode, drawing mode, drawn lines, S/R data, selected line

**Polling:** 3s live candle updates (existing)

---

### Layer 2 — Execution Center (Trading Control)

**Purpose:** Answer "what is the agent doing and should I intervene?" Second layer user visits when making decisions.

**UI elements owned (split into 4 sub-tabs):**

**Sub-tab: Overview**
- Equity hero (total equity, source badge paper/live)
- PnL / Daily PnL / Win Rate
- Status / Cash / Trades / Regime / Gen / Phase
- Open positions table
- Recent trades table

**Sub-tab: Execution**
- Current signals (action, confidence, reason, regime)
- Blocked reasons feed (why signals failed validation)
- Manual Scan Now button
- Mode switch (Paper / Live)
- Start / Stop / Revive

**Sub-tab: Risk**
- Risk limits form (max position %, exposure %, daily loss %, drawdown %, max positions, cooldown)
- **NEW: Live exposure meters** (current/max for each limit, color coded)
- Kill switch state display
- Blocked signal feed filtered by risk

**Sub-tab: Ops**
- OKX API keys form + connectivity status
- Strategy config (timeframe, watch mode, symbols, tick interval)
- Strategy params grid + presets management

**Backend domains consumed:**
- `agent` (status, control, strategy config/params/presets, signals, audit-log, lessons)
- `risk` (risk-limits)
- `execution` (okx-keys, okx-status)

**State owned:**
- `state/agent.js` — panel open, poll timer, agent status snapshot

**Polling:** 5s agent status (existing)

---

### Layer 3 — Research Lab (Verification & Evolution)

**Purpose:** Answer "is this strategy/pattern historically proven?" Drawer-based — doesn't interrupt main flow.

**UI elements owned (bottom drawer with sub-tabs):**

**Sub-tab: Backtest**
- Strategy backtest results (symbol, interval, params, metrics table)
- Parameter override inputs
- Strategy preset selector (for quick backtest switching)

**Sub-tab: Replay** (relocated from toolbar)
- Date/hour/timezone controls
- Replay mode toggle
- Controls to step through historical bars

**Sub-tab: Similar Cases**
- Similar lines list (from pattern-stats/line-similar)
- Pattern stats current-vs-history display

**Sub-tab: Ribbon**
- MA ribbon multi-timeframe view (15m/1h/4h/1d alignment score)
- MA ribbon backtest results

**Backend domains consumed:**
- `research` (backtest, optimize, ma-ribbon)
- `patterns` (pattern-stats/*)

**State owned:** Transient — no global state (panels keep their own local view state)

---

### Layer 4 — Control Bus (Ops + External Integration)

**Purpose:** Background ops and remote controls. Don't compete with primary flow.

**UI elements owned:**

**Dock: AI Chat** (bottom right dock, collapsed by default)
- Chat panel with model selector
- **NEW: Context-aware header** — shows current symbol/timeframe/regime/open positions injected automatically
- Send/clear

**Panel: On-Chain** (secondary panel, not primary tab)
- My wallets, watch wallets, token monitor
- Activity feed, whale alerts
- Token resolver

**Drawer: Ops** (collapsed, accessed via ⚙️ button in header)
- Self-healer status + manual trigger
- Agent logs viewer
- Health check dashboard

**Backend domains consumed:**
- `chat` (AI chat)
- `onchain` (smart money proxy)
- `ops` (telegram, logs, healer, health)

**State owned:**
- `state/ui.js` — dock/panel/drawer open flags, active layer

**Polling:** 15s on-chain feed (existing)

---

## 3. Current Element → New Layer Migration Table

Every existing UI element mapped to its new home.

| Current Element | Current Location | Current Handler | New Layer | New Module | Change |
|---|---|---|---|---|---|
| Symbol selector | toolbar | `selectTicker()` @ 1653 | Workbench | `workbench/ticker.js` | Move only |
| Timeframe buttons | toolbar | inline @ 1718 | Workbench | `workbench/timeframe.js` | Move only |
| Chart container | main area | `initChart()` @ 109 | Workbench | `workbench/chart.js` | Move only |
| `tab-recognizing` | top tab | `setToolMode('recognize')` | Workbench | `workbench/analysis.js` | **Merged** into Analysis Workspace mode picker |
| `tab-draw` | top tab | `setToolMode('draw')` | Workbench | `workbench/analysis.js` | **Merged** into Analysis Workspace mode picker |
| `tab-assist` | top tab | `setToolMode('assist')` | Workbench | `workbench/analysis.js` | **Merged** into Analysis Workspace mode picker |
| `tab-agent` | top tab | `toggleAgentPanel()` | Execution | `execution/panel.js` | **Renamed** to "Execution" tab, opens Execution Center |
| `tab-onchain` | top tab | `toggleOnchainPanel()` | Control | `control/onchain.js` | **Demoted** from primary tab to secondary (small icon) |
| `tab-chat` | top tab | `toggleChatPanel()` | Control | `control/chat.js` | **Demoted** to bottom-right dock |
| Drawing trend line btn | left sidebar | inline @ 1836 | Workbench | `workbench/drawing.js` | Move only |
| Drawing horizontal btn | left sidebar | inline @ 1842 | Workbench | `workbench/drawing.js` | Move only |
| Undo / Clear | left sidebar | inline @ 1848 | Workbench | `workbench/drawing.js` | Move only |
| S/R toggle | toolbar | inline @ 1744 | Workbench | `workbench/patterns.js` | Move only |
| Max lines select | toolbar | inline @ 1750 | Workbench | `workbench/patterns.js` | Move only |
| MA toggle btn | toolbar | `toggleMAOverlays()` @ 95 | Workbench | `workbench/ma_overlay.js` | Move only |
| MA legend | chart overlay | `drawMAOverlays()` @ 53 | Workbench | `workbench/ma_overlay.js` | Move only |
| OHLC legend | chart overlay | `updateOHLCLegend()` @ 607 | Workbench | `workbench/legends.js` | Move only |
| Trend indicator | chart overlay | `updateTrendIndicator()` @ 1309 | Workbench | `workbench/legends.js` | Move only |
| Scale toggle | bottom-right | `setScaleMode()` @ 523 | Workbench | `workbench/chart.js` | Move only |
| Magnet toggle | bottom-right | inline @ 1810 | Workbench | `workbench/chart.js` | Move only |
| Pattern stats content | left sidebar | `updateRecognizePanelContent()` @ 900 | Workbench | `workbench/analysis.js` | Move (renders inside unified analysis panel) |
| Pattern stats selected | left sidebar | `fetchPatternStats()` @ 911 | Workbench | `workbench/analysis.js` | Move |
| Similar lines list | left sidebar | `renderSimilarLinesList()` @ 451 | Research | `research/pattern_stats.js` | **Moved** to Research Drawer → Similar sub-tab |
| Replay panel | toolbar | inline @ 1771 | Research | `research/drawer.js` (Replay sub-tab) | **Moved** from toolbar to Research Drawer |
| Data refresh btn | toolbar | inline @ 2631 | Workbench | `workbench/chart.js` | Move only |
| Strategy preset select (chart) | toolbar | inline @ 2648 | Research | `research/backtest.js` | **Moved** to Research Drawer (pre-backtest setup) |
| Backtest button | toolbar | `runBacktest()` @ 1349 | Research | `research/drawer.js` | **Moved** — opens Research Drawer on Backtest sub-tab |
| Backtest panel | slide-out | `runBacktest()` @ 1349 | Research | `research/backtest.js` | Redesigned as sub-tab inside Research Drawer |
| Agent panel header | agent panel | `toggleAgentPanel()` @ 2045 | Execution | `execution/panel.js` | Renamed to "Execution Center" |
| Start / Stop / Revive btns | agent panel | inline @ 2414+ | Execution | `execution/controls.js` | Move to Execution sub-tab |
| Mode banner (Paper/Live) | agent panel | inline @ 2437+ | Execution | `execution/controls.js` | Move to Execution sub-tab |
| Equity hero | agent panel | `refreshAgentStatus()` @ 2128 | Execution | `execution/overview.js` | Move to Overview sub-tab |
| PnL / Daily PnL / Win Rate | agent panel | `refreshAgentStatus()` @ 2171+ | Execution | `execution/overview.js` | Move to Overview sub-tab |
| Stats grid (status/cash/trades/regime/gen/phase) | agent panel | `refreshAgentStatus()` @ 2106+ | Execution | `execution/overview.js` | Move to Overview sub-tab |
| Positions table | agent panel | `refreshAgentStatus()` @ 2219+ | Execution | `execution/overview.js` | Move to Overview sub-tab |
| Recent trades table | agent panel | `refreshAgentStatus()` @ 2230+ | Execution | `execution/overview.js` | Move to Overview sub-tab |
| Signals table | agent panel | `refreshAgentStatus()` @ 2242+ | Execution | `execution/execution_tab.js` | Move to Execution sub-tab |
| Scan Now button | agent panel | inline @ 2458 | Execution | `execution/execution_tab.js` | Move to Execution sub-tab |
| Strategy config form | agent panel | inline @ 2526+ | Execution | `execution/strategy_form.js` | Move to Ops sub-tab |
| Risk limits form | agent panel | inline @ 2569+ | Execution | `execution/risk_tab.js` | Move to Risk sub-tab + add live meters |
| Strategy params grid | agent panel | inline | Execution | `execution/strategy_form.js` | Move to Ops sub-tab |
| Strategy presets | agent panel | `loadPresetList()` @ 2665 | Execution | `execution/strategy_form.js` | Move to Ops sub-tab |
| Telegram config | agent panel (collapsible) | inline @ 2903 | Execution | `execution/ops_tab.js` | Move to Ops sub-tab |
| OKX API keys | agent panel (collapsible) | `refreshOKXStatus()` @ 2598 | Execution | `execution/ops_tab.js` | Move to Ops sub-tab |
| Agent logs | agent panel (collapsible) | `refreshAgentStatus()` @ 2356 | Control | `control/ops_drawer.js` | **Moved** to Ops Drawer |
| Self-healer | agent panel (collapsible) | `refreshAgentStatus()` @ 2385 | Control | `control/ops_drawer.js` | **Moved** to Ops Drawer |
| Strategy description | agent panel | `updateStrategyDescription()` @ 2866 | Execution | `execution/strategy_form.js` | Move (still auto-generated) |
| Chat panel | side panel | `toggleChatPanel()` @ 1879 | Control | `control/chat.js` | **Demoted** to bottom-right dock |
| Chat model selector | chat panel | inline | Control | `control/chat.js` | Move to chat dock |
| On-Chain panel (full) | side panel | `toggleOnchainPanel()` @ 2027 | Control | `control/onchain.js` | Move to secondary panel (not primary tab) |
| My wallets | onchain panel | `renderWalletList()` @ 2955 | Control | `control/onchain.js` | Move |
| Watch wallets | onchain panel | `renderWalletList()` @ 2955 | Control | `control/onchain.js` | Move |
| Token monitor | onchain panel | `_renderTokenList()` @ 2980 | Control | `control/onchain.js` | Move |
| Activity feed | onchain panel | `fetchOnchainSignals()` @ 3297 | Control | `control/onchain.js` | Move |
| Whale alerts | onchain panel | `fetchSmartMoneyWallets()` @ 3326 | Control | `control/onchain.js` | Move |

---

## 4. State Model Split

### `state/market.js`
```javascript
// State (mutable object)
export const marketState = {
  currentSymbol: 'HYPEUSDT',
  currentInterval: '4h',
  allSymbols: [],
  lastCandles: [],
  pricePrecision: null,
  currentScale: 'linear',  // 'linear' | 'log'
  magnetMode: 'weak',      // 'weak' | 'strong'
  replayEndTime: null,
  liveUpdateInterval: null,
};

// Setters (pub/sub aware — emit events)
import { publish } from '../util/events.js';

export function setSymbol(sym) {
  if (marketState.currentSymbol === sym) return;
  marketState.currentSymbol = sym;
  publish('market.symbol.changed', sym);
}

export function setInterval(iv) {
  if (marketState.currentInterval === iv) return;
  marketState.currentInterval = iv;
  publish('market.interval.changed', iv);
}

// ...etc for scale, magnet, replay
```

### `state/patterns.js`
```javascript
export const patternsState = {
  toolMode: 'recognize',   // 'recognize' | 'draw' | 'assist'
  drawingMode: null,        // 'trend' | 'horizontal' | null
  pendingDrawPoint: null,
  userDrawnLines: [],
  rawPatternData: null,
  patternStatsData: null,
  selectedLineIndex: null,
  similarLineIndices: [],
  srLineSegments: [],
  srVisible: true,
  maxSRLines: 0,
  patternResponseCache: new Map(),
};
```

### `state/agent.js`
```javascript
export const agentState = {
  panelOpen: false,
  activeSubTab: 'overview',  // 'overview' | 'execution' | 'risk' | 'ops'
  pollTimer: null,
  pollInFlight: false,
  lastStatus: null,
  lastOKXBalanceFetch: 0,
};
```

### `state/ui.js`
```javascript
export const uiState = {
  activeLayer: 'workbench',    // 'workbench' (default)
  chatDockOpen: false,
  onchainPanelOpen: false,
  researchDrawerOpen: false,
  researchActiveSubTab: 'backtest',
  opsDrawerOpen: false,
};
```

---

## 5. Migration Strategy — Non-Breaking Incremental Rollout

**Core principle:** Old `app.js` keeps running. New modules are added alongside. Features migrate one at a time. Final task swaps the HTML to use the new modules.

### Migration phases (inside Phase 2)

**Stage A — Scaffold (Tasks 1-3)**
1. Create directory structure `js/`, `js/state/`, `js/services/`, `js/workbench/`, `js/execution/`, `js/research/`, `js/control/`, `js/util/`
2. Create empty module files with export stubs
3. Add `/js/*` static file serving in `server/routers/ops.py`
4. Verify old `app.js` still works (no regression)

**Stage B — Utilities + Services (Tasks 4-6)**
4. Implement `util/dom.js`, `util/format.js`, `util/fetch.js`, `util/events.js`, `util/storage.js`
5. Implement all 9 service modules (`services/*.js`) — thin API wrappers
6. Write smoke tests: `services/test.html` that imports each service and calls a basic endpoint

**Stage C — State Modules (Task 7)**
7. Implement `state/*.js` with object-based state + pub/sub setters

**Stage D — Workbench Modules (Tasks 8-14)**
8. `workbench/chart.js` — port `initChart`, `loadData`, scale/magnet
9. `workbench/ticker.js` — port ticker dropdown
10. `workbench/timeframe.js` — port timeframe handlers
11. `workbench/ma_overlay.js` — port MA drawing
12. `workbench/patterns.js` — port S/R rendering
13. `workbench/drawing.js` — port drawing tools
14. `workbench/legends.js` + `workbench/analysis.js` — unified analysis workspace

**Stage E — Execution Modules (Tasks 15-18)**
15. `execution/panel.js` — panel shell + sub-tab switcher
16. `execution/overview.js` — equity/positions/trades sub-tab
17. `execution/execution_tab.js` + `execution/risk_tab.js`
18. `execution/ops_tab.js` + `execution/strategy_form.js` + `execution/controls.js`

**Stage F — Research + Control (Tasks 19-22)**
19. `research/drawer.js` + `research/backtest.js` + `research/pattern_stats.js` + `research/ma_ribbon.js`
20. `control/chat.js` — dock version with context-aware header
21. `control/onchain.js` — onchain panel (largely a move)
22. `control/ops_drawer.js` — healer + logs unified

**Stage G — HTML Split + Final Switchover (Tasks 23-26)**
23. Create new `index.html` with shell regions: `<header>`, `<main class="workbench">`, `<aside class="execution-center hidden">`, `<div class="research-drawer hidden">`, `<div class="chat-dock hidden">`, etc.
24. Create `js/main.js` — module entry, wires everything up
25. Rename old `app.js` to `app.legacy.js`, swap HTML to load `main.js`
26. Smoke test full flow: chart loads, symbol switch works, agent panel opens, backtest runs

**Stage H — Cleanup (Task 27)**
27. Delete `app.legacy.js` if all checks pass. Commit final restructure.

---

## 6. Tasks

### Task 1: Create directory scaffold + verify no regression

**Files:**
- Create: `frontend/js/` directory tree (empty files with `// placeholder` stubs)
- Modify: `server/routers/ops.py` (add `/js/{subpath}` static route)

- [ ] **Step 1:** Create empty directory tree

```bash
cd frontend
mkdir -p js/state js/services js/workbench js/execution js/research js/control js/util
```

- [ ] **Step 2:** Create placeholder files

For each of these 36 paths, create a file with just a top-of-file comment:

```
js/main.js
js/state/{market,agent,patterns,ui,index}.js
js/services/{market,patterns,research,agent,risk,execution,ops,chat,onchain}.js
js/workbench/{chart,ticker,timeframe,analysis,drawing,patterns,ma_overlay,replay,legends,decision_rail}.js
js/execution/{panel,overview,execution_tab,risk_tab,ops_tab,strategy_form,controls}.js
js/research/{drawer,backtest,pattern_stats,ma_ribbon}.js
js/control/{chat,onchain,ops_drawer}.js
js/util/{dom,format,fetch,events,storage}.js
```

Each file gets: `// <module path> — Phase 2 scaffold\nexport {};\n`

- [ ] **Step 3:** Add `/js/*` static route in `server/routers/ops.py`

```python
@router.get("/js/{subpath:path}")
async def serve_js_module(subpath: str):
    """Serve ES modules from frontend/js/."""
    full_path = FRONTEND_DIR / "js" / subpath
    if not full_path.is_file():
        return FileResponse(str(FRONTEND_DIR / "404.html"), status_code=404) if (FRONTEND_DIR / "404.html").exists() else {"error": "not found"}
    return FileResponse(
        str(full_path),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )
```

- [ ] **Step 4:** Restart server, smoke test

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8003/js/main.js
# Expected: HTTP 200 (empty file)
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8003/js/state/market.js
# Expected: HTTP 200
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8003/
# Expected: HTTP 200 (index.html still works — old app.js unchanged)
```

- [ ] **Step 5:** Commit

```bash
git add frontend/js/ server/routers/ops.py
git commit -m "feat(phase2): scaffold frontend/js/ directory tree + serve modules"
```

---

### Task 2: Implement util/ modules

**Files:**
- Modify: `frontend/js/util/dom.js`, `format.js`, `fetch.js`, `events.js`, `storage.js`

- [ ] **Step 1:** Implement `util/dom.js`

```javascript
// frontend/js/util/dom.js
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function on(el, event, handler, options) {
  if (typeof el === 'string') el = $(el);
  if (!el) return;
  el.addEventListener(event, handler, options);
  return () => el.removeEventListener(event, handler, options);
}

export function show(el) { (typeof el === 'string' ? $(el) : el)?.classList.remove('hidden'); }
export function hide(el) { (typeof el === 'string' ? $(el) : el)?.classList.add('hidden'); }
export function toggleClass(el, cls, force) {
  (typeof el === 'string' ? $(el) : el)?.classList.toggle(cls, force);
}

export function setText(el, text) {
  const node = typeof el === 'string' ? $(el) : el;
  if (node) node.textContent = text;
}

export function setHtml(el, html) {
  const node = typeof el === 'string' ? $(el) : el;
  if (node) node.innerHTML = html;
}
```

- [ ] **Step 2:** Implement `util/format.js`

```javascript
// frontend/js/util/format.js
export function inferPrecision(price) {
  if (price == null || !isFinite(price)) return 2;
  if (price >= 1000) return 2;
  if (price >= 1) return 4;
  if (price >= 0.01) return 6;
  return 8;
}

export function formatPrice(price, precision) {
  if (price == null || !isFinite(price)) return '—';
  const p = precision ?? inferPrecision(price);
  return Number(price).toFixed(p);
}

export function formatPct(pct, digits = 2) {
  if (pct == null || !isFinite(pct)) return '—';
  const sign = pct > 0 ? '+' : '';
  return `${sign}${Number(pct).toFixed(digits)}%`;
}

export function formatUsd(value, digits = 2) {
  if (value == null || !isFinite(value)) return '—';
  const sign = value < 0 ? '-' : '';
  return `${sign}$${Math.abs(Number(value)).toFixed(digits)}`;
}

export function pnlColorClass(pnl) {
  if (pnl == null || pnl === 0) return '';
  return pnl > 0 ? 'pnl-pos' : 'pnl-neg';
}
```

- [ ] **Step 3:** Implement `util/fetch.js`

```javascript
// frontend/js/util/fetch.js
const DEFAULT_TIMEOUT = 15000;

export class FetchError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export async function fetchJson(url, { method = 'GET', body, timeout = DEFAULT_TIMEOUT, signal } = {}) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeout);
  const combined = signal ? combineSignals([signal, ac.signal]) : ac.signal;

  try {
    const res = await fetch(url, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: combined,
    });
    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    if (!res.ok) throw new FetchError(`HTTP ${res.status}`, res.status, data);
    return data;
  } finally {
    clearTimeout(timer);
  }
}

function combineSignals(signals) {
  const ac = new AbortController();
  signals.forEach(s => {
    if (s.aborted) ac.abort();
    else s.addEventListener('abort', () => ac.abort(), { once: true });
  });
  return ac.signal;
}
```

- [ ] **Step 4:** Implement `util/events.js`

```javascript
// frontend/js/util/events.js
// Lightweight UI-local pub/sub. NOT the Phase 3 backend event bus.

const handlers = new Map();

export function subscribe(event, fn) {
  if (!handlers.has(event)) handlers.set(event, new Set());
  handlers.get(event).add(fn);
  return () => handlers.get(event)?.delete(fn);
}

export function publish(event, data) {
  const set = handlers.get(event);
  if (!set) return;
  for (const fn of set) {
    try { fn(data); }
    catch (err) { console.error(`[events] ${event} handler error:`, err); }
  }
}

export function unsubscribeAll(event) {
  handlers.delete(event);
}
```

- [ ] **Step 5:** Implement `util/storage.js`

```javascript
// frontend/js/util/storage.js
const PREFIX = 'cryptoTA:';

export function loadJSON(key, fallback = null) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

export function saveJSON(key, value) {
  try { localStorage.setItem(PREFIX + key, JSON.stringify(value)); return true; }
  catch { return false; }
}

export function remove(key) { localStorage.removeItem(PREFIX + key); }
```

- [ ] **Step 6:** Commit

```bash
git add frontend/js/util/
git commit -m "feat(phase2): implement util/ modules (dom, format, fetch, events, storage)"
```

---

### Task 3: Implement service modules (API clients)

**Files:**
- Modify: `frontend/js/services/*.js` (all 9 files)

- [ ] **Step 1:** Implement `services/market.js`

```javascript
// frontend/js/services/market.js
import { fetchJson } from '../util/fetch.js';

export const getSymbols = (includeExtended = false) =>
  fetchJson(`/api/symbols${includeExtended ? '?include_extended=true' : ''}`);

export const getSymbolInfo = (symbol) =>
  fetchJson(`/api/symbol-info?symbol=${encodeURIComponent(symbol)}`);

export const getOhlcv = (symbol, interval, days = 30, endTime = null) => {
  const params = new URLSearchParams({ symbol, interval, days: String(days) });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/ohlcv?${params}`);
};

export const getChart = (symbol, interval, days = 365, endTime = null) => {
  const params = new URLSearchParams({ symbol, interval, days: String(days) });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/chart?${params}`);
};

export const getTopVolume = (n = 20) => fetchJson(`/api/top-volume?n=${n}`);

export const getDataInfo = (symbol, interval) =>
  fetchJson(`/api/data-info?symbol=${encodeURIComponent(symbol)}&interval=${interval}`);
```

- [ ] **Step 2:** Implement `services/patterns.js`

```javascript
// frontend/js/services/patterns.js
import { fetchJson } from '../util/fetch.js';

export const getPatterns = (symbol, interval, days = 30, mode = 'full', endTime = null) => {
  const params = new URLSearchParams({ symbol, interval, days: String(days), mode });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/patterns?${params}`);
};

export const getPatternStatsBacktest = (symbol, interval, days = 365) =>
  fetchJson(`/api/pattern-stats/backtest?symbol=${symbol}&interval=${interval}&days=${days}`);

export const getPatternFeatures = (symbol, interval, endTime = null) => {
  const params = new URLSearchParams({ symbol, interval });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/pattern-stats/features?${params}`);
};

export const getCurrentVsHistory = (symbol, interval, days = 365, epsilon = 0.35) =>
  fetchJson(`/api/pattern-stats/current-vs-history?symbol=${symbol}&interval=${interval}&days=${days}&epsilon=${epsilon}`);

export const getLineSimilar = (symbol, interval, days, x1, y1, x2, y2, epsilon = 0.4, maxLines = 15) => {
  const params = new URLSearchParams({
    symbol, interval, days: String(days),
    x1: String(x1), y1: String(y1), x2: String(x2), y2: String(y2),
    epsilon: String(epsilon), max_lines: String(maxLines),
  });
  return fetchJson(`/api/pattern-stats/line-similar?${params}`);
};
```

- [ ] **Step 3:** Implement `services/research.js`

```javascript
// frontend/js/services/research.js
import { fetchJson } from '../util/fetch.js';

export const runBacktest = (symbol, interval, days = 365, overrides = {}) => {
  const params = new URLSearchParams({ symbol, interval, days: String(days) });
  Object.entries(overrides).forEach(([k, v]) => {
    if (v != null && v !== '') params.set(k, String(v));
  });
  return fetchJson(`/api/backtest?${params}`);
};

export const optimizeBacktest = (symbol, interval, days = 365, objective = 'total_pnl', maxiter = 80, method = 'L-BFGS-B') =>
  fetchJson(`/api/backtest/optimize?symbol=${symbol}&interval=${interval}&days=${days}&objective=${objective}&maxiter=${maxiter}&method=${method}`, { method: 'POST' });

export const getMaRibbon = (symbol) => fetchJson(`/api/ma-ribbon?symbol=${symbol}`);

export const getMaRibbonBacktest = (symbol, opts = {}) => {
  const params = new URLSearchParams({ symbol });
  Object.entries(opts).forEach(([k, v]) => params.set(k, String(v)));
  return fetchJson(`/api/ma-ribbon/backtest?${params}`);
};
```

- [ ] **Step 4:** Implement `services/agent.js`

```javascript
// frontend/js/services/agent.js
import { fetchJson } from '../util/fetch.js';

export const getStatus = () => fetchJson('/api/agent/status');
export const start = () => fetchJson('/api/agent/start', { method: 'POST' });
export const stop = () => fetchJson('/api/agent/stop', { method: 'POST' });
export const revive = () => fetchJson('/api/agent/revive', { method: 'POST' });
export const setConfig = (body) => fetchJson('/api/agent/config', { method: 'POST', body });

export const getSignals = () => fetchJson('/api/agent/signals');
export const getAuditLog = (limit = 50) => fetchJson(`/api/agent/audit-log?limit=${limit}`);
export const getLessons = () => fetchJson('/api/agent/lessons');

export const setStrategyConfig = (body) => fetchJson('/api/agent/strategy-config', { method: 'POST', body });
export const setStrategyParams = (params) => fetchJson('/api/agent/strategy-params', { method: 'POST', body: params });

export const getPresets = () => fetchJson('/api/agent/strategy-presets');
export const savePreset = (name) => fetchJson('/api/agent/strategy-presets/save', { method: 'POST', body: { name } });
export const loadPreset = (name) => fetchJson('/api/agent/strategy-presets/load', { method: 'POST', body: { name } });
export const deletePreset = (name) => fetchJson('/api/agent/strategy-presets/delete', { method: 'POST', body: { name } });
```

- [ ] **Step 5:** Implement remaining services

`services/risk.js`:
```javascript
import { fetchJson } from '../util/fetch.js';
export const setRiskLimits = (body) => fetchJson('/api/agent/risk-limits', { method: 'POST', body });
```

`services/execution.js`:
```javascript
import { fetchJson } from '../util/fetch.js';
export const setOkxKeys = (api_key, secret, passphrase) =>
  fetchJson('/api/agent/okx-keys', { method: 'POST', body: { api_key, secret, passphrase } });
export const getOkxStatus = () => fetchJson('/api/agent/okx-status');
```

`services/ops.js`:
```javascript
import { fetchJson } from '../util/fetch.js';
export const getHealth = () => fetchJson('/api/health');
export const setTelegramConfig = (body) => fetchJson('/api/agent/telegram-config', { method: 'POST', body });
export const getLogs = (limit = 50, filter = 'agent') => fetchJson(`/api/agent/logs?limit=${limit}&filter=${filter}`);
export const getHealerStatus = () => fetchJson('/api/healer/status');
export const triggerHealer = () => fetchJson('/api/healer/trigger', { method: 'POST' });
export const stopHealer = () => fetchJson('/api/healer/stop', { method: 'POST' });
export const startHealer = () => fetchJson('/api/healer/start', { method: 'POST' });
```

`services/chat.js`:
```javascript
import { fetchJson } from '../util/fetch.js';
export const sendChat = (message, sessionId = 'default', model = null) =>
  fetchJson('/api/chat', { method: 'POST', body: { message, session_id: sessionId, model } });
export const getModels = () => fetchJson('/api/chat/models');
export const getHistory = (sessionId = 'default') => fetchJson(`/api/chat/history?session_id=${sessionId}`);
export const clearHistory = (sessionId = 'default') => fetchJson(`/api/chat/clear?session_id=${sessionId}`, { method: 'POST' });
```

`services/onchain.js`:
```javascript
import { fetchJson } from '../util/fetch.js';
export const getHealth = () => fetchJson('/api/onchain/health');
export const getWallets = () => fetchJson('/api/onchain/wallets');
export const getSmartMoney = () => fetchJson('/api/onchain/wallets/smart-money');
export const getWallet = (address) => fetchJson(`/api/onchain/wallets/${address}`);
export const trackWallet = (address) => fetchJson(`/api/onchain/wallets/track/${address}`, { method: 'POST' });
export const untrackWallet = (address) => fetchJson(`/api/onchain/wallets/track/${address}`, { method: 'DELETE' });
export const getSignals = (limit = 50) => fetchJson(`/api/onchain/signals?limit=${limit}`);
export const getRecommendations = (limit = 20) => fetchJson(`/api/onchain/signals/recommendations?limit=${limit}`);
export const tokenAnalyze = (address, network = 'solana', pool = null) => {
  const params = new URLSearchParams({ network });
  if (pool) params.set('pool', pool);
  return fetchJson(`/api/onchain/token/analyze/${address}?${params}`);
};
```

- [ ] **Step 6:** Commit

```bash
git add frontend/js/services/
git commit -m "feat(phase2): implement all 9 service modules"
```

---

### Tasks 4-27

The detailed steps for Tasks 4-27 follow the same pattern:
- State modules: copy existing globals from app.js, wrap in exported objects
- Workbench modules: copy existing functions from app.js:53-1500 into their respective module files, replace direct DOM manipulation with imports from util/dom.js, replace fetch calls with imports from services/*.js
- Execution modules: same pattern for app.js:2024-2725
- Research modules: same pattern for app.js:1349-1500 (backtest) and relevant pattern stats code
- Control modules: same pattern for app.js:1856-2022 (chat), 2780-3349 (onchain)

**Key discipline for every Stage D-F task:**
1. Open old `app.js` to the exact line ranges
2. Copy functions verbatim
3. Replace inline DOM lookups (`document.getElementById(...)`) with `$('#...')` from `util/dom.js`
4. Replace `fetch('/api/...')` with imports from `services/*.js`
5. Replace global variable reads/writes with imports from `state/*.js`
6. Export the main entry function (e.g., `initChart`, `renderTickerList`)
7. In `main.js`, import and call the entry function during boot

**Testing each stage:**
- Stages B-F: Add a temporary `<script type="module" src="/js/<test>.js"></script>` that calls one function and logs success
- Verify via browser console that the module runs

**Switchover (Stage G / Tasks 23-26):**
- New `index.html` keeps all existing element IDs (so new modules can find them)
- Old `app.js` stays loaded until `main.js` is ready
- Final swap: remove `<script src="/app.js">`, add `<script type="module" src="/js/main.js">`
- Rename `app.js` → `app.legacy.js`
- Smoke test every existing feature (20-point checklist below)

---

## 7. Final Smoke Test Checklist (Task 26)

After Stage G completes, verify every feature still works:

- [ ] Chart loads on boot (BTCUSDT 4h default)
- [ ] Symbol dropdown opens, search works, selecting a symbol reloads chart
- [ ] Timeframe buttons switch (5m, 15m, 1h, 4h, 1d)
- [ ] S/R lines toggle on/off
- [ ] Max S/R lines select changes count
- [ ] MA overlay toggle shows/hides MAs
- [ ] Analysis workspace: switch Recognize / Draw / Assist modes
- [ ] Draw trendline: click two points, line appears
- [ ] Draw horizontal line: click one point, line appears
- [ ] Undo / Clear drawings
- [ ] Replay mode: set date, chart re-renders at historical point
- [ ] Scale Linear/Log toggle
- [ ] Magnet Weak/Strong toggle
- [ ] OHLC legend follows crosshair
- [ ] Trend indicator shows current direction
- [ ] Execution Center opens
- [ ] All 4 sub-tabs (Overview/Execution/Risk/Ops) switch
- [ ] Agent start/stop/revive
- [ ] Paper/Live mode switch
- [ ] Risk limits save
- [ ] Strategy config save
- [ ] OKX keys save + verify
- [ ] Telegram config save + test message
- [ ] Presets load/save/delete
- [ ] Research Drawer opens with all 4 sub-tabs
- [ ] Backtest runs
- [ ] MA ribbon view loads
- [ ] Chat dock opens, send message, get response
- [ ] On-chain panel opens, wallet tracking works
- [ ] Healer status + trigger button

---

## 8. Rollback Plan

If Phase 2 introduces regression:
1. Git revert to commit before Stage G
2. Old `app.js` is still intact through all of Stages A-F
3. Worst case: revert to `e6177f0` (Phase 1 baseline)

Phase 2 is designed so that until the Stage G switchover, the old app keeps running untouched.
