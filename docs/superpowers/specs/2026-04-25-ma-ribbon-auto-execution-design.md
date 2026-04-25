# MA Ribbon EMA21 Auto-Execution — Live Integration Spec

**Date**: 2026-04-25
**Author**: Claude (under user direction)
**Status**: brainstorming → draft, awaiting user review before writing-plans
**Scope**: ship the validated MA-ribbon EMA21 strategy as a **live auto-execution** mode in the main trading system, alongside the existing manual-line-driven flow. **Real-money production code.**

---

## 0. Executive summary

Live integration of the MA-ribbon strategy. The user enables a strategy entry in the v2 trading panel, configures parameters, and the system auto-scans Bitget every 60 s. When a multi-TF bullish (or bearish) MA-ribbon formation event fires, the system spawns up to 4 layered `ConditionalOrder`s (Strategy Y time-progressive: LV1 immediate, LV2 at next 15 m close, LV3 at next 1 h close, LV4 at next 4 h close) with EMA21-buffer trailing stops. Trades flow through the existing `server/conditionals/watcher.py` pipeline.

**Live-only.** Paper / dry-run validation happens **on the separate backtest panel** (`port 8765`, already shipped). The main system contains no paper mode for MA-ribbon — we either trade real money or we don't trade.

**Real-money safety: 6 hard gates** that activate independently. Any gate triggering halts new MA-ribbon orders without affecting manual-line trading:

1. Max **25** concurrent open MA-ribbon positions.
2. Per-layer hard size cap: LV1 0.1 % / LV2 0.25 % / LV3 0.5 % / LV4 2 % of equity.
3. Per-symbol cumulative MA-ribbon risk ≤ 2 % equity.
4. Strategy cumulative DD ≤ -15 % of strategy capital → auto-halt.
5. **14-day ramp-up**: day-1 total risk cap 2 % equity, +1 % per day to 15 % by day 14.
6. **Emergency Stop** red button: immediate full-flatten + 24 h lockout.

All gate breaches log to a strategy ledger, surface in UI as red banners, send a Telegram alert, and block the next `_submit_live` call.

---

## 1. Glossary (resolves naming clashes vs existing project)

| Term | This spec's meaning | Vs existing project |
|---|---|---|
| `buffer` (in MA-ribbon) | EMA21 → SL distance percent (e.g. `0.01` = 1 %). | **Different from** TA_BASICS.md §6 manual-line `buffer` (= line → entry distance). In code I call the new one `ribbon_buffer_pct` to avoid ambiguity. |
| Layer (LV1–LV4) | Position chunk anchored to one TF: LV1 = 5 m, LV2 = 15 m, LV3 = 1 h, LV4 = 4 h. | New concept; no analogue in manual-line system. |
| Lineage | `ConditionalOrder.lineage = "ma_ribbon"` tags every order this strategy spawns. | Existing manual-line lineage is `"manual_line"`. |
| Signal ID | One UUID per ribbon formation event. All 4 layers share it. | New. Not used in manual-line system. |
| Dry-run / paper | **Lives on backtest panel only.** Main system has no paper for MA-ribbon. | The watcher's existing `_submit_paper()` path is **NOT** used for MA-ribbon. |
| Strategy cap | Account-wide configurable USD or % of equity that this strategy is allowed to put at risk. | New. |
| DD halt | Strategy cumulative realized + unrealized PnL ≤ -15 % of strategy cap → block new orders. | New. Independent of any account-wide drawdown — manual-line trades are not counted. |

Per-TF buffer defaults (configurable) — same as Phase 2 panel:

| TF | `ribbon_buffer_pct` |
|---|---|
| 5 m | 1.0 % |
| 15 m | 4.0 % |
| 1 h | 7.0 % |
| 4 h | 10.0 % |

---

## 2. Architecture — Approach C (Hybrid scanner + adapter)

```
┌──────────────────────────────────────────────────────────────────────┐
│  ma_ribbon_auto_scanner.py  (NEW · asyncio task · tick every 60 s)   │
│  ─ load state                                                        │
│  ─ if not enabled OR halted → return                                 │
│  ─ enforce ramp-up day-cap + concurrent-cap + DD-halt                │
│  ─ fetch_universe_async() (re-uses Phase 2 data layer)               │
│  ─ detect new bull / bear formation events                           │
│  ─ for each new event: build 4 layer plans (Strategy Y)              │
│  ─ for layers ready now: hand to adapter                             │
│  ─ persist pending higher-TF layers in state                         │
└──────────────────────────────────────────────────────────────────────┘
                                │  (1 layer = 1 ConditionalOrder spec)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ma_ribbon_auto_adapter.py  (NEW · pure translation, ~120 lines)     │
│  ─ map signal → ConditionalOrder(                                    │
│      lineage="ma_ribbon",                                            │
│      manual_line_id=None,                                            │
│      side=None,            # ribbon has no DB line                   │
│      direction=long|short, # bull→long / bear→short                  │
│      sl_logic="ribbon_ema21_trailing",                               │
│      ribbon_meta={signal_id, layer, tf, buffer_pct,                  │
│                   ema21_at_signal, ramp_day_cap_at_spawn})           │
│  ─ insert into existing conditionals store                           │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  server/conditionals/watcher.py  (EXISTING · modified)               │
│  ─ _tick() polling unchanged                                         │
│  ─ _compute_qty(): for ribbon lineage, ignores manual-line tolerance │
│                    and uses ribbon_buffer_pct for entry-to-SL gap    │
│  ─ _sync_sl_to_line_now(): branches on cond.config.sl_logic;         │
│       "ribbon_ema21_trailing" path computes new sl from current      │
│       EMA21 (one fresh fetch + recompute) and applies                │
│       max(prev_sl, new_sl) for longs / min for shorts                │
│  ─ _submit_live() unchanged (real Bitget call)                       │
│  ─ _submit_paper() **NOT CALLED** for ribbon orders                  │
└──────────────────────────────────────────────────────────────────────┘
```

The watcher already owns: order-id reconciliation, Bitget API calls, qty calculation, fill detection, rate-limit handling, Telegram alerts, error retries. We do not duplicate any of this. The only change is the SL-logic branch.

---

## 3. Detailed component specs

### 3.1 `server/strategy/ma_ribbon_auto_scanner.py`

```python
async def scan_loop():
    while True:
        try:
            await tick()
        except Exception:
            log.exception("scanner tick failed")
        await asyncio.sleep(60)

async def tick():
    state = await load_state()
    if not state.enabled or state.halted: return
    if datetime.utcnow() < state.locked_until_utc: return  # post-emergency lockout

    enforce_ramp_up_day(state)
    if at_concurrent_cap(state): return
    if at_dd_halt(state):
        state.halted = True; state.halt_reason = "dd_-15pct"
        save_state(state); telegram("MA-ribbon halted at -15% DD"); return

    universe = await load_universe(state.universe_filter)
    data = await fetch_universe_async(universe, ALL_TFS, cfg=state.fetch_cfg)
    new_signals = detect_new_signals(data, state.last_processed_bar_ts)
    for sig in new_signals:
        if not size_within_caps(sig, state): continue
        await spawn_layer1(sig, state)
        register_pending_higher_layers(sig, state)
    await spawn_ready_higher_layers(state, data)
    save_state(state)
```

Notes:
- Bull and bear are detected separately. The scanner keeps two parallel pipelines (one per direction). Bear detection is the strict mirror: `close < MA5 < MA8 < EMA21 < MA55` with all comparison flips.
- `last_processed_bar_ts[(symbol, tf)]` deduplicates: a single TF bar only ever fires its layer once.
- `register_pending_higher_layers` writes `pending_lv2_eta`, `pending_lv3_eta`, `pending_lv4_eta` UNIX timestamps. Each layer fires when the next TF bar after the signal closes AND the ribbon is still aligned at that close. If alignment breaks at any TF's check, that layer is permanently skipped for this signal.
- After a signal's LV4 fires (or is skipped), the signal record is closed and removed from pending state.

### 3.2 `server/strategy/ma_ribbon_auto_state.py`

JSON file at `data/state/ma_ribbon_auto_state.json`. Fields:

```json
{
  "enabled": false,
  "halted": false,
  "halt_reason": null,
  "locked_until_utc": null,
  "first_enabled_at_utc": null,
  "current_ramp_day": 1,
  "config": {
    "universe_filter": {"min_volume_usd": 1_000_000, "product_types": ["USDT-FUTURES"]},
    "tfs": ["5m", "15m", "1h", "4h"],
    "directions": ["long", "short"],
    "ribbon_buffer_pct": {"5m": 0.01, "15m": 0.04, "1h": 0.07, "4h": 0.10},
    "layer_risk_pct": {"LV1": 0.001, "LV2": 0.0025, "LV3": 0.005, "LV4": 0.02},
    "max_concurrent_orders": 25,
    "per_symbol_risk_cap_pct": 0.02,
    "dd_halt_pct": 0.15,
    "strategy_capital_usd": 0,
    "fetch_cfg": {"pages_per_symbol": 30, "concurrency": 12}
  },
  "ledger": {
    "trades": [/* {signal_id, layer, opened_ts, closed_ts, pnl_usd, ...} */],
    "open_positions": [/* signal_id, layer, qty, entry, current_sl, ... */],
    "realized_pnl_usd_cumulative": 0.0
  },
  "last_processed_bar_ts": {"BTCUSDT_5m": 1777000000, "...": "..."},
  "pending_signals": [
    {"signal_id": "...", "symbol": "BTCUSDT", "direction": "long",
     "spawned_layers": ["LV1"], "pending_layers": [
        {"layer": "LV2", "tf": "15m", "trigger_at_bar_close_after_ts": 1777005000},
        {"layer": "LV3", "tf": "1h",  "trigger_at_bar_close_after_ts": 1777003600},
        {"layer": "LV4", "tf": "4h",  "trigger_at_bar_close_after_ts": 1777024400}
     ]}
  ],
  "errors_recent": [/* most recent 50 errors w/ ts + traceback */]
}
```

Atomic writes via temp file + rename (no torn writes).

### 3.3 `server/strategy/ma_ribbon_auto_adapter.py`

Pure translation, no I/O. Test via fixtures.

```python
def signal_to_conditional(
    sig: Phase1Signal,
    layer: str,
    state: AutoState,
) -> ConditionalOrder:
    cfg = state.config
    buffer_pct = cfg.ribbon_buffer_pct[sig.tf]
    risk_pct = cfg.layer_risk_pct[layer]
    risk_usd = state.account_equity_usd() * risk_pct

    if sig.direction == "long":
        sl_at_signal = sig.ema21_at_signal * (1 - buffer_pct)
    else:
        sl_at_signal = sig.ema21_at_signal * (1 + buffer_pct)
    entry_to_sl_pct = abs(sig.next_bar_open_estimate - sl_at_signal) / sig.next_bar_open_estimate
    qty_notional_usd = risk_usd / entry_to_sl_pct

    return ConditionalOrder(
        lineage="ma_ribbon",
        manual_line_id=None,
        symbol=sig.symbol,
        timeframe=sig.tf,
        direction=sig.direction,
        config=OrderConfig(
            sl_logic="ribbon_ema21_trailing",
            ribbon_meta={
                "signal_id": sig.signal_id,
                "layer": layer,
                "tf": sig.tf,
                "ribbon_buffer_pct": buffer_pct,
                "ema21_at_signal": sig.ema21_at_signal,
                "ramp_day_cap_pct_at_spawn": current_ramp_cap_pct(state),
                "reverse_on_stop": False,  # explicit OPT-OUT of P7 auto-reverse
            },
            risk_usd_target=risk_usd,
            qty_notional_target=qty_notional_usd,
            entry_offset_points=None,
            stop_points=None,
        ),
    )
```

### 3.4 `server/routers/ma_ribbon_auto.py`

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/ma_ribbon_auto/status` | GET | full state JSON for UI |
| `/api/ma_ribbon_auto/enable` | POST | flip enabled = True; require `confirm_acknowledged_p2_gate=true` body |
| `/api/ma_ribbon_auto/disable` | POST | flip enabled = False; pending higher layers expire |
| `/api/ma_ribbon_auto/config` | POST | update config block; rejected if invalid (e.g. layer_risk_pct exceeds 5 %) |
| `/api/ma_ribbon_auto/emergency_stop` | POST | flatten ALL ribbon positions immediately, set 24 h lock |

`/enable` rejects unless body has `{"confirm_acknowledged_p2_gate": true, "confirm_first_day_cap_2pct": true}`. Front-end forces a 2-checkbox modal before sending.

### 3.5 `frontend/js/workbench/strategy_card_ma_ribbon.js`

A new card on v2 page (or new `策略` tab — choice tracked as open question § 13.1). Renders:

- Status badge: `ENABLED` / `DISABLED` / `HALTED (reason)` / `LOCKED (until)`.
- Ramp-up progress bar: `Day 4 of 14 — current cap 5 %`.
- Open positions table (read from ledger).
- Realized + unrealized PnL.
- Recent triggered events (last 20).
- Config form (collapsible) — submit goes to `/api/ma_ribbon_auto/config`.
- 2 buttons: `Enable` (with 2-checkbox confirm) and `Emergency Stop` (red, with type-to-confirm `STOP`).
- Auto-poll status every 5 s.

### 3.6 `server/strategy/catalog.py` extension

Add one `StrategyTemplate` entry. Same shape as the 4 existing templates. Dropping into the catalog also lets it appear in any future "select strategy" picker.

```python
StrategyTemplate(
    template_id="ma_ribbon_ema21_auto",
    name="MA Ribbon EMA21 自动",
    name_en="MA Ribbon EMA21 Auto",
    description="多 TF MA-ribbon 自动扫单。5m/15m/1h/4h 形成多头排列时分层加仓 (Strategy Y 时间渐进)。SL 用当前 EMA21 × (1 - buffer%) 跟随。账户级 -15% DD 自动 halt。详见 spec 2026-04-25-ma-ribbon-auto-execution.",
    category="trend",
    supported_timeframes=("5m", "15m", "1h", "4h"),
    default_trigger_modes=("ribbon_formation",),
    default_params={
        "ribbon_buffer_pct": {"5m": 0.01, "15m": 0.04, "1h": 0.07, "4h": 0.10},
        "layer_risk_pct":    {"LV1": 0.001, "LV2": 0.0025, "LV3": 0.005, "LV4": 0.02},
        "max_concurrent_orders": 25,
        "dd_halt_pct": 0.15,
    },
    risk_level="high",
)
```

### 3.7 `server/conditionals/types.py` extension

Two opt-in additions to `OrderConfig`. Defaults preserve all existing manual-line behaviour.

```python
sl_logic: Literal["line_buffer", "ribbon_ema21_trailing"] = "line_buffer"
ribbon_meta: dict | None = None
```

When `sl_logic == "line_buffer"` (the default everywhere existing code creates it), behaviour is unchanged. Manual-line code paths never set `ribbon_meta`.

### 3.8 `server/conditionals/watcher.py` SL-logic branch

Change is localized to `_sync_sl_to_line_now` and `_compute_qty`. ~30 LOC total.

```python
if cond.config.sl_logic == "ribbon_ema21_trailing":
    meta = cond.config.ribbon_meta
    cur_ema21 = await fetch_current_ema21(cond.symbol, meta["tf"])
    if cond.direction == "long":
        candidate = cur_ema21 * (1 - meta["ribbon_buffer_pct"])
        new_sl = max(cond.current_sl, candidate)
    else:
        candidate = cur_ema21 * (1 + meta["ribbon_buffer_pct"])
        new_sl = min(cond.current_sl, candidate)
    return new_sl
# else: existing line-based path
```

`fetch_current_ema21(symbol, tf)` is a new helper — reuses `data_loader_async.fetch_ohlcv_async` for one (symbol, TF) pair, computes EMA21 on the result. Cached 60 s in process memory.

---

## 4. Data flow

Single bull formation event end-to-end:

```
T+0      5 m bar closes; scanner detects (False→True) at this bar
T+0      adapter builds LV1 ConditionalOrder; inserted into store
T+0      Telegram: "MA-ribbon BTCUSDT LONG LV1 about to submit ($X risk)"
T+0+ε    watcher.tick() picks up new conditional, calls _submit_live
         → real Bitget plan order placed
T+0+ε    state.ledger.open_positions += this layer
T+5m     5 m next bar opens; price fills at open (Bitget side); fill detected
         by reconcile loop; entry_price recorded in ledger
...      SL trails on every closed bar of TF=5m via _sync_sl_to_line_now
T+15m    next 15 m bar closes; scanner verifies still in ribbon
         → spawns LV2 ConditionalOrder (3× LV1 risk in USD per spec)
         → identical lifecycle from here
T+1h     spawns LV3 if 1h still aligned
T+4h     spawns LV4 if 4h still aligned
```

If any layer's bar closes NOT-aligned, that layer is permanently skipped for this signal. Already-open earlier layers continue running until their own SL hits.

When **any** layer's SL hits:
- `watcher` records the fill, computes realized PnL, closes the open position.
- `state.ledger.realized_pnl_usd_cumulative` updated.
- DD-halt check runs. If breached → `halted=True`.
- Other layers of the same signal continue running (they have their own SL).

Emergency stop:
- POST `/api/ma_ribbon_auto/emergency_stop`.
- Iterate `ledger.open_positions[ribbon]`, send Bitget cancel + market-close for each.
- Set `state.locked_until_utc = now + 24h`.
- Cancel all pending-layer plans.

---

## 5. Risk control & safety mechanisms

This is the section that gets reviewed FIRST. Each gate is:
- testable in isolation,
- logged on trigger,
- visible in UI,
- and **independent** — no single bug bypasses all gates.

### 5.1 Per-layer hard size caps

User-confirmed values (subject to user confirming LV3 = 0.5 %, my read of the audio):

```
LV1 risk_pct = 0.001  (0.1 %)
LV2 risk_pct = 0.0025 (0.25 %)
LV3 risk_pct = 0.005  (0.5 %)
LV4 risk_pct = 0.02   (2 %)
```

`signal_to_conditional` reads these. Anything > 5 % anywhere in config rejected by `/api/ma_ribbon_auto/config` validator.

### 5.2 Concurrent order cap

`len(state.ledger.open_positions) >= state.config.max_concurrent_orders` (default 25) → scanner skips spawning new layers (existing pending layers still complete).

### 5.3 Per-symbol cumulative risk cap

`sum(layer.risk_pct for layer in state.ledger.open_positions if layer.symbol == s) >= state.config.per_symbol_risk_cap_pct` (default 2 %) → scanner blocks any further layer for that symbol on either direction.

### 5.4 Strategy DD halt (-15 %)

```
total_pnl_pct = (realized_pnl_cumulative + sum(unrealized for open)) / strategy_capital_usd
if total_pnl_pct <= -dd_halt_pct: halted = True
```

`halted = True` stops new orders. Open positions continue trailing/exiting normally. User must manually un-halt via UI (separate confirmation modal).

### 5.5 14-day ramp-up

```
days_since_first_enabled = (now - first_enabled_at_utc).days
ramp_cap_pct = min(0.15, 0.02 + 0.01 * days_since_first_enabled)
# Day 1 = 0.02, Day 2 = 0.03, ..., Day 14 = 0.15
```

`current_total_risk_pct = sum(open_position.risk_pct)` must be ≤ ramp_cap_pct **before** spawning any new layer. Stored in `ribbon_meta.ramp_day_cap_pct_at_spawn` for the audit trail.

If user disables and re-enables, ramp does NOT reset — `first_enabled_at_utc` only set the very first time. (Otherwise users would just disable+enable to skip ramp.)

### 5.6 Emergency stop

Red button in UI. POST `/api/ma_ribbon_auto/emergency_stop`. Server:
1. Sends Bitget cancel for every open ribbon plan order.
2. Sends Bitget market close for every open ribbon position.
3. Sets `state.locked_until_utc = now + 24h`.
4. Sets `halted = True`, `halt_reason = "emergency_stop"`.
5. Logs to `data/logs/ma_ribbon_emergency_stop.log` with timestamp + position list + reason field from request body.
6. Telegram alert.

User must wait 24 h to re-enable (no override — encodes the intent that emergency stop = "I need to stop and think").

### 5.7 Bitget rate-limit (429) handling

On 429 from any Bitget call:
- Adapter / watcher pauses 30 s.
- Scanner skips this tick.
- `state.errors_recent` gets a structured record.
- Three 429s in a 5-minute window → Telegram alert + scanner pause for 5 min.

No silent retries (per `PRINCIPLES.md` §P10).

---

## 6. UI integration

Two options, picking one needs user input — listing for spec completeness:

**Option α (recommended)**: New tab on v2 page named `策略 / Strategies`, between current `市场` and `挂单`. Tab body lists ALL strategy templates from `STRATEGY_CATALOG` as cards. MA-ribbon card is the first one we ship; future strategies plug in alongside.

**Option β**: Inline card embedded in existing market view. Simpler but visually crowded.

Both options share the same card content (§ 3.5). Only the location differs. Plan defaults to α; user can override during plan-writing.

---

## 7. State persistence

- Single file: `data/state/ma_ribbon_auto_state.json`.
- Atomic writes: write to `.tmp`, fsync, rename.
- On startup, scanner loads state. If the file is corrupt (json parse error or schema mismatch), scanner does **not** auto-repair — it sets `enabled = False`, logs the corruption, alerts via Telegram, and waits for user to manually fix and re-enable. We never silently zero out a financial ledger.
- Backups: every state save also appends a snapshot to `data/state/ma_ribbon_auto_state_history.jsonl` (append-only; capped at 10 MB then rotated).

---

## 8. Testing strategy

`PRINCIPLES.md` §P15 binds: "completion = enumerated tests pass."

| Test layer | Coverage | Approx test count |
|---|---|---|
| **Adapter** | signal → ConditionalOrder field-by-field mapping, long + short | 12 |
| **Scanner — formation detection** | bull + bear detection on synthetic OHLCV; dedup of repeat bars | 8 |
| **Scanner — Strategy Y sequencing** | LV1 fires immediately; LV2 only fires at next 15m close while still aligned; skip if alignment broken | 10 |
| **Risk caps** | 25-order cap; per-symbol cap; per-layer hard size; ramp-up day formula | 14 |
| **DD halt** | inject -16 % → halt fires; pos cumulative recovers → halt stays until manual un-halt | 6 |
| **Emergency stop** | flatten + lock + 24 h gate enforced | 5 |
| **State store** | atomic writes; corrupt-file detection; ramp first-enabled-at non-reset | 8 |
| **Watcher SL branch** | ribbon_ema21_trailing path: SL never loosens (long) / never raises (short); manual-line path unchanged regression | 12 |
| **Router** | enable requires both confirm flags; config validator rejects > 5 % risk | 8 |
| **Integration (Bitget mock)** | full end-to-end with mocked httpx: enable → scanner detects fixture event → adapter → store → watcher → mocked Bitget call → reconcile fill → ledger updated → SL trails → SL hits → realized PnL recorded | 4 |
| **TOTAL** | | **~87** |

No live-Bitget tests in the test suite. All Bitget calls are mocked. Live verification is a manual gate (§ 10).

### Property checks the integration test must include

- Once enabled, scanner cannot spawn a layer with `ribbon_meta.ramp_day_cap_pct_at_spawn > 0.15`.
- For every spawned layer, `ribbon_meta.signal_id` shared across LV1/LV2/LV3/LV4 of the same event.
- After emergency_stop, no new ConditionalOrder of `lineage="ma_ribbon"` is inserted for 24 h.
- DD-halt computation never reads from manual-line PnL.

---

## 9. File structure

```
backtests/ma_ribbon_ema21/                       # UNCHANGED — backtest panel
  (existing files; this spec doesn't touch them)

server/strategy/
  ma_ribbon_auto_scanner.py        # NEW
  ma_ribbon_auto_state.py          # NEW
  ma_ribbon_auto_adapter.py        # NEW
  catalog.py                       # MOD: + 1 StrategyTemplate
server/conditionals/
  types.py                         # MOD: OrderConfig.sl_logic, ribbon_meta
  watcher.py                       # MOD: SL branch in _sync_sl_to_line_now, _compute_qty
server/routers/
  ma_ribbon_auto.py                # NEW: 5 endpoints
server/app.py                      # MOD: register router; start scanner task

frontend/js/workbench/
  strategy_card_ma_ribbon.js       # NEW
frontend/v2.html                   # MOD: + 策略 tab (Option α)

data/state/
  ma_ribbon_auto_state.json        # gitignored runtime state
  ma_ribbon_auto_state_history.jsonl  # gitignored append-only history

data/logs/
  ma_ribbon_emergency_stop.log     # gitignored

tests/strategy/
  test_ma_ribbon_auto_scanner.py   # NEW (8+10 tests)
  test_ma_ribbon_auto_adapter.py   # NEW (12 tests)
  test_ma_ribbon_auto_risk_caps.py # NEW (14 tests)
  test_ma_ribbon_auto_dd_halt.py   # NEW (6 tests)
  test_ma_ribbon_auto_emergency.py # NEW (5 tests)
  test_ma_ribbon_auto_state.py     # NEW (8 tests)
  test_ma_ribbon_auto_watcher.py   # NEW (12 tests)
  test_ma_ribbon_auto_router.py    # NEW (8 tests)
  test_ma_ribbon_auto_integration.py # NEW (4 tests)

docs/superpowers/
  specs/2026-04-25-ma-ribbon-auto-execution-design.md   # this file
  plans/2026-04-25-ma-ribbon-auto-execution-plan.md     # written next, by writing-plans
```

---

## 10. Acceptance gates

The implementation phase ends only when ALL of these are true:

- [ ] All ~87 unit + integration tests pass.
- [ ] No-lookahead test re-asserts that scanner uses only closed bars.
- [ ] `_submit_paper` is **not** invoked anywhere on a `lineage="ma_ribbon"` order — verified by grep + a regression test.
- [ ] Watcher regression test confirms `lineage="manual_line"` orders behave identically to before this change.
- [ ] `data/state/ma_ribbon_auto_state.json` is in `.gitignore`.
- [ ] Strategy template visible in `STRATEGY_CATALOG`; appears in `/api/strategy/catalog` response (if such endpoint exists; otherwise just verify import).
- [ ] Frontend card renders without console errors (Playwright smoke test).
- [ ] Telegram alerts work end-to-end (manual one-shot verification).

**Live-go gate** (NOT auto-passed; user clicks):

The user must, in person:
1. Enable in UI with both confirm checkboxes ticked.
2. Verify that day-1 cap is 2 % and ramp-up shows "Day 1 of 14".
3. Wait for the first real signal.
4. Inspect the Bitget app immediately after; confirm the order appears with the correct `ribbon_meta.signal_id` reflected in `clientOid`.
5. Confirm Telegram alert arrived.

Until the user has personally observed step 4 succeed at least once, the strategy stays in "supervised" mode (= scanner active, but each new layer requires a click in the UI to release to live submission). After 1 successful supervised cycle, supervised mode auto-disengages. UI shows a banner `Supervised — click to release` next to each pending layer until then.

This supervised-mode mechanic is **non-negotiable**. It exists to catch the failure modes from `CLAUDE.md` "How I lie" — first run on real money must be witnessed by a human.

---

## 11. Non-goals / explicitly deferred

- **Wyckoff TP** (spec §2.6.2). Phase 4 of original spec; not in this delivery.
- **Multi-symbol portfolio rebalancing.** Scanner treats each (sym, TF, layer) independently.
- **Cross-strategy DD halt.** This strategy's halt only stops itself.
- **Bear-side Wyckoff TP.** Not in scope; bear shorts exit only on trailing SL.
- **Funding-rate cost in PnL ledger.** Currently ignored; flagged as known approximation.
- **Per-symbol parameter optimisation.** Default ribbon_buffer_pct same across all symbols. Phase-5 backtest already shipped for that, but its outputs are NOT auto-applied to live — user must manually update config.
- **Auto-reverse on stop** (`PRINCIPLES.md` §P7). MA-ribbon orders explicitly opt out via `ribbon_meta.reverse_on_stop = False`. The Phase 1 / Phase 2 backtests measured non-reversing behaviour, so live must match.

---

## 12. Mapping to user's verbal requirements

Audit trail — each user statement → spec section.

| User said | Spec section |
|---|---|
| "全 4 个 TF 都开,Strategy Y 逐层进入" | § 4 data flow |
| "200-300 币都可以,按 24h vol 排序 top N" | § 3.2 universe_filter |
| "同时 25 单上限" | § 5.2 |
| "LV1=0.1 %, LV2=0.25 %, LV3=0.5 %, LV4=2 %" | § 5.1 + § 3.2 layer_risk_pct |
| "bear ribbon 也开,但是写好一点不要错" | § 3.1 (bear pipeline mirrors bull) + § 8 (12 adapter tests cover both) |
| "可选模式 / 可启用可禁用" | § 3.4 endpoints + § 3.5 UI |
| "自动定时扫描" | § 3.1 60 s tick |
| "最多不亏 15 %" | § 5.4 DD halt |
| "把 paper 拆出去,主系统只 live" | § 0 + § 1 glossary "Dry-run/paper" + § 8 grep gate |
| "首次启用 14 天 ramp-up" (Option A) | § 5.5 |
| "代码要严谨,不能错" | § 8 test budget (87) + § 10 supervised-mode gate |

---

## 13. Open questions (will resolve during plan-writing)

1. **UI placement** — Option α (new tab) vs β (inline card). Default: α.
2. **Strategy capital source** — does the scanner read live Bitget account equity, or a user-configured fixed USD? Default: configured fixed USD; auto-equity is Phase-2 of this integration.
3. **Universe refresh cadence** — hourly? once a day? Default: hourly.
4. **Telegram destination** — same chat as manual-line alerts, or new channel? Default: same chat with `[MA-RIBBON]` prefix.

---

## 14. Next steps

1. User reviews this spec. Fixes / changes are inline.
2. User approves.
3. Invoke `superpowers:writing-plans` with this spec as input.
4. Output: `docs/superpowers/plans/2026-04-25-ma-ribbon-auto-execution-plan.md` — a step-by-step task list.
5. User reviews plan.
6. Implementation via `superpowers:subagent-driven-development` or `executing-plans`.

**No code is written until step 5.**

End of spec.
