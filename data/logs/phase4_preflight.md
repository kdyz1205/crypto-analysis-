# Phase 4 Pre-Live Checklist

**Date:** 2026-04-19 UTC
**Equity (Bitget live):** $4.15313758 USDT
**Runner state:** idle (not running)
**Yesterday's outcome:** -81.2% DD (equity $4.93 → $0.93 intraday, then recovered to $4.15 overnight funding/close)
**Verdict:** **NO-GO** — see Section 1 and Section 5.

---

## 1. Config Sanity (DEFAULT_RUNNER_CFG)

Read from `server/strategy/mar_bb_runner.py` lines 52-125.

| Key | Current | Verdict | Notes / Safer Value |
|---|---|---|---|
| `top_n` | 100 | [OK] | Scan scope fine |
| `timeframes` | `["15m","1h","4h"]` | [OK] | 5m correctly dropped (memory note 2026-04) |
| `notional_usd` | 12.0 | [WARN] | Only used in `fixed_notional` fallback. $12 > $4.15 equity → Bitget would reject. But sizing_mode=`fixed_risk` so this is rarely hit. Lower to **6.0** to be safe. |
| `leverage` | 30 | [FAIL] | **CRITICAL**: 30x on $4 equity = 3.33% adverse move → liquidation. Typical 4h trendline SL ~0.5%, but noise + funding can spike. Propose **leverage = 10** (keeps liq at ~10% adverse). |
| `max_concurrent_positions` | 100 | [FAIL] | **CRITICAL**: Each trade uses ~$0.80 margin @ 30x. Equity only supports ~4 concurrent before margin call. Set to **3**. |
| `max_position_pct` | 0.50 | [WARN] | Allows $4.15 × 0.50 × 30 = **$62 notional per trade** — that's 15x equity in a single position. Propose **0.20** (max $24.90 per trade). |
| `risk_pct` fallback | 0.01 | [OK] | 1% fallback is sane; tf_risk drives actual sizing. |
| `tf_risk[4h]` | 0.030 (3%) | [WARN] | 3% of $4 = $0.12 risk per trade. On $4 account, a 5-loss streak = $0.62 = 15% DD. Propose keeping 3% BUT only if `max_concurrent <= 3`. |
| `dry_run` | false | [OK] | Intended for live. |
| `auto_start` | true | [FAIL] | **DANGEROUS** with $4 equity + unverified fixes. Set **auto_start=false**. User must start manually from UI after confirming modal. |
| `strategies` | `["trendline"]` | [OK] | MA Ribbon disabled — correct. |
| `model_gate.enabled` | true | [OK] | Both checkpoints readable (verified in Section 4). |
| `model_gate.fail_open` | true | [WARN] | If models fail to load, runner trades unfiltered. Propose **fail_open=false** for live — better to miss trades than run ungated. |
| `model_gate.min_trade_win_prob` | 0.55 | [OK] | Matches memory note. |
| `trendline_reversal[4h]` | false | [OK] | Reversal disabled — correct per backtest. |
| `daily_dd_tiers` | `[... (0, 0.50)]` | [FAIL] | `<$500` tier allows **50% daily DD**. On $4.15 that's $2.07 loss allowed → survivable but halves account in one day. Propose tightening to **0.20 (20%)** at the `<$500` tier. Yesterday's 81% DD slipped past 50% because risk math was wrong (see Section 5). |

### Proposed safer config for $4 account

```python
{
  "leverage": 10,
  "max_concurrent_positions": 3,
  "max_position_pct": 0.20,
  "auto_start": False,
  "model_gate": {"fail_open": False, ...},
  "daily_dd_tiers": [..., (500, 0.25), (0, 0.20)],
  "notional_usd": 6.0,
}
```

---

## 2. Bitget Connectivity Dry Check

**Endpoint:** `GET /api/live-execution/account?mode=live` (read-only, does NOT place orders).

```json
{
  "ok": true,
  "mode": "live",
  "account_accessible": true,
  "total_equity": 4.15313758,
  "usdt_available": 4.15313758,
  "positions": [],
  "pending_orders": []
}
```

- [OK] API keys work (code=`00000`-equivalent returned, account accessible).
- [OK] Equity $4.15 matches user's statement.
- [OK] **0 open positions** — no leftover exposure from yesterday.
- [WARN] `pending_orders: []` from this endpoint — BUT see `/api/live-execution/plan-orders`:

```
symbol=HYPEUSDT side=sell posSide=short size=0.15 trigger=43.21
  SL=43.253 TP=43.08 clientOid=replan_cond_278b40a849d593_17765
```

- [FAIL] **ORPHAN PLAN ORDER on HYPEUSDT** (short, notional ~$6.50 @ 43.21). This is a leftover from yesterday's session. `clientOid` prefix `replan_cond_` = came from the conditional-watcher replan path. If the runner starts now, it will NOT know about this order, and the same symbol could get a duplicate entry or close-conflict.

**Action required before start:** cancel HYPEUSDT plan order via
```
POST /api/live-execution/cancel-order?symbol=HYPEUSDT&order_id=1429669134232707073&mode=live
```

---

## 3. Kill Switches — Trace Each

### 3a) 紧急平仓 button → `POST /api/live-execution/flatten-all`

**File:** `server/routers/live_execution.py:94-181`

- Confirm code required: `"FLATTEN"` (line 100). [OK]
- Does: (1) fetches live positions from Bitget, (2) calls `close_live_position` for each, (3) if `cancel_plans=true`, cancels all trendline plan orders via `cancel_all_trendline_plan_orders`.
- **DOES close:** open positions, trendline-tagged plan orders (those with `clientOid` starting with known prefixes).
- **DOES NOT close:** plan orders NOT tagged as trendline (e.g., manually-drawn lines via `/api/drawings/manual/...` that use different prefix). **The orphan HYPEUSDT order has prefix `replan_cond_` which may or may not be in the cancel filter — needs verification.**
- [WARN] Verify `cancel_all_trendline_plan_orders` prefix list includes `replan_cond_`. If not, flatten-all leaves that order alive.

### 3b) 停止 runner button → `POST /api/mar-bb/stop`

**File:** `server/routers/mar_bb_runner.py:47-49` → `stop_runner()` in `mar_bb_runner.py:1865-1874`.

- Sets `_running = False`, cancels `_task` and `_maintenance_task`, sets status="stopped", saves state.
- **DOES close:** the scanning loop. No new orders will be placed.
- **DOES NOT close:** (1) already-placed Bitget plan orders, (2) open positions. These continue to manage themselves via Bitget's own TP/SL triggers. [OK expected behavior, documented]

### 3c) Daily DD auto-halt → `_check_daily_dd` in `mar_bb_runner.py:988-1017`

- Called from `_do_scan` before any order submission (line 1464).
- If halted: sets `last_error`, cancels outstanding trendline plan orders, returns early.
- **DOES close:** the scanning tick + pending trendline plans.
- **DOES NOT close:** existing open positions (they keep their own TP/SL).
- [WARN] As noted Section 1: `<$500` tier = 50% limit. On $4.15 that's $2.07 tolerated loss. Yesterday's actual 81% loss slipped the guardrail — investigation needed.

### 3d) Process kill (close `Trading OS.bat` window)

- Ctrl-C / window-close kills `python run.py` → uvicorn shuts down.
- **DOES close:** the entire HTTP server, runner loop, all Python tasks.
- **DOES NOT close:** (1) Bitget plan orders (exist exchange-side), (2) open positions, (3) any TP/SL attached to those positions. Bitget will keep managing them per its own triggers until manually cancelled via the app or another API call.

**Rule of thumb:** killing the process is safe-ish because positions have exchange-side SL/TP, but pending plan-orders that haven't triggered yet continue to live on Bitget's side as tripwires.

---

## 4. Model Gate Wired

- **Checkpoints on disk:**
  - `C:/Users/alexl/trading-system/checkpoints/trendline_quality/v3_trade_outcome_auc_0.825.pt` — 73,653 bytes [OK]
  - `C:/Users/alexl/Desktop/crypto-analysis-/checkpoints/trendline_quality/pattern_bounce_auc_0.928.pt` — 33,634 bytes [OK]
- **Dry-import test:** Ran `_load_model('trade', PRIMARY)` and `_load_model('aux', AUX)` from `trendline_model_gate.py` — both returned `_LoadedModel` instance, no exception. [OK]
- `score_trendline_gate()` wired at `mar_bb_runner.py` scan path (verified by `model_gate.enabled=true` in DEFAULT_RUNNER_CFG and min_trade/line thresholds = 0.55).
- [WARN] `fail_open=true` means a load failure silently passes everything through. Consider flipping to false for live (see Section 1).
- [INFO] No `gate.validate()` method exists on this module — validation is implicit via `_load_model`. Dry-import path above is the equivalent.

---

## 5. First-Trade Simulation (WITHOUT placing real order)

Using a real recent 15m long signal from `data/trade_log.jsonl` (PENDLEUSDT 2026-04-18T19:48):

| Field | Value |
|---|---|
| Symbol / TF / Dir | PENDLEUSDT / 15m / long |
| Entry | 1.3659759 |
| Stop | 1.3644748 |
| TP | 1.3884920 |
| SL distance | 0.1099% |
| Equity | $4.15 |
| `risk_pct[15m]` | 0.007 (0.7%) |
| Target risk $ | $0.0291 |
| Raw notional (risk / SL_dist) | $26.44 |
| Max notional cap (50% × 30× × $4.15) | $62.25 |
| **Final notional** | **$26.44** |
| Margin used (notional / 30) | $0.881 |
| Margin % of equity | **21.2%** |
| Qty | 19.35 PENDLE |
| **Risk $ if stop hits** | **$0.0291 (0.70%)** |

Per-trade risk check: **0.70% ≤ 3% cap** [OK].

### Catastrophic scenario checks

- **Liquidation distance:** 100/30 = **3.33% adverse move** wipes the margin. PENDLE 15m ATR can easily exceed that on news spikes. [WARN]
- **Slippage on stop:** market orders on thin books → real loss often 2-3× planned. $0.029 → $0.06-0.09 per trade.
- **Max concurrent 100 × $0.88 margin = $88** vs equity $4.15. Runner will hit Bitget margin rejection after ~4 concurrent. Orders `N+1` through `N+100` waste API quota and fill the `orders_rejected` counter.
- **4h 3% risk × 5-loss streak** = $0.62 (15% of equity). Hitting the 50%-tier DD halt requires ~17 sequential losses. With current model gate at 0.55 that's improbable but yesterday's 1W/7L = 12.5% WR says the gate may not be holding yet.

**Yesterday's actual damage (from `mar_bb_daily_risk.json`):**
- `equity_start`: $4.929
- `last_equity`: $0.926 → intraday DD = **81.2%**
- `halted: true` (but only AFTER the damage was done)
- 49 orders submitted, 8 reached `history-position`, 1 win / 7 losses, -$0.66 net realized

**Conclusion:** The daily-DD guard halts *future* scans, not in-flight orders already on the exchange. And 50% as the floor tier is too loose.

---

## 6. Rollback Plan — 30-Second Recovery

If the runner starts and begins doing bad things (wrong side, too-large sizes, runaway orders):

### Step-by-step (literal, memorize this order)

1. **Second 0-5:** Open a browser tab to `http://localhost:8000/v2`.
2. **Second 5-10:** Click the red **紧急平仓** button in the runner panel.
3. **Second 10-15:** Modal prompts for `FLATTEN` — type it, press Enter. This closes every open position and cancels every trendline plan order in one call.
4. **Second 15-20:** Click the **停止** runner button. This kills the scanning loop so no new orders can be issued.
5. **Second 20-30:** **VERIFY ON BITGET APP** — open the Bitget mobile app, go to Futures → 持仓 (positions) tab and 计划委托 (plan orders) tab. Both must be empty. Take a screenshot. If anything remains, use the app's manual close/cancel buttons immediately.

### Fallback if the UI is frozen

- **CLI emergency:**
  ```bash
  curl -X POST http://localhost:8000/api/live-execution/flatten-all \
    -H "Content-Type: application/json" \
    -d '{"mode":"live","confirm_code":"FLATTEN","cancel_plans":true}'
  curl -X POST http://localhost:8000/api/mar-bb/stop
  ```
- **Nuclear:** close the `Trading OS.bat` window. Positions continue being managed by Bitget-side SL/TP but **pending plan orders still live on Bitget** — you MUST log into the Bitget app and cancel them manually, otherwise they will trigger later when price touches and open unmanaged positions.

### Pre-start sanity (do these BEFORE clicking 启动 live)

1. Cancel the orphan HYPEUSDT plan order (Section 2).
2. Lower `leverage` to 10 and `max_concurrent_positions` to 3 via `POST /api/mar-bb/update-config`.
3. Set `auto_start=false` so a crash-restart doesn't auto-resume.
4. Verify the UI daily-halt indicator is green (yesterday's halt expired at UTC midnight — confirm today shows "NOT HALTED").
5. Watch the first 3 scan ticks in real-time (60s × 3 = 3min) before walking away. Read the server log for `[mar_bb] sizing:` lines — verify margin % of equity line is < 25%.

---

## Summary Table

| Item | Status |
|---|---|
| Config sanity | [FAIL] (leverage 30, max_concurrent 100, auto_start true, DD tier too loose) |
| Bitget connectivity | [WARN] (orphan HYPEUSDT plan order) |
| Kill switches wired | [OK] with caveat on plan-order prefix match |
| Model gate loads | [OK] |
| First-trade math sane | [WARN] (per-trade risk fine, but portfolio-level concurrency math dangerous) |
| Rollback plan documented | [OK] |
| **Overall** | **NO-GO** until config tightened + orphan order cleared |
