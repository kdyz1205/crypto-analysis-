# Known Issues — Detailed Diagnosis (2026-04-17)

> Written by Claude after 18+ hours of debugging with Axel.
> Every issue below has been encountered in LIVE trading with real money.
> This document is brutally honest about what went wrong and why.

---

## Issue 1: Trailing SL Never Actually Moves on Bitget

### What it should do
When a position is open (e.g., LINK 1h short), the SL should move at every bar boundary:
- 1h position → SL updates every hour at :00:00 UTC
- 5m position → SL updates every 5 minutes at :x0:00 and :x5:00
- New SL = trendline projection at current bar = `slope × (entry_bar + bars_since) + intercept`

### What actually happens
SL gets set ONCE when the position is first detected, then NEVER moves again.
LINK's SL stayed at 9.62-9.65 from 13:19 to 22:00+ (9 hours, should have moved 9 times for 1h).

### Root causes found

**Cause A: `_is_bar_boundary()` 90-second window too short**
The original code checked if we're within 90 seconds of a bar boundary:
```python
if tf == "1h":
    return m == 0 and s < 90  # only True for 90 seconds per hour
```
But the scan loop takes 60-120 seconds to process 100 coins × 4 TFs. By the time trailing SL code runs, the 90-second window has passed → skip → never updates.

**Fix applied:** Changed to `last_update_bar` tracking (compare `bars_since > last_bar`). This doesn't depend on wall-clock timing.

**Status:** Fix deployed but SL STILL doesn't move (see Cause B, C, D).

**Cause B: `opened_ts` is wrong after server restart**
`register_trendline_params()` sets `opened_ts = int(time.time())` on first registration. After every server restart, this resets to NOW. So `bars_since = (now - opened_ts) / bar_duration` starts from 1 again, even if the position has been open for hours.

This means the line projection recalculates to nearly the same value each restart, so the "never widen" check (`new_sl >= current_sl` for short) blocks the update.

**Fix needed:** Use actual position open time from Bitget (`openPriceAvg` timestamp or `cTime` from position history), not `time.time()` at registration.

**Cause C: `_calc_trendline_trailing_sl()` projection may be far from reality**
The function uses `slope` and `intercept` from the `ActiveLineOrder` which was created when the line was first detected. But `entry_bar` is set to `ao.bar_count - 1` where `bar_count` is the number of bars in the OHLCV data at scan time.

If the OHLCV data length changes between scans (e.g., new data downloaded, different CSV cache hit), `bar_count` changes, and the projection shifts dramatically. This caused IN's SL to be at 0.05871 when entry was 0.065 (10% off).

**Fix needed:** Store the absolute timestamp of the line's anchor points, not bar indices. Use timestamp-based projection that doesn't depend on data array length.

**Cause D: `update_position_sl_tp()` was not actually called**
In some scan cycles, the trailing code reaches `_update_trailing_stops()` but the function returns early because:
1. `_trendline_params` is empty (cleaned up by stale `held_symbols` check)
2. `bitget_positions` returns empty (API failure, returns `ok=False`)
3. All positions skip due to `bars_since <= last_bar`

No error is printed because each skip is a silent `continue`.

**Fix needed:** Add debug logging for every skip reason.

---

## Issue 2: Duplicate SL/TP Orders Keep Accumulating

### What it should do
Each position should have exactly:
- 1 SL order (either `loss_plan` from preset or `pos_loss` from trailing)
- 1 TP order (`profit_plan` from preset)

### What actually happens
Positions accumulate 4-6 orders:
- `loss_plan` (preset SL from plan order trigger) — CANNOT be cancelled without `planType` in body
- `pos_loss` (trailing SL from `place-tpsl-order`) — CAN be cancelled
- `profit_plan` (preset TP) — should stay
- `pos_profit` (trailing TP) — added on first trailing run

### Root cause: Cancel API requires `planType` in request body

**THE KEY DISCOVERY (found at 21:35 UTC after 8 hours of debugging):**

Bitget's `cancel-plan-order` endpoint SILENTLY FAILS if `planType` is not included in the request body:

```python
# This returns success but does NOTHING:
body = {"symbol": "LINKUSDT", "productType": "USDT-FUTURES", "orderId": "123"}
# Response: {"code":"00000","data":{"successList":[],"failureList":[]}}
# ^^^ Empty successList = nothing cancelled!

# This ACTUALLY cancels:
body = {"symbol": "LINKUSDT", "productType": "USDT-FUTURES", "orderId": "123", 
        "planType": "loss_plan"}  # <-- MUST include this
# Response: {"code":"00000","data":{"successList":[{"orderId":"123"}],"failureList":[]}}
```

The Bitget API documentation does NOT clearly state this requirement. The cancel returns HTTP 200 with `code: "00000"` (success) regardless — the only way to tell is checking `successList`.

**Bitget order sub-types (all queried under `planType: "profit_loss"`):**
| planType | UI Name | Origin | Cancel needs |
|----------|---------|--------|-------------|
| `loss_plan` | 部分止损 | Preset from plan order trigger | `planType: "loss_plan"` |
| `profit_plan` | 部分止盈 | Preset from plan order trigger | `planType: "profit_plan"` |
| `pos_loss` | 仓位止损 | `place-tpsl-order` by trailing | `planType: "pos_loss"` |
| `pos_profit` | 仓位止盈 | `place-tpsl-order` by trailing | `planType: "pos_profit"` |

**Fix applied:** Cancel body now includes `planType` from the order's actual `planType` field.
**Verified on real Bitget:** COIN and LINK preset SL successfully cancelled after fix.

---

## Issue 3: System Places New Plan Orders on Symbols with Existing Positions

### What it should do
If MSFT already has an open short position, do NOT place another plan order for MSFT.

### What actually happens
MSFT has:
- Open short position (entry 420.85)
- AND a new plan order (trigger 421.18) trying to open ANOTHER short

### Root cause
The `held_symbols` check at scan start uses `_get_bitget_positions()`, but the trendline order manager compares against its own `existing` dict (active plan orders). If the position was opened by a plan order that's no longer in `active_orders.json` (e.g., after restart), the manager doesn't know the position exists and places a new order.

**Fix needed:** Pass `held_symbols` (from Bitget positions) to `update_trendline_orders()` and skip any symbol already in it. Codex's Finding 5 addresses this.

---

## Issue 4: Entry Price Field Name Mismatch

### What it should do
Read position entry price from Bitget API response.

### What actually happens
Code reads `averageOpenPrice` or `openAvgPrice`, but Bitget returns `openPriceAvg`.
Entry price reads as 0 → SL/TP identification fails → cancel can't distinguish SL from TP.

### Fix applied
Added `openPriceAvg` as first priority in all entry price reads:
```python
entry = float(row.get("openPriceAvg") or row.get("averageOpenPrice") or row.get("openAvgPrice") or 0)
```

---

## Issue 5: Server Port Conflict on Restart

### What it should do
Kill old server → start new server on port 8000.

### What actually happens
`Stop-Process` sometimes doesn't release the port immediately. New server gets `Errno 10048` (port in use). The OLD server continues running with OLD code. All code changes appear to have no effect.

This caused hours of confusion: we'd fix a bug, restart, test, see no change — because the test was hitting the OLD server.

### Fix needed
Before starting new server:
1. Find exact PID holding port 8000: `Get-NetTCPConnection -LocalPort 8000`
2. Kill that specific PID
3. Wait 5 seconds for port release
4. Verify port is free before starting

---

## Issue 6: Python `__pycache__` Serves Stale Code

### What it should do
After editing `.py` files, the server should use the new code.

### What actually happens
Python caches compiled bytecode in `__pycache__/*.pyc`. Even after restarting the server, if the `.pyc` file has a matching timestamp, Python uses the cached version.

This caused the drawing learner capture code to "not exist" for hours — the `.py` file had the code but the `.pyc` served the old version without it.

### Fix needed
Always delete `.pyc` files before restarting:
```bash
find server -name "*.pyc" -delete
```
Or start uvicorn with `--reload` (but this has its own issues on Windows).

---

## Issue 7: Backtest vs Live Execution Mismatch

### What backtest does
- Passive limit fill (price must reach entry level)
- SL checked every bar with pessimistic ordering (stop before target)
- Walking stop recalculated every single bar
- 0.10% taker commission flat

### What live does differently
| Aspect | Backtest | Live | Impact |
|--------|----------|------|--------|
| Entry | `bar.low <= entry` (passive) | Plan order market trigger | Slippage |
| SL activation | Instant (same bar) | Preset from plan order | Similar |
| SL movement | Every bar automatically | Trailing code at bar boundaries | SL often stale |
| Entry price field | Known | `openPriceAvg` (was reading wrong field) | SL/TP calc wrong |
| Commission | 0.10% flat | Varies (maker/taker) | Minor |
| Buffer | Fixed per config | Per-TF from `tf_buffer` map | OK if consistent |

The biggest gap: **SL movement**. Backtest moves SL every bar perfectly. Live trailing has never successfully moved SL more than once.

---

## Issue 8: PyTorch Training Keeps Getting Killed

### What it should do
Run `train_trendline_quality.py` to completion (~2 hours).

### What actually happens
Every time the server is restarted (which happens frequently during debugging), `Stop-Process -Name python` kills ALL python processes, including the PyTorch training.

The training has been started 5+ times and never completed.

### Fix applied (partial)
Started PyTorch as independent Windows process via `Start-Process`. But still gets killed by broad `Stop-Process` calls.

### Fix needed
Run PyTorch training on a completely separate machine, or use a process manager that protects specific PIDs.

---

## Summary: What Works and What Doesn't

### Works ✅
- Drawing trendlines from pivots
- Placing plan orders with preset SL/TP
- Moving plan orders at bar boundaries (MOVED)
- Detecting when plan orders trigger (fill detection)
- Cancel preset SL with `planType: "loss_plan"` in body
- Place new SL via `place-tpsl-order`
- ML drawing capture (after fixing timestamp + pycache issues)
- Backtest framework (V3 simulator, trailing TP comparison)
- PnL tracking (RESOLVED events)

### Does NOT Work ❌
- **Trailing SL auto-movement at bar boundaries** — the core feature
- **Preventing duplicate orders on same symbol**
- **PyTorch training completing**
- **CUDA GPU acceleration**
- **Correct line projection over time** (bar index drift)

### The One Fix That Would Make Everything Work
If `update_position_sl_tp()` correctly:
1. Reads `openPriceAvg` ✅ (fixed)
2. Cancels with `planType` in body ✅ (fixed)
3. Gets called at every bar boundary ❌ (trailing code still broken)
4. Calculates correct new SL from line projection ❌ (bar index drift)

Items 3 and 4 are the remaining blockers.
