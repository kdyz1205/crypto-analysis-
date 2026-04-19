# Incident post-mortem: 2026-04-18 81.2% daily DD (limit 50%)

**Status:** Forensic. No code changes. Recommendations only — team lead decides what to merge.

## TL;DR

Equity went from $4.9289 → $0.9263 (loss $4.00, 81.2% of day's start). The 50% DD line ($2.46 loss) was crossed silently and another $1.54 of equity bled out before the halt flag was persisted at 23:11:38 UTC. Three structural defects worked together:

1. **Halt is checked only at scan start (once per 60s)**, not on every fill / every equity delta. A fast streak of SL hits can overshoot the limit by >30 percentage points within a single scan interval.
2. **The 10 s maintenance loop has zero halt awareness.** Preset SL/TP continues to fire on Bitget, in-flight plan orders continue to trigger, and `_sync_trendline_fills_and_update_trailing` keeps registering new fills and moving trailing stops regardless of halt state.
3. **`cancel_all_trendline_plan_orders` returned `cancelled=0, failed=0`** at halt time — meaning Bitget had no pending plan orders to cancel. The damage had already been done via positions that had *already filled*; the halt action was a no-op for every live position.

On top of that, `trade_log.jsonl` has systematically corrupt price fields (`entry_price=0, close_price=0, pnl_pct=0` on every `close` event for 2026-04-18) because the Bitget-field-name fallback chain in `mar_bb_runner.py:815-816` uses the wrong primary key (`openPriceAvg` / `closePriceAvg`) while Bitget actually returns `openAvgPrice` / `closeAvgPrice` (see `mar_bb_history.py:52` and `_row_float` usage in `trendline_order_manager.py:497-498`).

---

## 1. Why the halt fired LATE

### 1.1 Exact code path (`server/strategy/mar_bb_runner.py`)

The only place the halt flag is set is inside `_do_scan()` (line 1441):

```
L1454  equity = await _get_equity()                   # one HTTP call per scan
L1464  dd_halted, dd_current, dd_limit = _check_daily_dd(equity, cfg)
L1465  if dd_halted:
L1474      await cancel_all_trendline_plan_orders(cfg, status="daily_halt")
L1482      return                                      # stop scan, no new orders
```

`_check_daily_dd` (line 988-1017):

```
dd_pct = (_daily_equity_start - equity) / _daily_equity_start
limit  = _get_daily_dd_limit(_daily_equity_start, cfg)
halted = dd_pct >= limit
_save_daily_risk(equity=equity, dd_pct=dd_pct, limit=limit, halted=halted)
```

Scan loop (line 1803-1821) runs `_do_scan` then `await asyncio.sleep(60)`. So:

- Minimum window between halt evaluations: 60 s (configurable via `scan_interval_s`, default 60).
- Actual gap can be >60 s if a scan itself takes time (last recorded `last_scan_duration_s = 0.74`, but under full top-100 scan of 4 TFs it can be 15-30 s in practice).

### 1.2 The parallel 10 s maintenance loop is halt-blind

`_maintenance_loop()` (line 1830-1842) runs every `maintenance_interval_s=10`:

```python
while _running:
    await _run_trendline_fast_maintenance(cfg)   # NO halt check anywhere
    await asyncio.sleep(10)
```

`_run_trendline_fast_maintenance` (line 860-888) calls two side-effectful functions every 10 s regardless of halt:

1. `update_trendline_orders([], current_bar_index=-1, cfg=...)` — does NOT place new entries because `new_signals=[]`, but DOES cancel orphaned orders on Bitget via `cancel_plan_order`. Benign.
2. `_sync_trendline_fills_and_update_trailing(cfg)` — this one is **not benign**:
   - Detects fills that happened since last check and registers them in `_trendline_params`.
   - Calls `_update_trailing_stops(cfg)` which issues `adapter.update_position_sl_tp(...)` — **moves SL on live positions during halt**.
   - If a held position disappears from Bitget since last sync, it logs the close (via the wrong-field path — see §2).

Grep confirms: `server/strategy/trendline_order_manager.py` and `server/execution/live_adapter.py` contain **zero** references to `halted`, `daily_halt`, or `dd_halted`.

### 1.3 Server-side Bitget orders keep firing

Every trendline entry is a `normal_plan` trigger-market order preset on Bitget:

- Trigger: price crosses the trendline ± buffer.
- Once triggered, Bitget market-opens the position with attached SL/TP presets.
- **Neither side is cancellable by local halt once they're armed on Bitget.**

`cancel_all_trendline_plan_orders` (line 1009 in `trendline_order_manager.py`) only cancels:
- Pending plan orders that are still un-triggered on Bitget (these fire, open a position, and Bitget's preset SL closes it — all server-side, zero halt interaction).
- It does NOT close existing open positions. It does NOT cancel their attached SL/TP.

Log evidence at halt time:
```
[mar_bb] DAILY DD HALT: lost 81.2% today (limit 50% for $5 tier). No new trades until UTC midnight.
[mar_bb] daily DD halt cancelled trendline plans: {'cancelled': 0, 'failed': 0, 'status': 'daily_halt'}
```

`cancelled=0` → by the time halt fired, no pending plan orders existed to cancel. All damage was already baked into live positions.

### 1.4 Enumerated ways loss continued past the 50% line

Concrete paths by which $2.50 loss grew to $4.00 after the threshold was crossed:

| # | Source | Halt-blind? | Why |
|---|--------|-------------|-----|
| A | **Preset SL/TP attached to open position** | Yes (Bitget-side) | SL is stored on Bitget; when price touches it, Bitget market-closes. Local halt can't reach it. |
| B | **Pending plan order that triggers between scans** | Yes | Trigger armed on Bitget; local halt is set later. Position opens, immediately goes underwater, SL eventually fires. |
| C | **Maintenance loop moves SL on halt day** | Yes (local, but not halt-aware) | `_update_trailing_stops` in `mar_bb_runner.py:682` continues running every 10 s. Moving SL can lock in a worse exit. |
| D | **Trendline reversal on SL** | Yes (half the time) | `_check_and_fire_reversals` (line 214) runs inside `_do_scan` *after* signal scanning. If a scan passes the halt check, then a SL hits mid-scan, the reversal fires before next halt check. Disabled on all TFs in current config, but still a latent footgun. |
| E | **Stale equity read** | Yes | `_get_equity()` is called ONCE per scan at line 1454. If equity drops another 10% during the 15-30 s scan, the check below still uses the stale pre-scan value. |
| F | **Dust positions** | Yes | Positions too small for the history-position detector to match within 120 s close window. Their losses are in equity but never get a `log_close` → invisible. |

### 1.5 Accounting gap (strong evidence for paths A/B/F)

Sum of logged `close.pnl` for 2026-04-18: **−$0.32**.
Actual equity loss: **−$4.00**.
Unaccounted: **−$3.68** (92% of the day's loss).

Something closed or bled from open positions where the `log_close` call in `mar_bb_runner.py:819` was never reached — either because the `history-position` fallback returned empty (`_sync_trendline_fills_and_update_trailing` line 811-817) or because the position was closed by Bitget's preset before the 10 s maintenance loop noticed it was gone. The `pnl` number shown in `last_equity=$0.93` is truth (Bitget equity); the logged `sum(pnl)` is a vast under-count. **Trade log is not a reliable source of truth for PnL** — only for trade count / orders placed.

---

## 2. Why `trade_log.jsonl` has zero prices

### 2.1 Evidence

All 22 `close` events on 2026-04-18: `entry_price=0.0, close_price=0.0, pnl_pct=0`. The `pnl` field IS correct (it comes from `netProfit`). All 22 closes have empty `order_id` — confirms they all went through the `_sync_trendline_fills_and_update_trailing` fallback, not `_log_closed_from_history`.

### 2.2 Root cause — one wrong field name

`server/strategy/mar_bb_runner.py:815-816`:

```python
open_price  = float(last.get("openPriceAvg")  or 0)   # WRONG KEY
close_price = float(last.get("closePriceAvg") or 0)   # WRONG KEY
```

Bitget's `/api/v2/mix/position/history-position` actually returns **`openAvgPrice`** / **`closeAvgPrice`** (camelCase, `Avg` in middle). Documented in the same repo:

- `server/strategy/mar_bb_history.py:52` — comment lists Bitget fields as `openAvgPrice, closeAvgPrice`.
- `server/strategy/mar_bb_history.py:133-134` — `r.get("openAvgPrice")` / `r.get("closeAvgPrice")` (correct).
- `server/strategy/mar_bb_runner.py:261` — `row.get("openAvgPrice")` (correct, used elsewhere in same file).
- `server/strategy/trendline_order_manager.py:497-498` — `_row_float(row, "openPriceAvg", "openAvgPrice", "averageOpenPrice")` tries all three.
- `server/execution/live_adapter.py:712, 765-767` — uses all three with fallback.

So the pattern `openAvgPrice` is the correct one and is used correctly in 4 places; only line 815-816 of `mar_bb_runner.py` got it wrong. This is a single-location bug.

Because `last.get("openPriceAvg")` returns `None` → `float(None or 0) = 0.0`, both open_price and close_price end up at 0. `pnl_pct = pnl / margin` on line 817 uses the `margin` field which also may or may not return — worth checking if margin falls back to 1 via `or 1`.

### 2.3 Proposed fix (do not apply — informational)

**File:** `server/strategy/mar_bb_runner.py`
**Lines:** 815-817

Replace:
```python
pnl         = float(last.get("netProfit") or last.get("achievedProfits") or 0)
open_price  = float(last.get("openPriceAvg") or 0)
close_price = float(last.get("closePriceAvg") or 0)
pnl_pct     = pnl / float(last.get("margin") or 1) if float(last.get("margin") or 0) > 0 else 0
```

With the same fallback pattern already used in `trendline_order_manager.py:497-501` (DRY: extract `_row_float` to a shared helper):

```python
def _row_float(row, *keys):
    for key in keys:
        raw = row.get(key)
        if raw in (None, "", 0, "0"): continue
        try:
            v = float(raw)
            if v != 0: return v
        except (TypeError, ValueError):
            continue
    return 0.0

pnl         = _row_float(last, "netProfit", "achievedProfits", "net_pnl", "pnl")
open_price  = _row_float(last, "openAvgPrice", "openPriceAvg", "averageOpenPrice")
close_price = _row_float(last, "closeAvgPrice", "closePriceAvg", "averageClosePrice")
margin      = abs(_row_float(last, "margin", "positionMargin", "openMargin"))
pnl_pct     = pnl / margin if margin > 0 else 0.0
```

Then the `log_close(..., entry_price=open_price, ...)` call at line 819-821 will record real prices. Same change also needed in `log_close` ordering — current signature already has `close_price` as 4th positional arg.

Apply the same principle check (per `CLAUDE.md`):
> "Bug fix ≠ patching the symptom. Bug fix = abstract the principle + grep every violation + fix in one pass."

Principle: **Bitget field names must be read with the full fallback chain `openAvgPrice, openPriceAvg, averageOpenPrice` every time.**

Grep of other violations:
- `server/strategy/mar_bb_runner.py:261` — OK (only `openAvgPrice`, but this is one variant, inconsistent).
- `server/strategy/mar_bb_runner.py:445-447` — position fetch, uses all three ✓.
- `server/strategy/mar_bb_runner.py:594` — position fetch, uses all three ✓.
- `server/strategy/mar_bb_runner.py:815-816` — **THIS IS THE BUG**, uses only wrong variant.
- `server/strategy/evolution/daily_report.py:142` — only `openAvgPrice`; acceptable for history rows but will quietly return 0 on some rows.
- `server/conditionals/watcher.py:913` — uses all three ✓.

**Recommended action:** introduce `server/strategy/_bitget_fields.py` with `_row_float`, `_row_ts`, `_open_price(row)`, `_close_price(row)` helpers. Migrate every price-extract site to use them. Prevents class of bug.

### 2.4 Integrity guard — fail loud

The write-time assertion should be in `server/strategy/trade_log.py:log_close` (line 60):

```python
def log_close(order_id, symbol, direction, close_price, pnl, pnl_pct, reason="", **extra):
    entry_price = extra.get("entry_price", 0)
    assert close_price > 0 or pnl == 0, (
        f"log_close: close_price=0 with non-zero pnl for {symbol} (caller passed wrong Bitget field?)"
    )
    assert entry_price > 0 or pnl == 0, (
        f"log_close: entry_price=0 with non-zero pnl for {symbol}"
    )
    ...
```

Or softer: print a loud warning and write the row anyway (so we don't break live trading on a logging bug). In production prefer the soft version; in CI / tests use `pytest.raises(AssertionError)` to pin.

---

## 3. Root cause of the poor win rate

### 3.1 2026-04-18 breakdown by TF (22 close events from trade_log.jsonl)

| TF   | Count | Wins | Losses | WR     | Total PnL |
|------|------:|-----:|-------:|-------:|----------:|
| 5m   | 12    | 4    | 8      | 33.3%  | $+0.0979  |
| 15m  | 9     | 2    | 7      | 22.2%  | $−0.3561  |
| 1h   | 1     | 0    | 1      | 0.0%   | $−0.0649  |
| 4h   | 0     | 0    | 0      | —      | $0        |
| **Total** | **22** | **6** | **16** | **27.3%** | **$−0.3231** |

### 3.2 2-day window (2026-04-17 + 2026-04-18, 52 closes — matches the user's "~45 closes / 17.8 % WR" framing, which they may have rounded / used a tighter filter)

| TF   | Count | Wins | Losses | WR     | Total PnL |
|------|------:|-----:|-------:|-------:|----------:|
| 5m   | 27    | 5    | 22     | 18.5%  | $−1.1165  |
| 15m  | 18    | 4    | 14     | 22.2%  | $−0.5995  |
| 1h   | 6     | 0    | 6      | **0.0%** | $−1.4278  |
| 4h   | 1     | 0    | 1      | 0.0%   | $−0.6369  |
| **Total** | **52** | **9** | **43** | **17.3%** | **$−3.7807** |

### 3.3 Per-TF plan-orders placed on 2026-04-18

| TF  | plan-orders placed |
|-----|------:|
| 5m  | 75 |
| 15m | 21 |
| 1h  | 19 |
| 4h  | 0 |

5m accounted for **65% of attempts** but only 23% of closed (lots of plan orders never triggered / got cancelled).

### 3.4 Verdict — was 5 m the main killer?

**5 m was the volume driver, but NOT the largest dollar-loss cause yesterday.** On 4-18 alone, 5 m actually netted **positive $0.10** across 12 closes (small wins offset small losses). 15 m was the biggest dollar drain on 4-18 at −$0.36 across 9 closes. Across the 2-day window, however, **5 m was the worst single TF by trade count (27) and cumulative loss ($−1.12), and 1 h had a catastrophic 0% WR over 6 trades**.

This aligns with the memory notes (`project_trendline_tf_decision.md`): 5 m edge is negative / marginal; 15 m edge is +0.065% but needs ML gate ≥ 0.55; only 1 h and 4 h have unfiltered edge. The fact that 1 h posted 0/6 yesterday is *not* a statement about strategy — n=6 is way below the noise floor. The memory says 1 h edge is +0.184%, which over 6 trades has large variance.

**The biggest contributor to the 81 % DD is NOT the strategy's win rate; it is the accounting gap**: logged PnL totals −$0.32 on 4-18 while equity fell $4.00. 92 % of the loss is unaccounted in the trade log. The strategy may or may not be the problem — the trade log simply doesn't tell us. See §1.5 for why (Bitget-side preset SL closes + the wrong-field bug hiding close metadata).

### 3.5 Recommended priority

1. Fix the accounting hole first (§2.3). Without it we're flying blind.
2. Drop 5 m immediately (the memory already did this in `DEFAULT_RUNNER_CFG`, but live `mar_bb_state.json` still has it enabled — **the running process was using stale config**).
3. Re-sample 1 h / 4 h edge with proper logging over ≥50 closes before deciding.
4. 15 m should remain on the ML-gate-only list per the memory note.

---

## 4. Proposed guards (pseudo-diff, do not apply)

### 4.1 Halt short-circuit: kill all in-flight plan orders AND close all positions

**File:** `server/strategy/mar_bb_runner.py`
**Location:** inside `_check_daily_dd` immediately after `halted = True`, and inside `_do_scan` after `if dd_halted:` block.

Pseudo-diff:

```python
# mar_bb_runner.py  — extend halt action
if dd_halted:
    _state.last_error = f"DAILY DD HALT: lost {dd_current*100:.1f}% ..."
    print(f"[mar_bb] {_state.last_error}", flush=True)

    # (existing) cancel pending plan orders
    from server.strategy.trendline_order_manager import cancel_all_trendline_plan_orders
    await cancel_all_trendline_plan_orders(cfg, status="daily_halt")

    # NEW #1: close every open position at market
    try:
        from server.execution.live_adapter import LiveExecutionAdapter
        adapter = LiveExecutionAdapter()
        positions, ok = await _get_bitget_positions()
        if ok:
            for p in positions:
                sym = (p.get("symbol") or "").upper()
                hold_side = (p.get("holdSide") or p.get("posSide") or "").lower()
                size = float(p.get("total") or p.get("available") or 0)
                if size > 0 and sym:
                    try:
                        await adapter.close_position_at_market(sym, hold_side,
                                                               size, mode=cfg.get("mode", "live"))
                        print(f"[mar_bb] HALT-FLAT {sym} {hold_side} size={size}", flush=True)
                    except Exception as exc:
                        print(f"[mar_bb] HALT-FLAT err {sym}: {exc}", flush=True)
    except Exception as exc:
        print(f"[mar_bb] halt flatten-all err: {exc}", flush=True)

    # NEW #2: propagate halt flag to maintenance loop
    _state.halt_active = True      # add to RunnerState dataclass
    _save_state()
    return
```

**File:** `server/strategy/mar_bb_runner.py`, function `_run_trendline_fast_maintenance` (line 860).

```python
async def _run_trendline_fast_maintenance(cfg: dict) -> dict:
    # NEW: respect halt
    if getattr(_state, "halt_active", False):
        return {"ok": True, "skipped": "halt_active"}
    ...
```

**File:** `server/execution/live_adapter.py` — add missing method if not present:

```python
async def close_position_at_market(self, symbol, hold_side, size, mode="live"):
    """Immediately market-close a position, bypassing preset SL/TP."""
    body = {
        "symbol": symbol, "productType": "USDT-FUTURES",
        "marginCoin": "USDT", "holdSide": hold_side,
        "size": str(size), "orderType": "market", "side": "close",
    }
    return await self._bitget_request("POST", "/api/v2/mix/order/close-positions",
                                      mode=mode, body=body)
```

### 4.2 Mid-day circuit breaker: 3 losses in a row → 1-hour pause

**File:** `server/strategy/mar_bb_runner.py` — new module-level state + check inside `_do_scan` before signal placement.

```python
# module-level state
_recent_closes: list[tuple[int, float]] = []     # [(ts, pnl), ...]
_pause_until_ts: int = 0

def _record_close_for_circuit_breaker(pnl: float):
    global _recent_closes, _pause_until_ts
    now = int(time.time())
    _recent_closes.append((now, pnl))
    _recent_closes = [(t, p) for (t, p) in _recent_closes if now - t < 3600]
    last_three = [p for (_, p) in _recent_closes[-3:]]
    if len(last_three) == 3 and all(p < 0 for p in last_three):
        _pause_until_ts = now + 3600
        print(f"[mar_bb] CIRCUIT-BREAKER: 3 losses in a row → pause 1 h (until {_pause_until_ts})", flush=True)

# in _do_scan, right after dd_halted check
if int(time.time()) < _pause_until_ts:
    _state.last_error = f"circuit breaker: paused until {_pause_until_ts}"
    print(f"[mar_bb] {_state.last_error}", flush=True)
    return
```

Call `_record_close_for_circuit_breaker(pnl)` at every `log_close` site (lines 818-821, and `trendline_order_manager.py:506-517`).

Note: "3 losses in a row" is a crude signal; a more robust version weights by dollar loss or restricts to losses within a short window (e.g., 3 losses within 30 minutes, not just 3 consecutive).

### 4.3 trade_log integrity: non-zero price assertions

**File:** `server/strategy/trade_log.py`

```python
def log_close(order_id, symbol, direction, close_price, pnl, pnl_pct, reason="", **extra):
    entry_price = extra.get("entry_price", 0)
    if pnl != 0 and (close_price <= 0 or entry_price <= 0):
        # Fail loud but don't kill the process. Log to stderr and a quarantine file.
        import sys
        msg = (f"[trade_log] INTEGRITY WARN {symbol}: close_price={close_price} "
               f"entry_price={entry_price} pnl={pnl} — caller passed wrong Bitget field?")
        print(msg, file=sys.stderr, flush=True)
        # write to quarantine
        Path("data/trade_log_quarantine.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with open("data/trade_log_quarantine.jsonl", "a") as qf:
            qf.write(json.dumps({
                "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol, "close_price": close_price, "entry_price": entry_price,
                "pnl": pnl, "caller_reason": reason, **extra,
            }) + "\n")
    ...  # write the normal row regardless
```

And add a CI test:

```python
# tests/strategy/test_trade_log_integrity.py
def test_log_close_warns_on_zero_price(capsys):
    from server.strategy.trade_log import log_close
    log_close("oid", "BTCUSDT", "long", close_price=0.0, pnl=-1.0, pnl_pct=0.0,
              entry_price=0.0, reason="sl_or_tp")
    captured = capsys.readouterr()
    assert "INTEGRITY WARN" in captured.err
```

---

## 5. Summary of what to fix (ranked)

| # | Severity | Area | File(s) | Effort |
|---|----------|------|---------|--------|
| 1 | **P0** | Halt doesn't close open positions | `mar_bb_runner.py:1465-1482` | M |
| 2 | **P0** | Maintenance loop is halt-blind | `mar_bb_runner.py:1830-1842` + `_run_trendline_fast_maintenance` | S |
| 3 | **P0** | Wrong Bitget field name → all close PnL accounting is blind | `mar_bb_runner.py:815-816` | S |
| 4 | P1 | Stale `mar_bb_state.json` config (5 m still enabled) | ops: `update_config` restart | XS |
| 5 | P1 | Trade-log integrity assertion | `trade_log.py` | S |
| 6 | P1 | Mid-day 3-loss-in-a-row circuit breaker | `mar_bb_runner.py` | M |
| 7 | P2 | Bitget field extractor helper (DRY) | new `server/strategy/_bitget_fields.py` | M |
| 8 | P2 | Equity re-read inside scan for fast dips | `mar_bb_runner.py:1454` + per-TF loop | M |

## 6. Evidence paths (for the on-call next time)

- Daily risk ledger: `C:\Users\alexl\Desktop\crypto-analysis-\data\mar_bb_daily_risk.json`
- Runner state: `C:\Users\alexl\Desktop\crypto-analysis-\data\mar_bb_state.json`
- Trade log: `C:\Users\alexl\Desktop\crypto-analysis-\data\trade_log.jsonl` (923 lines total)
- Halt-check fn: `C:\Users\alexl\Desktop\crypto-analysis-\server\strategy\mar_bb_runner.py:988` (`_check_daily_dd`)
- Halt action block: `C:\Users\alexl\Desktop\crypto-analysis-\server\strategy\mar_bb_runner.py:1464-1482`
- Maintenance loop: `C:\Users\alexl\Desktop\crypto-analysis-\server\strategy\mar_bb_runner.py:1830-1846`
- Wrong Bitget fields: `C:\Users\alexl\Desktop\crypto-analysis-\server\strategy\mar_bb_runner.py:815-816`
- Correct Bitget field ref: `C:\Users\alexl\Desktop\crypto-analysis-\server\strategy\mar_bb_history.py:52, 133-134`
- Cancel-all function: `C:\Users\alexl\Desktop\crypto-analysis-\server\strategy\trendline_order_manager.py:1009`

---

*Drafted by drawdown-auditor, 2026-04-19. Forensic only — no code touched.*
