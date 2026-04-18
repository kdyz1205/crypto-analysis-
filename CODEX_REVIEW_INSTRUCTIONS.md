# Codex Code Review & Implementation Instructions

## Context

This is a crypto trading bot that automatically:
1. Draws trendlines from pivot points across 100 Bitget coins × 4 TFs (5m/15m/1h/4h)
2. Places plan orders (计划委托) at projected line coordinates + buffer
3. Moves orders at each TF bar boundary
4. Stop loss = line itself (穿线即止损)
5. Trailing SL follows line projection at bar boundaries
6. RR=15, per-TF buffer (5m=0.05%, 15m=0.10%, 1h=0.20%, 4h=0.30%)

## MANDATORY: Read These First

1. `PRINCIPLES.md` — immutable rules for this codebase
2. `TRENDLINE_TRADING_RULES.md` — Axel's trading logic definition
3. `CLAUDE.md` — development rules and anti-patterns

## Code Review Tasks

### 1. Order Execution Flow (CRITICAL)

Review the full order lifecycle and verify each step is correct:

**Files:**
- `server/strategy/trendline_order_manager.py` — plan order placement + bar-boundary updates
- `server/execution/live_adapter.py` — Bitget API calls (plan orders, SL/TP)
- `server/strategy/mar_bb_runner.py` — main scan loop, trailing SL, fill detection

**Check:**
- [ ] Plan order uses `orderType: "market"` with `triggerPrice` = entry price
- [ ] Preset SL attached: `stopLossTriggerPrice` = line price (NOT line + ATR or any offset)
- [ ] Preset TP attached: `stopSurplusTriggerPrice` = entry ± buffer × RR
- [ ] Bar boundary detection (`_is_bar_boundary`) is correct for each TF
- [ ] At bar boundary: cancel old plan order → recalculate line projection → place new order
- [ ] Per-TF buffer is used correctly: `tf_buffer_map.get(tf)` not a fixed value
- [ ] Per-TF risk sizing: `cfg.get("tf_risk", {}).get(tf, risk_pct)`
- [ ] No "broken line detection" or manual `close-positions` — preset SL handles this

### 2. Trailing SL Logic

**File:** `server/strategy/mar_bb_runner.py` (lines ~355-450)

**Check:**
- [ ] `_calc_trendline_trailing_sl()` returns `slope × (entry_bar + bars_since) + intercept`
- [ ] Walking stop: SL = projected line price at current bar (no offset)
- [ ] Only updates at bar boundaries: `if current_sl > 0 and not _is_bar_boundary(tf): continue`
- [ ] Never widens SL (only tightens in profitable direction)
- [ ] `last_sl_set` prevents redundant cancel+replace every scan
- [ ] Price precision: `_round_price()` matches Bitget contract requirements
- [ ] `update_position_sl_tp()` cancels ALL existing SL/TP types before placing new ones

### 3. Fill Detection & Registration

**File:** `server/strategy/mar_bb_runner.py` (lines ~1163-1200)

**Check:**
- [ ] Detects when plan order triggers by comparing active_orders vs Bitget positions
- [ ] `register_trendline_params()` uses `ao.bar_count - 1` (not `ao.anchor2_bar`)
- [ ] `created_ts` from the active order (not `time.time()`)
- [ ] `tp_price` from the active order
- [ ] TF from the active order (not hardcoded)
- [ ] Cleanup uses fresh position data (not stale `held_symbols` from scan start)

### 4. Safety Guards

**File:** `server/strategy/mar_bb_runner.py`

**Check:**
- [ ] P1: Position fetch failure → halt scan (don't proceed with empty held_symbols)
- [ ] P2: Equity = 0 → halt scan (don't fall back to fixed notional)
- [ ] Daily drawdown check works correctly
- [ ] No `try/except: pass` that silently swallows critical errors

### 5. Data Integrity

**Files:**
- `server/strategy/trade_log.py` — trade lifecycle logging
- `server/strategy/ml_trade_db.py` — ML training data
- `server/strategy/drawing_learner.py` — user drawing feature capture

**Check:**
- [ ] Every trade event is logged (plan_placed, fill, sl_move, close)
- [ ] ML trade DB captures full features for PyTorch training
- [ ] Drawing learner captures market features when user draws on website
- [ ] No data loss on server restart (persistent JSON files)

### 6. PyTorch Model Pipeline

**Files:**
- `trading-system/ts/models/trendline_quality/` — full model directory
- `trading-system/scripts/train_trendline_quality.py` — training entry point
- `trading-system/scripts/export_training_data.py` — data export

**Check:**
- [ ] Feature extraction produces 24 features consistently
- [ ] Walk-forward validation (not random split) for time-series data
- [ ] Dual-head model: classification (win/loss) + regression (PnL)
- [ ] StandardScaler fit on train only, applied to test
- [ ] Checkpoint saves model + scaler + config
- [ ] Inference scorer loads checkpoint and scores new trades

## Implementation Tasks (Priority Order)

### P0: Fix Order Workflow
1. Verify plan orders on Bitget actually have correct SL/TP attached
2. Verify MOVED orders at bar boundaries have updated SL/TP (not stale)
3. Add logging: print actual Bitget response to confirm SL/TP fields

### P1: ATR Trailing TP
Replace fixed RR=15 TP with ATR trailing (backtest showed +0.914%/trade on 4h vs +0.505% for RR=15):
- Track peak favorable excursion per position
- Exit when price retraces 1.5×ATR from peak
- Keep preset TP as safety net at RR=30

### P2: Complete PyTorch Training
- Ensure training completes on all TFs (4h → 1h → 15m → 5m)
- Deploy model for live filtering (skip low-quality lines)
- Add preference learning from user drawings

### P3: Bidirectional Learning System
- Capture user drawings with market features
- Compare user-drawn vs algorithm-drawn lines
- Train preference model: P(user_would_draw_this_line)
- Surface "blind spot" discoveries to user

## Key Invariants (NEVER violate)

1. **Buffer IS the total risk.** No separate stop_buffer.
2. **Line IS the stop.** SL = projected line price. Period.
3. **Leverage = account_risk_pct / buffer.** Derived, not chosen.
4. **Orders move at bar boundaries only.** 5m→every 5min, 1h→every hour, etc.
5. **No manual close-positions.** Preset SL handles broken lines automatically.
6. **Per-TF buffer:** 5m=0.05%, 15m=0.10%, 1h=0.20%, 4h=0.30%
7. **Per-TF risk:** 5m=0.3%, 15m=0.7%, 1h=1.5%, 4h=3%
