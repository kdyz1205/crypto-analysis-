# MA Ribbon + EMA21 Buffer Multi-Timeframe Strategy — Design Spec

**Date**: 2026-04-24
**Author**: Claude (under user direction)
**Status**: Awaiting user review before writing-plans
**Phasing**: P1 + P2 detailed. P3 / P4 / P5 architectural — refined after P2 results.

---

## 0. Executive summary

This document specifies a multi-timeframe MA-ribbon trend-continuation strategy that:

1. Detects bullish ribbon alignment (`close > MA5 > MA8 > EMA21 > MA55`) per timeframe.
2. Records the percentage distance between price and each MA at signal time, so we can answer: **"once the ribbon has formed, am I already too late to enter?"**
3. Scales into positions across `5m → 15m → 1h → 4h` as higher TFs confirm continuation.
4. Trails stops via the **current** EMA21 of each TF, offset by a per-TF buffer (`SL = EMA21_TF × (1 − buffer_pct_TF)` for longs).
5. Exits per-TF on either a **baseline TP** (RSI > 75 + MA5 cross-down) or a **Wyckoff distribution + RSI bearish-divergence** detector — both run side-by-side in every backtest, so the data picks the winner.
6. Re-activates per TF when the ribbon re-forms after stop-out / take-profit.

**Two questions to answer with data, not opinion:**

- (Q1) When price pulls back into the EMA21 buffer, is that the last flush before continuation, or actual trend failure? Quantified per `(symbol, TF, buffer_pct)`.
- (Q2) Does the multi-TF scale-in pyramid (5m → 15m → 1h → 4h) actually outperform a single-layer entry on the same TF?

**Non-goals (this spec)**: live trading, real Bitget order placement, cross-symbol "田忌赛马" allocation, short-side mirror. All deferred.

---

## 1. Glossary (single source of truth for terminology)

| Term | Meaning |
|---|---|
| **MA5 / MA8 / MA55** | Simple moving averages over 5 / 8 / 55 closed bars |
| **EMA21** | Exponential moving average over 21 closed bars |
| **Bullish ribbon** | `close > MA5 > MA8 > EMA21 > MA55` (configurable subset) |
| **Layer / LV** | A position chunk anchored to one TF: `LV1=5m`, `LV2=15m`, `LV3=1h`, `LV4=4h` |
| **Buffer (this strategy)** | `% distance from EMA21 used as SL offset`. **NOT** the same `buffer` as in `TA_BASICS.md` §6 (which is line→entry). To avoid collision, code uses `ema21_buffer_pct` |
| **Activation** | A TF transitions from "no eligible signal" to "may fire next layer" |
| **Reset** | After SL/TP, that TF re-enters `INACTIVE` until a fresh activation occurs |
| **Cooldown** | Bars after exit during which reactivation is blocked |
| **Strategy Y** | Time-progressive scale-in: 5m fires first, then await each higher TF's next bar close |
| **Strategy X** | Confluence scale-in: at 5m fire moment, fire all higher TFs that are already aligned |
| **In-sample (IS)** | First 70% of historical data by time — used for parameter tuning |
| **Out-of-sample (OOS)** | Last 30% of historical data by time — final validation, never seen during tuning |
| **MAE / MFE** | Max adverse / favorable excursion of an open position |
| **R** | Risk unit — equity × `risk_budget_pct` for one layer |

---

## 2. Strategy specification

### 2.1 Bullish alignment

For long entries on a given TF, all of the following must be true on the **most recent CLOSED bar**:

```
close > MA5
close > MA8
close > EMA21
close > MA55
MA5 > MA8
MA8 > EMA21
EMA21 > MA55     # configurable, default ON
```

Configurable subset:

```json
{
  "bullish_alignment": {
    "require_close_above_ma5": true,
    "require_close_above_ma8": true,
    "require_close_above_ema21": true,
    "require_close_above_ma55": true,
    "require_ma5_above_ma8": true,
    "require_ma8_above_ema21": true,
    "require_ema21_above_ma55": true
  }
}
```

**Hard rule**: indicators are computed from the bar's CLOSE price. Until the bar has closed, its values do not exist. No look-ahead.

### 2.2 Distance-from-MA features (Phase 1's core question)

At every bar where bullish alignment is `true` AND was `false` on the previous bar (a fresh formation event), record:

```python
distance_to_ma5_pct   = (close - MA5)   / MA5
distance_to_ma8_pct   = (close - MA8)   / MA8
distance_to_ema21_pct = (close - EMA21) / EMA21
distance_to_ma55_pct  = (close - MA55)  / MA55
ribbon_width_pct      = (MA5  - MA55)   / MA55
```

Record forward returns at +5, +10, +20, +50 bars from the formation event (pure forward, computed AFTER the event using closed bars only).

**Distance buckets** (for forward-return cohort analysis):

```
[0%, 0.5%), [0.5%, 1%), [1%, 2%), [2%, 4%), [4%, 7%), [7%, ∞)
```

**Phase 1 deliverable**: per `(symbol, TF, distance_bucket)`:
- count of formation events
- mean / median forward return at +5, +10, +20, +50 bars
- win rate (forward return > 0)
- worst-case drawdown after formation (within 50 bars)

This answers: **"once you can see the ribbon has formed, is it already too late?"**

### 2.3 Multi-TF scale-in — Strategy Y (default) vs Strategy X (alternative)

Both strategies are implemented and run in every backtest. Reports show side-by-side comparison.

#### Strategy Y — time-progressive

State machine per TF:

```
INACTIVE → ARMED → ENTERED → STOPPED_OUT or EXITED_TP → COOLDOWN → INACTIVE
```

Sequence:

1. **5m fires (LV1)**: 5m alignment forms (false → true on a closed 5m bar). Open LV1 at the **next 5m bar's open**.
2. **15m confirms (LV2)**: while LV1 is open, on the next 15m bar close, if 15m alignment is `true` → open LV2 at the next 15m bar's open.
   - If 15m alignment is `false` at that close → 15m enters `COOLDOWN`. LV2 is skipped for this cycle.
3. **1h confirms (LV3)**: same, on the next 1h bar close.
4. **4h confirms (LV4)**: same, on the next 4h bar close.

Each higher-TF layer can only fire **once per activation cycle**. After all layers exit, the entire chain resets.

#### Strategy X — confluence

When 5m alignment forms, snapshot the most-recently-closed 15m / 1h / 4h bars:
- For each higher TF that is currently aligned → fire its layer at the next 5m bar's open (same instant as LV1).
- For each higher TF that is NOT currently aligned → that TF's layer is unavailable for this activation cycle (no late firing).

#### AB-test

Every backtest runs both strategies on the same data and emits both PnL series. Reports include:
- mean PnL per trade (Y vs X)
- Sharpe per trade (Y vs X)
- max drawdown (Y vs X)
- coverage: how often does Y manage to fire all 4 layers vs X firing fewer?

The user's intuition is Y. The data may say X. We respect the data.

### 2.4 Trailing SL — per-layer EMA21 buffer

For each open layer `LV_k`:

```python
# at every closed bar of LV_k's TF:
ema21_now    = EMA21 of TF_k at the bar that just closed
candidate_sl = ema21_now * (1 - ema21_buffer_pct[TF_k])  # for longs
sl[k]        = max(sl[k], candidate_sl)                  # never loosen
```

Default `ema21_buffer_pct` (Phase 2 starting point, per-symbol-tuned in Phase 5):

```
5m  : 1.0%
15m : 4.0%
1h  : 7.0%
4h  : 10.0%
```

These are **examples**, not hardcoded truths. Phase 5 sweeps `{0.25%, 0.5%, 0.75%, 1%, 1.5%, 2%, 3%, 4%, 5%, 7%, 10%}` per `(symbol, TF)`.

#### SL trigger semantics (no look-ahead, no over-optimism)

For a long LV_k, on each bar of TF_k:

```python
if bar.low <= sl[k]:
    # Two cases:
    if bar.open <= sl[k]:
        # Gap down through SL — fill at open (worst case, realistic)
        fill_price = bar.open
    else:
        # Intra-bar break — assume fill at sl[k] (best-case-but-realistic)
        fill_price = sl[k]
    close LV_k at fill_price
```

#### ATR-based alternative (configurable)

```
sl = bar.close - atr_mult * ATR(14)  # tested in Phase 5 as baseline
```

Reports compare EMA21-buffer SL vs ATR SL vs fixed-pct SL.

### 2.5 Position sizing — per-layer risk budget

Two semantics, configurable. The user previously confirmed both `buffer_pct` and `risk_budget_pct` exist as separate concepts.

```python
# at layer-fire time:
entry_price       = open of next bar of TF_k
initial_sl_price  = EMA21_TF_k * (1 - ema21_buffer_pct[TF_k])
sl_distance_pct   = (entry_price - initial_sl_price) / entry_price
risk_dollars_k    = equity * risk_budget_pct[TF_k]
notional_k        = risk_dollars_k / sl_distance_pct
```

Default `risk_budget_pct` (per-layer, NOT cumulative):

```
LV1 (5m) : 1%
LV2 (15m): 3%
LV3 (1h) : 3%
LV4 (4h) : 3%
```

**Cumulative max risk** if all 4 SLs hit simultaneously: `1% + 3% + 3% + 3% = 10% of equity`.

Risk-cap check: before firing LV_k, compute total risk-if-all-stops-hit. If it exceeds `total_risk_cap` (default 12%) → **skip** that layer (do not partially fill).

### 2.6 Take-profit

#### 2.6.1 Baseline TP (mandatory, run every backtest)

For each TF, fire baseline-TP signal when:

```python
RSI(14, TF) > 75
AND (
    MA5(TF) crosses below MA8(TF)
    OR close < MA5(TF) for 2 consecutive closed bars
)
```

When baseline-TP fires on TF_k → close LV_k only (TF-specific layer closure, see §2.6.3).

This is intentionally simple. Its purpose is to be a **floor** that the Wyckoff detector must beat.

#### 2.6.2 Wyckoff distribution + RSI bearish divergence TP

5-stage detector per TF. All 5 stages must be true on a single bar for the signal to fire.

##### Stage A — Prior rally

```python
gain_pct      = (close - close[bars_back_for_rally]) / close[bars_back_for_rally]
ribbon_width  = (MA5 - MA55) / MA55

stage_a = gain_pct >= rally_min_pct[TF] AND ribbon_width >= ribbon_width_min[TF]
```

Defaults (Phase 4 starting point):

```
TF    bars_back  rally_min_pct  ribbon_width_min
5m    20         3%             1.5%
15m   15         5%             3%
1h    10         8%             5%
4h    8          15%            8%
```

##### Stage B — Platform / range

Within the most recent `platform_window_bars` closed bars:

```python
range_pct = (max(high) - min(low)) / mean(close)
stage_b   = range_pct <= platform_range_max_pct[TF]
```

Defaults:

```
TF    platform_window_bars  platform_range_max_pct
5m    10                    2%
15m   8                     3%
1h    6                     5%
4h    4                     7%
```

##### Stage C — MA compression

```python
ma_gap = abs(MA5 - MA8) / EMA21

stage_c = (ma_gap <= ma_compress_max_pct[TF]) OR (MA5 just crossed below MA8)
```

Defaults:

```
TF    ma_compress_max_pct
5m    0.3%
15m   0.5%
1h    0.7%
4h    1.0%
```

##### Stage D — Final push + RSI bearish divergence

Pivot detection (no look-ahead): a bar is confirmed as a "swing high" only after `pivot_confirm_bars` later bars all close below it.

```python
# confirmed pivots only — pivot detection lags by pivot_confirm_bars
recent_pivot       = most recent confirmed swing high in the last pivot_lookback_bars
current_higher_high = close > recent_pivot.price

rsi_now             = RSI(14, TF) at current bar
rsi_at_recent_pivot = RSI(14, TF) at recent_pivot.bar_idx

stage_d = current_higher_high AND (rsi_now < rsi_at_recent_pivot - rsi_div_delta[TF])
```

Defaults:

```
TF    pivot_confirm_bars  pivot_lookback_bars  rsi_div_delta
5m    3                   30                   2.0
15m   3                   25                   2.0
1h    2                   20                   2.0
4h    2                   15                   2.0
```

##### Stage E — Confirmation

At least one of:
- The bar AFTER the new high closes back inside the platform range (`high > platform_high BUT close < platform_high`).
- Volume on the new-high bar is below average of previous `volume_lookback` bars by `volume_weakness_pct`.

Both signals are configurable. Either suffices for Stage E.

##### Wyckoff signal firing

When A AND B AND C AND D AND E are all true on a single bar:
- Wyckoff signal fires on that TF.
- Close LV_k at the next bar's open (TF-specific closure, see §2.6.3).

#### 2.6.3 TF-specific layer closure

When TF_k's TP signal (baseline OR Wyckoff) fires:

| TF firing | Layers closed |
|---|---|
| 5m  | LV1 only |
| 15m | LV1 + LV2 |
| 1h  | LV1 + LV2 + LV3 |
| 4h  | LV1 + LV2 + LV3 + LV4 (all) |

Rationale: if the larger TF says "distribution," the smaller-TF layers within it are also implicated.

#### 2.6.4 Mandatory side-by-side comparison

Every backtest run emits two parallel PnL series:
- One using **baseline TP only** (§2.6.1)
- One using **Wyckoff TP** (§2.6.2), with baseline still active as fallback (whichever fires first wins)

Reports include for each `(symbol, TF, params)`:
- baseline TP: total PnL, Sharpe, max DD, # trades
- Wyckoff TP: total PnL, Sharpe, max DD, # trades, # baseline overrides
- delta: did Wyckoff add or subtract value?

If across the symbol universe Wyckoff TP fails to outperform baseline TP on OOS data, **the Wyckoff feature is killed** in the final report's recommendations.

### 2.7 Reactivation after exit

After ANY layer closes (SL or TP):

1. That TF enters `COOLDOWN` for `cooldown_bars[TF]` (default: 3 bars on TF).
2. After cooldown:
   - That TF returns to `INACTIVE`.
   - On the next bar close where bullish alignment transitions from `false → true`, the TF activates again.
3. Other TFs' layers (if currently open) are unaffected — they trail/exit on their own schedule.

A "wave" can repeat indefinitely while the larger trend continues:

```
Activate → enter → trail → exit (SL or TP) → cooldown → activate again → ...
```

---

## 3. Backtest engine architecture

### 3.1 Bar-event-driven simulation (no time grid)

The simulator advances by **events**, not by uniform time steps. Events are bar-close events from any TF.

```python
event_queue = merge sorted (
    5m bar_close events,
    15m bar_close events,
    1h bar_close events,
    4h bar_close events,
)

for event in event_queue:
    timestamp = event.bar_close_time
    tf = event.tf
    bar = event.bar
    update indicators on tf  # only after this bar closes
    process strategy logic for tf
    update equity curve
```

When two events share the same timestamp (e.g., a 5m close at 10:15 and a 15m close at 10:15), processing order is **smallest TF first** (so 5m is processed before 15m at the same timestamp).

### 3.2 Multi-TF time alignment

At any simulation time T:
- "Latest closed bar" on TF_x = the bar whose `close_time ≤ T`.
- All signals on TF_x use ONLY that bar and earlier.
- Bars whose `close_time > T` are invisible to the simulation.

Test: `test_multi_tf_alignment.py` — at T = 10:10, 5m's latest closed bar is `[10:05, 10:10]`, 15m's latest closed bar is `[09:45, 10:00]`, 1h's latest closed bar is `[09:00, 10:00]`. Assert each.

### 3.3 No-lookahead enforcement

`test_no_lookahead.py`:
- Fixture: 1000 synthetic bars with a known formation at bar 500.
- Walk through one bar at a time, computing signals.
- At each bar `i`, assert that signals depend ONLY on bars `[0, i]` (CLOSED bars 0..i).
- Specifically: at bar 499, the formation must NOT yet be detected.

Run on every commit via pytest.

### 3.4 Fees & slippage

```python
fee_per_side    = 0.0005   # Bitget perp taker, configurable
slippage_per_fill = 0.0001 # 1 basis point, configurable
```

Each fill (entry or exit) deducts `(fee + slippage) × notional` from PnL.

### 3.5 Data source

- Reuse existing project's Bitget OHLCV pipeline (`server/services/data_service.py` or equivalent).
- Per `PRINCIPLES.md` §P1: depth = max available from Bitget. No `tfDays` cap.
- Per `PRINCIPLES.md` §P2: if cache has insufficient depth, refetch.

If existing pipeline cannot serve the backtest needs (e.g., bulk download), build a thin adapter `data_loader.py` that wraps Bitget's history-candles endpoint with pagination + local CSV cache.

### 3.6 Configuration model

One JSON config file per backtest run. Example skeleton:

```json
{
  "universe": ["BTCUSDT", "ETHUSDT", "...", "<40-50 symbols>"],
  "timeframes": ["5m", "15m", "1h", "4h"],
  "data_split": {"train_pct": 0.70, "test_pct": 0.30, "split_by": "time"},
  "moving_averages": {"ma_fast_1": 5, "ma_fast_2": 8, "ema_mid": 21, "ma_slow": 55},
  "rsi": {"period": 14},
  "bullish_alignment": {"...": "see §2.1"},
  "ema21_buffer_pct": {"5m": 0.01, "15m": 0.04, "1h": 0.07, "4h": 0.10},
  "risk_budget_pct":  {"5m": 0.01, "15m": 0.03, "1h": 0.03, "4h": 0.03},
  "total_risk_cap":   0.12,
  "scale_in_strategy": "both",
  "tp": {
    "baseline_enabled": true,
    "wyckoff_enabled":  true,
    "wyckoff_params":   {"...": "see §2.6.2"}
  },
  "cooldown_bars":  {"5m": 3, "15m": 3, "1h": 3, "4h": 3},
  "fees":           {"per_side": 0.0005, "slippage_per_fill": 0.0001},
  "phase":          "P1"
}
```

Phase 5's batch runner enumerates over parameter grids defined in a separate file.

---

## 4. Phasing

Each phase has:
- A self-contained deliverable
- A specific question it answers
- A pass/fail acceptance gate
- A user-review checkpoint before advancing

User must approve each phase's report before the next phase begins.

### Phase 1 — MA alignment + distance analysis

**Deliverable**: For 40–50 USDT-perp symbols × 5m/15m/1h/4h:
- Detect every bullish-alignment formation event (false → true transition).
- Compute distance-to-each-MA at the formation bar.
- Compute forward returns at +5, +10, +20, +50 bars.
- Per-bucket cohort report.

**Question answered**: Once the ribbon has formed, what's the distribution of forward returns? Is the strategy's premise even valid?

**Tests** (all must pass):
- `test_ma_alignment.py` — alignment detection on hand-crafted fixtures
- `test_distance_features.py` — distance calc correctness
- `test_no_lookahead.py` — see §3.3
- `test_multi_tf_alignment.py` — see §3.2
- `test_70_30_split.py` — train/test boundary placement

**Acceptance gate**:
- For at least 30% of symbols on at least one TF, mean forward return at +20 bars in the smallest distance bucket (`[0, 0.5%)`) is `> +1%` post-fee.
- Otherwise: the premise is dead; we stop here, no Phase 2.

**User review**: see the per-bucket forward-return table. Decide whether to proceed.

### Phase 2 — Single-layer entry + EMA21 trailing SL

**Deliverable**: For each `(symbol, TF)`:
- Enter on alignment formation (entry at next bar's open).
- Trail SL via EMA21 buffer.
- No TP. Position closes only on SL hit.
- Sweep `ema21_buffer_pct ∈ {0.5%, 1%, 1.5%, 2%, 3%, 4%, 5%, 7%, 10%}` per TF.
- Compare vs ATR-based SL baseline and fixed-pct SL baseline.

**Question answered**: Does EMA21-buffer trailing SL reduce false stop-outs and improve PnL vs simpler stops? Per coin / TF, what's the best buffer?

**Tests** (additional):
- `test_trailing_stop.py` — SL trails up, never down
- `test_buffer_recovery.py` — for each SL hit, did price recover above EMA21 within 50 bars? (false-stop metric)
- `test_strategy_x_vs_y.py` — placeholder; X / Y compare meaningfully only at P3, but state machine is built here

**Acceptance gate**:
- IS: for at least 30% of symbols, at the symbol-best buffer, post-fee Sharpe > 0.5 on at least one TF.
- OOS: Sharpe ≥ 0.5 × IS Sharpe (else: overfitting flag).
- False-stop rate < 50% for the symbol-best buffer (else: SL is too tight).

**User review**: per-coin best-buffer report + IS/OOS comparison.

### Phase 3 — Multi-TF scale-in (Y vs X)

**Deliverable**:
- Implement LV1–LV4 state machines per TF.
- Implement Strategy Y (time-progressive) and Strategy X (confluence).
- Position sizing per §2.5.
- Reports include both strategies' PnL side-by-side per `(symbol, TF set)`.

**Question answered**: Does pyramiding outperform single-layer? Y or X?

**Acceptance gate**:
- The better of (Y, X) outperforms the best Phase-2 single-layer Sharpe by ≥ 20% on OOS, on at least 30% of symbols.
- Otherwise: pyramiding adds no value; recommend single-layer.

### Phase 4 — Wyckoff TP

**Deliverable**:
- Implement baseline TP (§2.6.1) and Wyckoff TP (§2.6.2).
- Both TPs run in every backtest. Comparison is mandatory.
- Wyckoff thresholds enter the parameter grid.

**Question answered**: Does Wyckoff TP add value over baseline TP on OOS data?

**Acceptance gate**:
- On OOS, Wyckoff TP must beat baseline TP on Sharpe AND total PnL on at least 30% of symbols.
- Otherwise: drop Wyckoff from final recommendation; keep baseline only.

### Phase 5 — Batch param scan + symbol categorization + report

**Deliverable**:
- Batch runner across parameter grids.
- Per-symbol best params output.
- Symbol categorization (e.g., `tight_trender`, `volatile_trender`, `wick_heavy`, `mean_reverting`, `slow_recovery`, `high_false_breakdown`).
- Per-category recommended config.
- Final report: which symbols + TFs + params should the user actually trade live.

**Acceptance gate**:
- Final OOS Sharpe ≥ 1.0 (post-fee) for the recommended config on the recommended symbol subset.
- Recommendations are robust: removing the top 3 best symbols still yields OOS Sharpe ≥ 0.7.

---

## 5. Out-of-sample validation policy (cross-cutting)

This is non-negotiable per `PRINCIPLES.md` §P15.

- All historical OHLCV per `(symbol, TF)` is split by **time**: first 70% IS, last 30% OOS.
- ALL parameter optimization (grid search, Wyckoff threshold tuning, buffer sweep) uses ONLY the IS portion.
- The OOS portion is locked away during tuning. It is read EXACTLY ONCE per final report.
- Reports always show IS and OOS side-by-side. **Never show only IS.**
- Acceptance gates always evaluate OOS, not IS.
- If OOS Sharpe < 0.5 × IS Sharpe → overfitting flag → that parameter is rejected from recommendations.

Walk-forward (refit per period) is a possible future enhancement but **not in scope** for this spec.

---

## 6. File structure

```
backtests/ma_ribbon_ema21/
  README.md                       # how to run
  config.example.json
  config.phase1.json
  config.phase2.json

  data_loader.py                  # Bitget OHLCV with max-depth caching
  indicators.py                   # MA5/MA8/EMA21/MA55/RSI/ATR

  ma_alignment.py                 # alignment detection + state transitions
  distance_features.py            # distance-from-MA features

  signal_state.py                 # per-TF state machine
  multi_tf_levels.py              # LV1-LV4 coordination + Strategy X/Y
  trailing_stop.py                # EMA21 buffer trailing
  position_sizing.py              # risk-budget → notional

  take_profit/
    __init__.py
    baseline_tp.py                # §2.6.1
    wyckoff/
      prior_rally.py              # Stage A
      platform.py                 # Stage B
      ma_compression.py           # Stage C
      rsi_divergence.py           # Stage D (no-lookahead pivot detection)
      exhaustion_confirm.py       # Stage E
      wyckoff_tp.py               # composes A..E

  backtest_engine.py              # bar-event-driven simulator
  batch_runner.py                 # phase 5 grid search
  reports.py                      # IS/OOS side-by-side, cohort tables
  metrics.py                      # Sharpe, MAE/MFE, false-stop rate

  symbol_labeler.py               # phase 5 categorization

  tests/
    test_ma_alignment.py
    test_distance_features.py
    test_no_lookahead.py
    test_multi_tf_alignment.py
    test_70_30_split.py
    test_trailing_stop.py
    test_buffer_recovery.py
    test_signal_activation.py
    test_signal_reset.py
    test_multi_tf_scale_in.py
    test_strategy_x_vs_y.py
    test_position_sizing.py
    test_baseline_tp.py
    test_rsi_divergence_no_lookahead.py
    test_wyckoff_full_chain.py
    test_take_profit_tf_specific.py

docs/superpowers/specs/
  2026-04-24-ma-ribbon-ema21-multi-tf-strategy-design.md   # this file
docs/superpowers/plans/
  2026-04-24-ma-ribbon-ema21-phase1-plan.md                # written next, by writing-plans
```

This directory is **independent** of:
- `server/strategy/` (production manual-line strategy — different system)
- `ma_ribbon_backtest.py` / `ma_ribbon_screener.py` (legacy CLI scripts — left untouched)
- `server/ma_ribbon_service.py` (a different multi-TF score service — unrelated)

---

## 7. Acceptance checklist (cross-phase)

Per `PRINCIPLES.md` §P15: completion requires enumerated test results, not "it worked once."

For every phase delivery, the report must include:

- [ ] All listed tests pass (CI green).
- [ ] No-lookahead test passes on every commit.
- [ ] IS and OOS metrics shown side-by-side.
- [ ] Acceptance-gate metric is computed and pass/fail is stated explicitly.
- [ ] Failed cases are NOT hidden (per `CLAUDE.md` "do not lie"): if 3 of 50 symbols had Sharpe < 0, list them.
- [ ] Fees and slippage are applied (verified by a test that compares pre-fee and post-fee).
- [ ] No use of `try / except: pass` (`PRINCIPLES.md` §P10).
- [ ] User has reviewed the report and approved advancing to next phase.

Banned phrases per `CLAUDE.md` until a committed test log proves them: "能用了", "修好了", "done", "fixed", "能走通", "已经做完", "工作正常", "一切正常".

---

## 8. Open questions deferred to later phases

Listed here so they aren't forgotten. Not in scope for the next plan.

1. **Cross-symbol allocation ("田忌赛马")** — defer to a Phase 6 spec only if Phase 5 yields a profitable per-symbol strategy.
2. **Short-side mirror** — defer; build long-only first and prove it works.
3. **Walk-forward refit** — defer; static IS/OOS split is the MVP.
4. **Live execution wiring** — explicitly out of scope per spec.
5. **Funding rate cost on perp** — initially ignored; revisit if Phase 5 OOS Sharpe is marginal.
6. **Per-symbol risk-budget tuning** — Phase 5 may tune per symbol.
7. **Anti-correlation portfolio constraints** — defer.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wyckoff detector is fuzzy and may not generalize | Mandatory side-by-side baseline-TP comparison in §2.6.4. Wyckoff is killed if it loses on OOS. |
| Survivorship bias from "Top-50 today" | At minimum, document the bias in the report. Better: use point-in-time top-N if Bitget historical listings are available. |
| Fee underestimation (taker vs maker, funding) | Default to taker fee (worst case). Add 1bp slippage. Fees test confirms application. |
| Multi-TF alignment bug → look-ahead | `test_multi_tf_alignment.py` is a hard gate. Run every commit. |
| Pivot detection lag in Stage D causes systematic late TP | Acknowledged. Reports include a "TP delay" metric (bars from peak to exit). User can adjust `pivot_confirm_bars`. |
| Overfitting via 12+ parameter grid × 50 symbols | OOS gate per §5. If OOS << IS, parameter rejected. |
| Existing data pipeline insufficient for bulk | `data_loader.py` adapter; cache to local CSV per `(symbol, TF)`. |

---

## 10. Next steps after user review of this spec

1. User reviews this spec file. Requests edits.
2. Spec edits applied; user re-reviews.
3. User approves spec.
4. Invoke `superpowers:writing-plans` with this spec as input.
5. Output: `docs/superpowers/plans/2026-04-24-ma-ribbon-ema21-phase1-plan.md` — a step-by-step Phase 1 implementation plan.
6. User reviews Phase 1 plan.
7. User approves Phase 1 plan.
8. Implementation begins via `superpowers:executing-plans` (or equivalent).

**No code is written until step 7.**

---

## 11. Final response template (cross-phase, when each phase ships)

When a phase delivers, the agent must report:

- Files created or modified (with paths and line counts).
- How to run the phase's backtest (exact command).
- Where the report lands (path).
- Configuration knobs (which JSON keys control what).
- Where the test log lives.
- Assumptions made that the user should re-verify.
- What the next phase needs from the user before starting.

The agent must NOT claim:

- "the strategy works" without OOS Sharpe ≥ acceptance gate.
- "fixed" / "done" without committed test logs.
- a winning Wyckoff TP without baseline-TP side-by-side numbers.

---

## Appendix A — Mapping to user's spoken requirements

This is a sanity-check that the user's verbal-spec lines are covered.

| User said | Section |
|---|---|
| "5m fires first, only LV1, then wait for next 15m" | §2.3 Strategy Y |
| "if higher TF already in ribbon when 5m fires, fire all together" | §2.3 Strategy X (alternative, AB-tested) |
| "trailing stop = current EMA21 buffer" | §2.4 |
| "buffer per TF: 5m=1%, 15m=4%, 1h=7%, 4h=10%" | §2.4 defaults, §2.5 risk budget |
| "5m stops out, can re-enter when ribbon re-forms" | §2.7 |
| "TP = MA fan + platform + MA cross + RSI divergence" | §2.6.2 Wyckoff (Stages A–E) |
| "TP per TF — 5m TP closes 5m layer, 4h TP closes all" | §2.6.3 |
| "after TP exit, slowly re-activate next wave" | §2.7 |
| "test on 40–50 coins" | §3.6 universe field |
| "make Wyckoff thresholds configurable, tune iteratively" | §2.6.2 (all defaults are starting points), §4 phasing |
| "I gave you max permission, write the full spec" | This document |

---

## Appendix B — Why we are NOT touching existing code

- `ma_ribbon_backtest.py` and `ma_ribbon_screener.py` (project root) are CLI scripts; they don't share the strategy semantics being specified here (no scale-in, no EMA21 buffer trailing, no Wyckoff TP).
- `server/ma_ribbon_service.py` is a multi-TF score endpoint (0–10 score by TF weights); it's read-only metadata, not a backtester.
- `server/strategy/` is the manual-line / conditional-order production system; this strategy is **standalone** and must not interact with manual-line orders, conditionals, or the `OrderConfig` types.
- This spec's `backtests/ma_ribbon_ema21/` is a fresh directory. No imports from `server/`. Data loading is the only external dependency, and even that gets a thin adapter if needed.

End of spec.
