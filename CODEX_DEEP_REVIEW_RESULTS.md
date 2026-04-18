# CODEX_DEEP_REVIEW_RESULTS

Scope: reviewed `CODEX_DEEP_REVIEW.md`, `PRINCIPLES.md`, `TRENDLINE_TRADING_RULES.md`, `CLAUDE.md`, and the referenced `CODEX_REVIEW_INSTRUCTIONS.md`. No real Bitget order was placed; Bitget-facing verification used adapter fakes plus request-body assertions.

## Findings

## Finding 1
- Severity: P0
- File: `server/execution/live_adapter.py:178`
- Issue: Trendline entry plans were submitted as market-trigger plans, conflicting with `TRENDLINE_TRADING_RULES.md` section 11: entry must be a pre-placed limit order.
- Impact: Live behavior chased a triggered market fill instead of passively waiting at the line+buffer coordinate; backtest/live entry semantics diverged.
- Fix: `submit_live_plan_entry()` now sends Bitget `orderType: "limit"` and `executePrice` equal to the normalized trigger/limit price (`server/execution/live_adapter.py:218-220`), while still attaching preset SL/TP (`server/execution/live_adapter.py:228-233`).
- Verification: `tests/execution/test_live_adapter.py:137` asserts limit trigger + preset SL/TP; `scripts/verify_trendline_data_flow.py` prints `plan_order.order_type = "limit"`, entry `100.2`, SL `100.0`, TP `103.206`.

## Finding 2
- Severity: P0
- File: `server/execution/live_adapter.py:297`
- Issue: SL refresh could cancel `profit_loss` TP orders because previous identification used only side/tradeSide and did not compare trigger price to entry.
- Impact: Moving a trailing SL could silently remove take-profit protection.
- Fix: `update_position_sl_tp()` now fetches the position entry price (`server/execution/live_adapter.py:444`), cancels `pos_loss`, and only cancels `profit_loss` entries that are SL by side plus trigger direction vs entry (`server/execution/live_adapter.py:371-373`).
- Verification: `tests/execution/test_live_adapter.py:153` proves long TP is not cancelled while SL is replaced; returned `cancelled_order_ids` excludes the TP.

## Finding 3
- Severity: P0
- File: `server/strategy/trendline_order_manager.py:344`
- Issue: Moved orders were not using the per-timeframe risk value consistently, so replacement quantity could differ from the original line risk.
- Impact: A moved 1h order could resize from fallback risk instead of 1h risk, changing dollar risk without user intent.
- Fix: Both new and moved orders now read `cfg["tf_risk"][tf]` (`server/strategy/trendline_order_manager.py:224`, `server/strategy/trendline_order_manager.py:371`) and route sizing through `_qty_for_risk()` with max notional cap (`server/strategy/trendline_order_manager.py:96-111`).
- Verification: `tests/strategy/test_trendline_order_manager.py:90` asserts moved 1h quantity is `75.0`, matching equity `1000 * risk 0.015 / stop_distance 0.2`.

## Finding 4
- Severity: P1
- File: `server/strategy/mar_bb_runner.py:1159`
- Issue: Existing trendline plan orders only moved when the current scan produced new trendline signals.
- Impact: A valid active order could miss bar-boundary movement if no new signal was emitted in that scan cycle.
- Fix: The runner now calls `update_trendline_orders()` whenever the trendline strategy is enabled, even with an empty signal list (`server/strategy/mar_bb_runner.py:1159-1179`). The order manager moves existing orders based on elapsed TF bars from `last_updated_ts` (`server/strategy/trendline_order_manager.py:326`).
- Verification: `tests/strategy/test_trendline_order_manager.py:90` exercises an existing order move with no new order placement.

## Finding 5
- Severity: P1
- File: `server/strategy/trendline_order_manager.py:316`
- Issue: Active plan orders were not reliably marked filled when the exchange already had a held position for the symbol.
- Impact: Restart or fill-sync could duplicate or continue moving an already-filled plan order.
- Fix: The manager receives `held_symbols`, marks matching active orders as `filled`, and preserves them in state without replacement (`server/strategy/trendline_order_manager.py:316-319`). Fill sync also saves `filled` status (`server/strategy/mar_bb_runner.py:1243-1248`).
- Verification: `tests/strategy/test_trendline_order_manager.py:126`; `scripts/verify_trendline_state_recovery.py` prints `active_status_after_restart_sync = "filled"` and `duplicate_orders_submitted = 0`.

## Finding 6
- Severity: P1
- File: `server/strategy/mar_bb_runner.py:311`
- Issue: Restart registration could reset `last_sl_set` to zero and use the plan creation timestamp rather than the actual position open timestamp when available.
- Impact: The first trailing pass after restart could think no SL existed and compute bars-since from the wrong clock.
- Fix: `register_trendline_params()` accepts `last_sl_set` (`server/strategy/mar_bb_runner.py:314-324`), fill sync passes the active order SL (`server/strategy/mar_bb_runner.py:1223`), and `_position_open_ts()` extracts exchange open time (`server/strategy/mar_bb_runner.py:329`).
- Verification: `tests/strategy/test_mar_bb_trailing.py:60`; `scripts/verify_trendline_state_recovery.py` prints restored `last_sl_set = 100.0`.

## Finding 7
- Severity: P2
- File: `server/execution/live_adapter.py:679`
- Issue: Price normalization treated `pricePlace` ambiguously and did not prefer raw Bitget tick metadata when present.
- Impact: Some contracts could round to the wrong tick and be rejected by Bitget precision checks.
- Fix: `_normalize_price()` now prefers `tickSz`, then raw `pricePlace` + `priceEndStep`, then legacy decimal `pricePlace` fallback (`server/execution/live_adapter.py:679-706`).
- Verification: `tests/execution/test_limit_order.py:33`; `scripts/verify_trendline_precision_units.py` prints `raw_pricePlace_3_endStep_5 = "50.12"`.

## Finding 8
- Severity: P2
- File: `server/strategy/trendline_order_manager.py:49`
- Issue: Active order load/save and some runner logging paths swallowed exceptions too quietly.
- Impact: State desync could be invisible, violating `PRINCIPLES.md` P10/P11.
- Fix: `_load_active()` and runner event/log paths now print the error and traceback (`server/strategy/trendline_order_manager.py:61-62`, `server/strategy/mar_bb_runner.py:1113`, `server/strategy/mar_bb_runner.py:1134`, `server/strategy/mar_bb_runner.py:1150`).
- Verification: Static evidence plus compile pass; no silent `except: pass` remains in the touched critical paths.

## Checklist Results

| Area | Item | Result | Evidence |
|---|---|---:|---|
| trendline_order_manager | `update_trendline_orders(new_signals, current_bar_index, cfg)` signature | PASS | `server/strategy/trendline_order_manager.py:139-143` |
| trendline_order_manager | cfg fields enumerated | PASS | Uses `equity_usd`, `risk_pct`, `leverage`, `rr`, `prices`, `mode`, `max_position_pct`, `held_symbols`, `tf_risk`, `tf_buffer`, `buffer_pct`: `server/strategy/trendline_order_manager.py:155-167`, `206`, `224`, `344`, `371` |
| trendline_order_manager | `buffer_pct` percent vs fraction | PASS | `_buffer_fraction_for_tf()` treats config values as percent and divides by 100: `server/strategy/trendline_order_manager.py:89-93`; script output 1h `0.20 -> 0.002` |
| trendline_order_manager | `tf_buffer` format | PASS | Runner sends percent map `{"5m":0.05,"15m":0.10,"1h":0.20,"4h":0.30}` at `server/strategy/mar_bb_runner.py:1171`; P2 script confirms fractions |
| trendline_order_manager | support long entry above line | PASS | `limit_px = proj * (1 + buffer_pct)`: `server/strategy/trendline_order_manager.py:209` |
| trendline_order_manager | resistance short entry below line | PASS | `limit_px = proj * (1 - buffer_pct)`: `server/strategy/trendline_order_manager.py:214` |
| trendline_order_manager | SL equals line | PASS | `stop_px = proj`: `server/strategy/trendline_order_manager.py:210`, `215` |
| trendline_order_manager | TP long/short formula | PASS | Long `limit_px * (1 + buffer_pct * rr)`, short `limit_px * (1 - buffer_pct * rr)`: `server/strategy/trendline_order_manager.py:211`, `216` |
| trendline_order_manager | risk USD from per-TF risk | PASS | `per_tf_risk = cfg["tf_risk"][tf]`: `server/strategy/trendline_order_manager.py:224`, `371` |
| trendline_order_manager | quantity = risk / stop distance | PASS | `_qty_for_risk()` uses `risk_usd / stop_distance`: `server/strategy/trendline_order_manager.py:96-109`; 1h script qty `75.0` |
| trendline_order_manager | minimum distance filter | PASS | Replaced ambiguous fixed `dist < 0.003` behavior with passive trigger guard `_would_trigger_immediately()`: `server/strategy/trendline_order_manager.py:120-126`, `220-221` |
| trendline_order_manager | bar-boundary move window | PASS | Replaced fragile 90s wall-clock boundary with full TF bars elapsed from `last_updated_ts`: `server/strategy/trendline_order_manager.py:326` |
| live_adapter | entry order type | PASS | Fixed to Bitget limit plan: `server/execution/live_adapter.py:218-220`; old review text saying market is superseded by `TRENDLINE_TRADING_RULES.md` section 11 |
| live_adapter | trigger price precision | PASS | Normalized trigger at `server/execution/live_adapter.py:197`; precision helper at `679-706` |
| live_adapter | preset SL equals line | PASS | `stopLossTriggerPrice` from normalized `stop_price`: `server/execution/live_adapter.py:224-228`; data-flow script SL `100.0` |
| live_adapter | preset TP equals target | PASS | `stopSurplusTriggerPrice` from normalized `tp_price`: `server/execution/live_adapter.py:231-233`; data-flow script TP `103.206` |
| live_adapter | triggerType mark price | PASS | `triggerType: "mark_price"` retained: `server/execution/live_adapter.py:217` |
| live_adapter | size min unit | PASS | `_normalize_size()` uses `minTradeNum`/`lotSz` and rejects below min: `server/execution/live_adapter.py:200-202`, `663-674` |
| live_adapter | cancel only SL, not TP | PASS | SL identification by entry-relative trigger: `server/execution/live_adapter.py:326-373`; test at `tests/execution/test_live_adapter.py:153` |
| live_adapter | place `pos_loss` with normalized SL | PASS | `server/execution/live_adapter.py:397-405`; test asserts normalized `101.23` |
| live_adapter | trailing does not pass TP | PASS | Runner calls `new_tp=None`: `server/strategy/mar_bb_runner.py:439`; test at `tests/strategy/test_mar_bb_trailing.py:21` |
| trailing | `current_sl` source | PASS(local) | Restored from active order and updated only after adapter success: `server/strategy/mar_bb_runner.py:398`, `453`; no live Bitget read was run |
| trailing | TF per position | PASS | `tf = params.get("tf", "1h")`: `server/strategy/mar_bb_runner.py:396`; restored from active order at `1223` |
| trailing | new-bar detection | PASS | `last_update_bar` skip for same bar: `server/strategy/mar_bb_runner.py:400-411`; test moves 3 distinct bars |
| trailing | projection formula | PASS | `entry_bar + bars_since`, then `slope * current_bar + intercept`: `server/strategy/mar_bb_runner.py:364-365` |
| trailing | never widen | PASS | Long only raises SL, short only lowers SL: `server/strategy/mar_bb_runner.py:419-424` |
| trailing | same-value threshold | PASS | `0.0001` relative threshold: `server/strategy/mar_bb_runner.py:427` |
| trailing | memory update after exchange success | PASS | `params["last_sl_set"] = placed_sl`, `last_update_bar = bars_since`: `server/strategy/mar_bb_runner.py:453-454` |
| register params | entry bar | PASS | Fill sync uses `ao.bar_count - 1`: `server/strategy/mar_bb_runner.py:1217` |
| register params | fill/open timestamp | PASS | `_position_open_ts(pos)` preferred over plan creation time: `server/strategy/mar_bb_runner.py:329`, `1214` |
| register params | TP and TF propagation | PASS | `tf=ao.timeframe`, `tp_price=ao.tp_price`: `server/strategy/mar_bb_runner.py:1221-1224` |
| fill detection | active JSON format | PASS | `ActiveLineOrder(**x)` load and dataclass save: `server/strategy/trendline_order_manager.py:54-69` |
| fill detection | held position detection | PASS | Position size from `total`/`available` via `_position_size`; held symbols collected at `server/strategy/mar_bb_runner.py:954-958` |
| fill detection | duplicate same-symbol orders | PASS | Active/held symbols are excluded for new placements: `server/strategy/trendline_order_manager.py:167-177`; held active order marked filled at `316-319` |
| PnL tracking | history-position API | PASS(local/static) | Close cleanup logs via `log_position_closed`: `server/strategy/mar_bb_runner.py:1287-1288`; no live close was available to verify Bitget fields |
| numeric | buffer unit end-to-end | PASS | P2 script prints 5m `0.0005`, 15m `0.001`, 1h `0.002`, 4h `0.003` |
| numeric | price precision end-to-end | PASS | Contract tick normalization tests and P2 script pass |
| numeric | quantity cap | PASS | `_qty_for_risk()` caps notional at equity * max_position_pct * leverage: `server/strategy/trendline_order_manager.py:111-116` |
| numeric | fee model | FAIL/documented | Backtest uses a fixed fee model; live sizing still does not pre-deduct variable Bitget fees. No safe code change made because real fee tier/account data is not available locally. |
| state | `_trendline_params` vs positions | PASS(local) | Fill/restart sync restores active held positions: `server/strategy/mar_bb_runner.py:1195-1248`; P3 script confirms |
| state | active orders vs plan orders | PASS(local) | Active orders move/preserve/fill status; no live Bitget pending-plan reconciliation was run |
| state | `last_sl_set` vs actual exchange SL | PASS(local) | Adapter returns placed normalized SL and runner stores it: `server/execution/live_adapter.py:439`, `server/strategy/mar_bb_runner.py:453`; actual live SL read not executed |
| state | API failure visibility | PASS | Critical failures print and preserve/reject state rather than silently passing: `server/strategy/trendline_order_manager.py:61-62`, `307`, `367`, `413` |
| edge | same symbol across multiple TF | PASS | One active/held symbol blocks additional new symbol-level line orders: `server/strategy/trendline_order_manager.py:167-177` |
| edge | fill during scan | PASS(local) | Held-symbol sync after order manager marks filled and registers params: `server/strategy/mar_bb_runner.py:1195-1248` |
| edge | SL and TP same bar | FAIL/inherent live difference | Backtest pessimistic intrabar order cannot be reproduced exactly with Bitget real-time preset SL/TP without exchange execution history. No deterministic local fix. |
| edge | horizontal line | PASS | SL remains line; P3 script uses slope `0.0` and keeps `last_sl_set=100.0` |
| edge | steep line | PASS(guarded) | Never-widen checks prevent moving SL in the wrong direction; no explicit "cross entry" cap exists |
| edge | equity below min | PASS | `_normalize_size()` rejects below min as `size_below_min_trade`: `server/execution/live_adapter.py:200-202` |
| edge | API rate limit | FAIL/not tested | No rate-limit simulation was added; current scan still can cover top_n * TFs. |
| backtest/live | entry method | PASS | Live now uses limit plan; data-flow script matches entry `100.2` and formula |
| backtest/live | SL check | PASS(with caveat) | Live uses preset SL; backtest intrabar pessimism remains intentionally different |
| backtest/live | SL movement | PASS | Runner trailing test moves 3 bars; order manager moves active pending orders by TF elapsed bars |
| backtest/live | TP check | PASS | Preset TP attached; trailing SL updates do not touch TP |
| backtest/live | sweep same bar | FAIL/inherent live difference | Exchange handles real-time order sequence; local bar sweep model is not a live guarantee |
| backtest/live | fees | FAIL/documented | Variable live fees not represented in sizing |
| backtest/live | per-bar dedup | PASS | Active symbol dedup prevents duplicate line orders; filled positions block replacements |
| backtest/live | per-TF buffer | PASS | Runner sends TF buffer map and manager converts percent to fraction |

## P0-P3 Implementation Tasks

| Task | Result | Evidence |
|---|---:|---|
| P0 data flow: scan -> signal -> plan -> fill -> SL set -> SL move | PASS | `scripts/verify_trendline_data_flow.py`; output entry `100.2`, SL `100.0`, TP `103.206`, qty `75.0`, SL move to `100.1` |
| P1 SL movement over consecutive bars | PASS | `tests/strategy/test_mar_bb_trailing.py:21`; three updates `101`, `102`, `103`, with `new_tp=None` |
| P2 precision/unit scan | PASS | `scripts/verify_trendline_precision_units.py`; percent-to-fraction and tick examples printed |
| P3 restart/state recovery | PASS(local) | `scripts/verify_trendline_state_recovery.py`; status `filled`, `duplicate_orders_submitted=0`, restored SL/TP |

## Verification Commands

```powershell
python -m compileall server\execution\live_adapter.py server\strategy\trendline_order_manager.py server\strategy\mar_bb_runner.py server\strategy\ml_trade_db.py server\strategy\drawing_learner.py
pytest -p no:cacheprovider tests/execution/test_live_adapter.py tests/execution/test_limit_order.py tests/strategy/test_trendline_order_manager.py tests/strategy/test_mar_bb_trailing.py
python scripts\verify_trendline_data_flow.py
python scripts\verify_trendline_precision_units.py
python scripts\verify_trendline_state_recovery.py
```

Result: targeted pytest passed `16 passed in 0.65s`. P0/P2/P3 scripts passed and printed the values summarized above.

Wide suite note: `pytest -p no:cacheprovider tests/execution tests/strategy` produced `154 passed, 7 failed`. The remaining failures are outside the changed trendline live-order path: paper router stream expectation, two backtest trade-count expectations, config default `min_rr_ratio`, strategy `/config` market precision response, and two trendline invalidation tests that generate no resistance line. These should be handled separately because they are not caused by the limit-order/trailing fixes above.
