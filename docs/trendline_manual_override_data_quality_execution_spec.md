# Trendline / Manual Override / Data Quality / Execution Spec v1

Updated: 2026-04-09

## 1. Purpose

This document defines the next product-quality implementation line for `crypto-analysis-`.

The current system is no longer blocked on "can the main chain run at all?".
It is now blocked on "is the product trustworthy and usable for an actual trader?".

This document does not replace `docs/strategy_spec.md`.
`docs/strategy_spec.md` remains the strategy-core source of truth.

This document freezes the next product phases that must be completed before the system can be considered a trader-grade terminal:

1. trendline / touch / invalidation quality,
2. manual drawing and human override,
3. full-history and data completeness,
4. execution center / GUI / loading reliability,
5. auto runtime for subaccounts.

Portfolio allocation is intentionally not part of the current mainline.
That can be deferred to a future phase because subaccounts can solve funding separation first.

## 2. Current Product Gaps

### 2.1 Detection Quality Gap

Current issues:

1. too many touch markers are rendered,
2. low-quality bar touches visually compete with true structural touches,
3. invalidations can appear as repeated `X` markers,
4. candidate lines and trader-usable lines are mixed together,
5. overlays still behave more like a debug console than a trading view.

### 2.2 Manual Control Gap

Current issues:

1. the user cannot manually draw a line,
2. the user cannot delete, lock, or extend a manual line,
3. the user cannot suppress or promote automatic lines from the chart workspace,
4. the system cannot capture human corrections as first-class objects.

### 2.3 Data Integrity Gap

Current issues:

1. the chart defaults to a bounded window instead of explicit full history,
2. long-horizon structural context can be silently truncated,
3. the UI does not clearly show whether the chart is in fast mode or full-history mode,
4. linear vs log display is not exposed as an explicit chart mode.

### 2.4 GUI / Execution Gap

Current issues:

1. Overview, Paper Execution, Live Bridge, and legacy sections still overlap conceptually,
2. the user cannot immediately tell which execution surface is the current truth source,
3. paper-first loading exists, but some surrounding UI still looks transitional,
4. the product still feels like an engineering workbench rather than a coherent trading terminal.

### 2.5 Auto Runtime Gap

Current issues:

1. live execution is still mainly reconcile / preview / submit / close,
2. there is no fully automated approved-intent scan loop,
3. there is no durable restart recovery for runtime state,
4. there is no subaccount-oriented runtime control surface.

## 3. Scope

The current mainline includes exactly these phases:

1. Phase A: Trendline / Touch / Invalidation Quality Correction
2. Phase B: Manual Drawing / Human Override
3. Phase C: Full History / Data Quality / Scale Modes
4. Phase D: Execution Center / GUI / Loading Reliability
5. Phase E: Auto Runtime for Subaccounts

The following is explicitly out of scope for this mainline:

1. portfolio allocator,
2. sleeve budgets,
3. multi-strategy capital attribution,
4. multi-user runtime,
5. persistence-backed portfolio analytics.

Those may appear later as a future phase or appendix.

## 4. Product Principles

### 4.1 Backend Truth

The backend remains the only truth source for:

1. pivots,
2. touch classes,
3. line states,
4. invalidations,
5. signals,
6. orders,
7. positions,
8. runtime state.

The frontend may filter or toggle views, but it must not re-score or reclassify strategy objects on its own.

### 4.2 Trader View Over Debug View

Default overlays must show only what a trader should reason about directly.

Debug detail may still exist, but it must:

1. live behind explicit toggles,
2. be visually lower-priority,
3. not dominate the default chart.

### 4.3 Human Override Is First-Class

Manual lines are not cosmetic annotations.
They are first-class objects that can:

1. coexist with automatic lines,
2. suppress or promote automatic results,
3. optionally feed the backend strategy layer,
4. become review data for later model improvement.

### 4.4 Data Completeness Must Be Explicit

The UI must always make it clear whether the user is looking at:

1. a fast bounded window,
2. full available history,
3. a truncated or partially recovered history set.

### 4.5 Runtime Must Be Default-Safe

Any automation added after this point must remain:

1. default-off,
2. gated,
3. restart-safe,
4. subaccount-friendly,
5. auditable.

## 5. Phase A: Trendline / Touch / Invalidation Quality

### 5.1 Goal

Make automatically detected lines and touch events structurally credible and visually clean enough for trader use.

### 5.2 Required Backend Classification

The backend must explicitly distinguish:

1. `anchor_pivot`
2. `confirming_touch`
3. `bar_touch`
4. `display_line`
5. `debug_line`
6. `primary_invalidation`
7. `debug_invalidation`

### 5.3 Touch Rules

#### Confirming Touch

A confirming touch:

1. must come from a confirmed pivot,
2. must satisfy tolerance and residual bounds,
3. must satisfy spacing rules,
4. must not remain valid once the line is structurally invalidated,
5. is allowed to affect line score and confirmation count.

#### Bar Touch

A bar touch:

1. may come from any bar,
2. may participate in armed / rejection / failed-breakout logic,
3. must not increment structural confirmation count,
4. must not flood the default chart.

### 5.4 Display Filtering

Default chart rendering must show:

1. only the highest-utility lines per side,
2. confirming touches,
3. a limited number of recent high-value bar touches tied to active lines,
4. one collapsed invalidation marker per display line lifecycle.

Raw candidates remain debug-only.

### 5.5 Acceptance Criteria

Phase A is complete only if:

1. repeated invalidation stacks are collapsed in default mode,
2. default chart no longer renders a noisy wall of touch markers,
3. primary lines are clearly separated from debug lines,
4. backend tests cover touch classification, invalidation collapse, and display ranking,
5. frontend still does not re-score lines itself.

## 6. Phase B: Manual Drawing / Human Override

### 6.1 Goal

Allow the trader to draw, manage, and optionally elevate manual lines into the same workspace as automatic lines.

### 6.2 Manual Line Object

Minimum fields:

```text
manual_line_id
symbol
timeframe
side
source = manual
t_start
t_end
price_start
price_end
extend_left
extend_right
locked
label
notes
comparison_status
override_mode
created_at
updated_at
```

### 6.3 Required Capabilities

1. enter draw mode,
2. create a line from two anchors,
3. select a manual line,
4. delete the selected line,
5. lock / unlock it,
6. extend left / right,
7. rename / label it,
8. compare it against nearby auto lines,
9. optionally mark it as backend strategy input.

### 6.4 Strategy Boundary

Drawing a manual line does not automatically change strategy behavior.
Manual influence must remain opt-in and backend-owned.

### 6.5 Acceptance Criteria

Phase B is complete only if:

1. manual lines can be created, edited, deleted, and persisted,
2. manual and automatic lines can be compared on the same chart,
3. manual override does not break automatic overlays,
4. the backend records whether a signal came from auto, manual, or hybrid input.

## 7. Phase C: Full History / Data Quality / Scale Modes

### 7.1 Goal

Make chart history completeness explicit and sufficient for long-horizon structure work.

### 7.2 Required Metadata

For each chart payload, expose:

1. `history_mode`
2. `loaded_bar_count`
3. `earliest_loaded_timestamp`
4. `latest_loaded_timestamp`
5. `is_full_history`
6. `is_truncated`
7. `truncation_reason`
8. `listing_start_timestamp` when known

### 7.3 Required Modes

The chart must support:

1. `fast_window`
2. `full_history`

The UI must clearly show which mode is active.

### 7.4 Scale Modes

The chart must expose:

1. `linear`
2. `log`

This is a presentation toggle first.
Backend strategy logic does not silently change because the chart view changed.

### 7.5 Acceptance Criteria

Phase C is complete only if:

1. the user can tell whether the chart is fast or full-history,
2. full-history mode loads all locally available / exchange-available history for the selected symbol-timeframe,
3. the chart exposes linear/log explicitly,
4. completeness metadata is visible in the UI.

## 8. Phase D: Execution Center / GUI / Loading Reliability

### 8.1 Goal

Reduce information duplication and make the product read as one coherent terminal instead of multiple half-overlapping control surfaces.

### 8.2 Execution Center Direction

The main UI story should be:

1. Overview: high-level account/runtime summary,
2. Paper/Demo Runtime: active orders, positions, killswitch, recent fills,
3. Live Bridge: reconcile / preview / submit / close,
4. Debug / Legacy: older agent and ops surfaces, hidden by default.

### 8.3 Loading Reliability

The UI must preserve:

1. paper-first execution rendering,
2. independent section failure,
3. stale-request guards,
4. fast-first chart loading,
5. degraded-but-usable behavior when heavy endpoints are slow.

### 8.4 Acceptance Criteria

Phase D is complete only if:

1. the main user can identify one primary execution path,
2. legacy/debug surfaces are visually secondary,
3. the product does not regress on paper-first rendering,
4. the chart and execution center still degrade gracefully under slow auxiliary endpoints.

## 9. Phase E: Auto Runtime for Subaccounts

### 9.1 Goal

Enable stable, default-off automatic runtime for subaccount-oriented instances without requiring a full portfolio allocator.

### 9.2 Runtime Model

Recommended operating model:

1. one subaccount = one runtime instance,
2. one runtime instance = one strategy flavor / whitelist / live config,
3. each runtime consumes approved intents only,
4. each runtime persists its own idempotency and recovery state.

### 9.3 Required Capabilities

1. automatic scan of approved intents,
2. runtime start / stop controls,
3. persisted idempotency state,
4. restart recovery,
5. reconcile-before-resume,
6. structured logs,
7. global and instance-level kill switch.

### 9.4 Acceptance Criteria

Phase E is complete only if:

1. a runtime can restart without re-submitting already handled intents,
2. runtime state can be recovered after restart,
3. reconcile gates resumption,
4. the runtime remains default-off and explicitly controlled.

## 10. Future Phase Appendix

The following are future-phase items and are not part of the current mainline:

1. portfolio allocator,
2. sleeve budgets,
3. strategy attribution dashboards,
4. unified multi-strategy account allocator,
5. multi-user orchestration.

## 11. Recommended Execution Order

1. Phase A: detection quality
2. Phase B: manual drawing / human override
3. Phase C: full history / data quality / scale modes
4. Phase D: execution center / GUI / loading reliability
5. Phase E: auto runtime for subaccounts

This order is deliberate:

1. do not scale automation on top of low-quality line recognition,
2. do not trust full-history analysis until history completeness is explicit,
3. do not automate runtime before the human can override structure,
4. do not treat GUI duplication as cosmetic when it reflects product ambiguity.
