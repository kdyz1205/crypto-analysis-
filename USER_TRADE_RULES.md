# User Trade Rules - Execution-Specific Semantics

This file captures the user's own trading semantics that are stricter
than generic TA language. Read this BEFORE changing any code that
computes entry / SL / TP from a manual line.

## 1. Buffer and stop are separate distances from the line

For a LONG setup:

- `buffer_pct` means the entry sits ABOVE the line by that percent
- `stop_pct` means the SL sits BELOW the line by that percent
- total entry-to-SL distance = `buffer_pct + stop_pct`

Example from the user's ZEC rule:

- line_at_now ~= 315.95
- buffer_pct = 0.41
- stop_pct = 0.04
- entry ~= `315.95 * 1.0041`
- stop ~= `315.95 * 0.9996`

Do NOT collapse buffer+stop into one field mentally.
Do NOT describe the setup as "0.45% buffer". It is:

- 0.41% line -> entry
- 0.04% line -> SL
- 0.45% total entry -> SL

## 2. The user reads the line at the current visual intersection

When the user says "intersect is around X", they mean the line's value
where it crosses the current chart vertical / current live bar.

Use:

- current line geometry
- current timeframe
- log projection
- current visual intersection semantics

Do NOT justify a mismatched order price by using an older snapshot if
the user is describing what they saw on screen at order time.

### 2a. Concrete time-reference rule — resolved 2026-04-23

The line's reference time for server-side placement is **the rightmost
candle's open_ts** on the user's chart — NOT wall-clock now, NOT
`floor(now / tf_seconds) * tf_seconds`. The chart visually places each
candle at its `time` property (= bar_open_ts), so the line crosses
today's candle at the pixel position corresponding to bar_open_ts. The
user's eye reads the line at that pixel.

**Bitget 1d anchor is UTC+8 midnight, not UTC 00:00.** For ZECUSDT 1d
the bar opens at UTC 16:00 (not UTC 00:00). Any "floor to UTC midnight"
math is wrong for 1d. Always use the actual last candle's `time` from
the OHLCV feed. Different symbols may use different anchors; trust the
data, don't assume.

Three bugs came from this oversight:

- v1 (exact click ms): drifts within the bar.
- v2 (floor-to-midnight bar-open snap): drifts 16h for Bitget 1d.
- v3 (wall-clock now): drifts 2pt when bar_open is hours behind now.
- **v4 (current bar's actual open_ts from OHLCV)**: matches visual to
  < 0.1 point.

Wiring:

- Frontend `trade_plan_modal` reads `chart.getCandles()` last item's
  `time`, passes it as `reference_ts` in the POST.
- Backend `api_place_line_order` uses `reference_ts` if provided, else
  wall-clock now fallback.
- Frontend modal preview uses the same `reference_ts` for the
  displayed `line @ 当前蜡烛 (log)` row, so preview and placed
  number agree.

### 2b. Everything in UTC, no hidden shift

As of 2026-04-23, the default chart timezone is `utc` (was `la`). All
time references — candle boundaries, chart X-axis labels, server-side
`now`, line projection `reference_ts`, % change baseline — share ONE
time base. Mixing UTC (% change) with LA (chart axis) had been
masking the line-projection gap because UTC 00:00 looks like a
reasonable bar boundary while Bitget's actual boundary is UTC 16:00
(= UTC+8 midnight).

## 3. Active orders protect the rationale line

If a line has an active order:

- it may move
- it may NOT be deleted

Reason: the line is part of the order rationale and post-mortem context.

Do NOT reintroduce any "delete line also cancels everything" behavior
unless the user explicitly asks to change that product rule.

## 4. Setup loss / reset is unacceptable

The user's saved setups are part of their trading system. They must not:

- silently reset to default
- disappear because of local-only storage
- switch active selection without visible explanation

Any fallback to default must be explicit and observable.

## 5. Pre-placement preview is MANDATORY

Added 2026-04-23. Every place-line-order modal MUST show the exact
server-side values BEFORE the user clicks confirm:

- `line @ 当前蜡烛 (log)` — what the server computed as the line's
  value at the reference_ts
- `入场价` — entry = line × (1 ± buffer%)
- `止损价` — SL = line × (1 ∓ stop%), annotated with distance to entry
- `止盈价` — TP based on RR target

Preview must use the EXACT SAME reference_ts and formula as the
submission path. If preview shows 317.03 entry and placement writes
319.02 to Bitget, that is a lie and is unacceptable.

## 6. Line editing — gestures the user expects

Recorded 2026-04-24 after the user was forced to double-click each
anchor to relocate a line.

### 6a. Click-and-drag the MIDDLE of a line to move the whole line

Gesture: user clicks on the line body (not an endpoint), holds mouse,
drags to a new position; on release both anchors shift by the same
(dx, dy), preserving slope and length.

Today's workaround is to drag each anchor separately, which is
tedious and tends to produce degenerate lines (e.g. ZEC's zero-span
`support-1775750400-1775750400` line that crashed the projection with
`span = 0`).

### 6b. Duplicate a line next to itself

Gesture / affordance: a "复制" button or right-click menu item on a
selected line. Creates a NEW `manual_line_id` with the same slope,
anchored slightly below (or above, user preference) the original so
the duplicate is immediately visible and draggable. This lets the
user seed a parallel channel or a parallel level in one click.

Neither 6a nor 6b is implemented yet — both are OPEN FEATURE REQUESTS.
When building, respect sections 1–5 (buffer+stop geometry, reference
time, delete-guard, setup persistence, preview accuracy).

## 7. Degenerate line protection

Today the backend `_project_manual_line_price` silently returns
`price_start` when `span <= 0`. The frontend renders a zero-span line
as a dot or degenerate segment that can be "selected" but projects
nonsense.

Rule: when a user creates or edits a line such that `t_start == t_end`
OR `price_start == price_end` on a slope, the system must either:

- reject the save with a clear error ("拖动端点形成零时长的线"), or
- auto-correct to a meaningful minimum span.

Silently accepting the line and then silently returning `price_start`
for projections is NOT acceptable — it lies to the user about what
the trade plan will use.
