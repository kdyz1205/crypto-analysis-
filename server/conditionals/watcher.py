"""Background watcher that polls live price vs. each pending conditional order.

Runs one asyncio task that cycles through all pending conditionals and, for
each one, decides whether it's time to re-poll (based on the conditional's
own poll_seconds). Stays lightweight even with dozens of pending orders.

Triggers are delivered via:
  - Telegram subscriber (reuses existing subscribers/telegram.py)
  - Optional: exchange submission via existing paper/live execution engines

IMPORTANT: this is a VIRTUAL order watcher. It does NOT place limit orders
on the exchange in advance. It only submits when the line is actually
touched, which avoids cancel/replace storms as the line's slope moves the
trigger price over time.
"""
from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any

from .store import ConditionalOrderStore, now_ts
from .types import ConditionalEvent, ConditionalOrder


# ─────────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────────
_task: asyncio.Task | None = None
_running: bool = False
_store = ConditionalOrderStore()

# How often the main loop checks for any pending work. Individual conditionals
# decide when to actually re-poll based on their own poll_seconds. We tick at
# 1s so the per-cond minimum effective poll = 1s.
_LOOP_TICK_SECONDS = 1

# How often we re-plan a Bitget plan order to track a sloped line.
# Tied to the line's timeframe — one bar = one replan.
_TF_REPLAN_SECONDS = {
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "2h":  7200,
    "4h":  14400,
    "6h":  21600,
    "12h": 43200,
    "1d":  86400,
    "1w":  604800,
}
# Only replan when the new line price differs from current trigger by at
# least this many %. Avoids burning cancel+replace cycles for tiny moves.
_REPLAN_DRIFT_PCT = 0.05

# Reconcile against Bitget every N ticks. Marks any local triggered cond
# whose exchange_order_id is no longer on Bitget as cancelled, so the UI
# stops showing zombies and the watcher stops trying to replan ghosts.
_RECONCILE_EVERY_TICKS = 60   # 60s with 1s tick — halved to reduce
                               # Bitget API pressure. User 2026-04-22:
                               # symbol/TF switching was 5-15s because
                               # Bitget was throttling us. Foreground
                               # requests (OHLCV, place-line-order) need
                               # priority over our periodic audit.
_reconcile_counter = 0


def _bar_open_ts(ts: int, timeframe: str) -> int:
    interval = _TF_REPLAN_SECONDS.get(timeframe, 900)
    return int(ts // interval * interval)

# Outcome backfill: scan pending trade snapshots and try to fill in
# exit_price / PnL / MAE / MFE from 1m K-line history. Runs much less
# often than reconcile because it loads 1m data per symbol.
_OUTCOME_BACKFILL_EVERY_TICKS = 300   # 5 minutes with 1s tick
_outcome_backfill_counter = 0

# Set of cond IDs flagged for immediate replan (e.g. user dragged a
# line's anchor). The watcher checks this set on every tick and clears
# it as entries are processed. See `force_replan(conditional_id)`.
_force_replan_set: set[str] = set()

# Short-lived cache of Bitget pending-order IDs so many conds in one
# tick share a single fetch. TTL chosen small enough that a freshly-
# cancelled plan won't get a stale "still present" result for long.
_PENDING_OIDS_CACHE_TTL = 3.0   # seconds
_pending_oids_cache: dict[str, tuple[float, set[str]]] = {}

async def _get_cached_pending_oids(mode: str) -> set[str]:
    """Return set of Bitget pending oids for this mode. 3s TTL cache.

    CRITICAL: on total fetch failure (both regular+plan endpoints raised),
    we RAISE instead of caching an empty set. Code-reviewer 2026-04-21:
    caching-on-fail poisoned every replan check for 3s on any transient
    Bitget outage — watcher silently skipped replans thinking "our plan
    is gone". Caller at _maybe_replan already has except-fallthrough
    that does the right thing on raise (attempt replan anyway, safer
    than stalling the whole line-tracking system).
    """
    import time as _t
    entry = _pending_oids_cache.get(mode)
    if entry and _t.time() - entry[0] < _PENDING_OIDS_CACHE_TTL:
        return entry[1]
    from server.execution.live_adapter import LiveExecutionAdapter
    adapter = LiveExecutionAdapter()
    oids: set[str] = set()
    err_regular: Exception | None = None
    err_plan: Exception | None = None
    ok_regular = False
    ok_plan = False
    try:
        for row in await adapter.get_pending_orders(mode):  # type: ignore
            oid = str(row.get("orderId") or row.get("order_id") or "")
            if oid:
                oids.add(oid)
        ok_regular = True
    except Exception as e:
        err_regular = e
    try:
        for row in await adapter.get_pending_plan_orders(mode, plan_type="normal_plan"):  # type: ignore
            oid = str(row.get("orderId") or row.get("order_id") or "")
            if oid:
                oids.add(oid)
        ok_plan = True
    except Exception as e:
        err_plan = e
    if not (ok_regular or ok_plan):
        # Total failure — do NOT cache; let caller fall through to attempt
        # replan. Bitget outage will also cause replan itself to fail, but
        # we won't silently skip EVERY cond for 3 seconds.
        raise RuntimeError(
            f"pending-oids fetch failed for mode={mode}: "
            f"regular={err_regular!r} plan={err_plan!r}"
        )
    _pending_oids_cache[mode] = (_t.time(), oids)
    return oids


def _invalidate_pending_oids_cache(mode: str | None = None) -> None:
    """Drop the cache so the next check refetches fresh."""
    if mode is None:
        _pending_oids_cache.clear()
    else:
        _pending_oids_cache.pop(mode, None)


def _add_pending_oid_to_cache(mode: str, oid: str) -> None:
    """Eagerly add a just-placed oid to the cache so the next check
    sees it as pending.

    CRITICAL fix 2026-04-21 per reviewer: if cache is empty/expired,
    SEED a fresh cache with just this oid (short-lived, will be
    replaced by the next real refetch). Old behavior dropped the oid
    silently, which defeated the purpose — _invalidate_pending_oids_
    cache clears the dict, then _add was a no-op, and a subsequent
    tick could fetch Bitget's eventually-consistent list BEFORE the
    new plan was visible -> false "my plan is gone" early-return.
    """
    if not oid:
        return
    import time as _t
    entry = _pending_oids_cache.get(mode)
    if entry and _t.time() - entry[0] < _PENDING_OIDS_CACHE_TTL:
        entry[1].add(str(oid))
        return
    # Empty or expired: seed a fresh single-oid cache. Other pending
    # oids on this mode will be picked up on the next real refetch
    # when this TTL expires.
    _pending_oids_cache[mode] = (_t.time(), {str(oid)})


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def start_watcher() -> None:
    """Start the background watcher task (idempotent)."""
    global _task, _running
    if _task is not None and not _task.done():
        return
    _running = True
    _task = asyncio.create_task(_loop(), name="conditional_watcher")
    print("[conditional_watcher] started", flush=True)


def stop_watcher() -> None:
    """Stop the background watcher task."""
    global _running
    _running = False
    if _task and not _task.done():
        _task.cancel()
    print("[conditional_watcher] stopped", flush=True)


def force_replan_line(manual_line_id: str) -> int:
    """Mark all triggered conds attached to this line for immediate replan.

    Called by PATCH /api/drawings/{id} when the user drags an anchor.
    The next watcher tick will pick up the flag and replan without
    waiting for the usual bar-interval gate. Returns count flagged.

    IMPORTANT: this MUST NOT mutate the conditional's stored line
    geometry. Those fields are the approval-time snapshot / trade
    rationale. Replan should read the CURRENT drawing separately.
    """
    global _force_replan_set
    n = 0
    try:
        for cond in _store.list_all(status="triggered", manual_line_id=manual_line_id):
            _force_replan_set.add(cond.conditional_id)
            n += 1
    except Exception as e:
        print(f"[force_replan] {manual_line_id}: {e}", flush=True)
    return n


def _project_line_geometry(
    *,
    t_start: int,
    t_end: int,
    price_start: float,
    price_end: float,
    extend_left: bool,
    extend_right: bool,
    ts: int,
) -> float:
    import math

    span = int(t_end) - int(t_start)
    if span <= 0:
        return float(price_start)
    if ts <= int(t_start) and not bool(extend_left):
        return float(price_start)
    if ts >= int(t_end) and not bool(extend_right):
        return float(price_end)
    p_start = float(price_start)
    p_end = float(price_end)
    if p_start <= 0 or p_end <= 0:
        slope_per_sec = (p_end - p_start) / span
        return p_start + slope_per_sec * (ts - int(t_start))
    ratio = (ts - int(t_start)) / span
    return math.exp(math.log(p_start) + ratio * (math.log(p_end) - math.log(p_start)))


_BITGET_1D_ANCHOR_SEC = 16 * 3600  # Bitget 1d bars open at UTC 16:00 (UTC+8 midnight)
_TF_SECONDS_FOR_ANCHOR = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
}


def _current_bar_open_ts(tf: str, now_i: int) -> int:
    """Return the open_ts of the currently-live bar for the given timeframe.

    Mirrors server/routers/conditionals.py so place-line + replan use the
    same reference point. See TA_BASICS.md Section 2a and USER_TRADE_RULES.md
    Section 2a. Bitget 1d anchored at UTC+8 midnight = UTC 16:00; intraday
    bars align to UTC hour/minute.
    """
    if tf == "1d":
        return ((now_i - _BITGET_1D_ANCHOR_SEC) // 86400) * 86400 + _BITGET_1D_ANCHOR_SEC
    if tf == "1w":
        return ((now_i - _BITGET_1D_ANCHOR_SEC) // (7 * 86400)) * (7 * 86400) + _BITGET_1D_ANCHOR_SEC
    sec = _TF_SECONDS_FOR_ANCHOR.get(tf)
    if sec:
        return (now_i // sec) * sec
    return now_i


def _line_price_for_replan(cond: ConditionalOrder, ts: int) -> float:
    """Project the ACTIVE line geometry for replan decisions.

    Priority:
      1. current drawing store row (what the user is editing now)
      2. conditional snapshot (what the user originally approved)

    The projection is anchored to the CURRENT LIVE BAR's open_ts (= the
    rightmost candle's time on the chart). This keeps replan's "line_now"
    in lock-step with what the user reads visually AND with what
    place-line-order used when the plan was first placed, so there's no
    drift between place and re-place. (2026-04-24 ZEC: replan was pulling
    wall-clock now and drifting the trigger away from the user's original
    317.03 placement toward 319.41 every reconcile tick.)
    """
    anchor_ts = _current_bar_open_ts(cond.timeframe, int(ts))
    try:
        from server.drawings.store import ManualTrendlineStore

        drawing = ManualTrendlineStore().get(cond.manual_line_id)
        if drawing is not None:
            return _project_line_geometry(
                t_start=int(drawing.t_start),
                t_end=int(drawing.t_end),
                price_start=float(drawing.price_start),
                price_end=float(drawing.price_end),
                extend_left=bool(getattr(drawing, "extend_left", False)),
                extend_right=bool(getattr(drawing, "extend_right", True)),
                ts=anchor_ts,
            )
    except Exception as exc:
        print(f"[watcher] current drawing fetch failed {cond.manual_line_id}: {exc}", flush=True)
    return cond.line_price_at(anchor_ts)


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
async def _loop() -> None:
    print("[conditional_watcher] main loop entering", flush=True)
    try:
        while _running:
            try:
                await _tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[conditional_watcher] tick error: {e}", flush=True)
                traceback.print_exc()
            await asyncio.sleep(_LOOP_TICK_SECONDS)
    except asyncio.CancelledError:
        print("[conditional_watcher] cancelled, exiting cleanly", flush=True)


async def _tick() -> None:
    """One pass through all conditionals.

    - status='pending': legacy bounce/break trigger watcher
    - status='triggered' WITH exchange_order_id: live Bitget plan order
      whose trigger price needs to be RE-PLANNED on every bar of the
      line's timeframe (cancel + place new) to track sloped lines.
    """
    global _reconcile_counter, _outcome_backfill_counter
    now = now_ts()

    # Periodic reconciliation: drop conds whose Bitget order is gone
    _reconcile_counter += 1
    if _reconcile_counter >= _RECONCILE_EVERY_TICKS:
        _reconcile_counter = 0
        try:
            await _reconcile_against_bitget()
        except Exception as e:
            print(f"[watcher] reconcile err: {e}", flush=True)

    # Periodic outcome backfill: walk pending snapshots and fill in
    # exit/PnL/MAE/MFE from 1m K-line history. Best-effort, non-blocking.
    _outcome_backfill_counter += 1
    if _outcome_backfill_counter >= _OUTCOME_BACKFILL_EVERY_TICKS:
        _outcome_backfill_counter = 0
        try:
            from .snapshots import backfill_pending_outcomes
            n = await backfill_pending_outcomes()
            if n > 0:
                print(f"[watcher] backfilled {n} outcome record(s)", flush=True)
        except Exception as e:
            print(f"[watcher] backfill err: {e}", flush=True)

    # Sloped-line replan path
    triggered = _store.list_all(status="triggered")
    for cond in triggered:
        if not cond.exchange_order_id:
            continue
        try:
            await _maybe_replan(cond, now)
        except Exception as e:
            print(f"[watcher] replan err: {e}", flush=True)

    pending = _store.list_all(status="pending")
    if not pending:
        return
    for cond in pending:
        # Time-based expiry
        if cond.trigger.max_age_seconds > 0:
            age = now - cond.created_at
            if age > cond.trigger.max_age_seconds:
                _expire(cond, f"age {age}s > max_age {cond.trigger.max_age_seconds}s")
                continue

        # Respect per-order poll rate
        last_poll = cond.last_poll_ts or 0
        if now - last_poll < cond.trigger.poll_seconds:
            continue

        try:
            await _poll_one(cond, now)
        except Exception as e:
            _append_event(cond, ConditionalEvent(
                ts=now, kind="poll",
                message=f"poll error: {e}",
            ))


async def _poll_one(cond: ConditionalOrder, now: int) -> None:
    """Fetch current price + ATR, update conditional state, maybe trigger."""
    market_price = await _fetch_market_price(cond.symbol)
    if market_price is None:
        _append_event(cond, ConditionalEvent(
            ts=now, kind="poll", message="market price unavailable",
        ))
        # Update last_poll_ts so we don't hammer the API
        _update_last_poll(cond, now, market_price=None)
        return

    # ATR is unreliable for short timeframes / illiquid pairs and was
    # producing tiny values that made even small distances look like 10+
    # ATRs, triggering auto-cancel. Use a price-percent floor as the real
    # ATR proxy: 0.3% of price, never less.
    atr_raw = await _fetch_current_atr(cond.symbol, cond.timeframe) or 0.0
    atr_floor = market_price * 0.003     # 0.3% minimum
    atr = max(atr_raw, atr_floor)

    line_price = cond.line_price_at(now)
    distance = abs(market_price - line_price)
    distance_atr = distance / atr if atr > 0 else 0.0
    distance_pct = (distance / market_price * 100) if market_price > 0 else 0.0

    tolerance = cond.trigger.tolerance_atr * atr
    break_thresh = cond.trigger.break_threshold_atr * atr

    # Distinguish "price broke through line" vs "price touched line"
    # For support: broken = close BELOW line by break_thresh
    # For resistance: broken = close ABOVE line by break_thresh
    broken = False
    if cond.side == "support" and (line_price - market_price) > break_thresh:
        broken = True
    elif cond.side == "resistance" and (market_price - line_price) > break_thresh:
        broken = True

    touched = distance <= tolerance

    event_msg = (
        f"market={market_price:.4f} line={line_price:.4f} "
        f"dist_atr={distance_atr:.3f} kind={cond.order.order_kind} "
        f"touched={touched} broken={broken}"
    )

    # Auto-cancel: drifted too far (applies to both kinds — line is unreachable)
    if distance_atr > cond.trigger.max_distance_atr:
        _update_last_poll(cond, now, market_price, line_price, distance_atr)
        _append_event(cond, ConditionalEvent(
            ts=now, kind="drifted_far",
            price=market_price, line_price=line_price, distance_atr=distance_atr,
            message=f"distance {distance_atr:.2f} > max {cond.trigger.max_distance_atr} ATR",
        ))
        _store.set_status(
            cond.conditional_id, "cancelled",
            reason=f"drifted too far ({distance_atr:.1f} ATR)",
        )
        return

    # ── BOUNCE kind ────────────────────────────────────────────
    if cond.order.order_kind == "bounce":
        # Line broken = line is dead for bounce trade → cancel
        if broken:
            _update_last_poll(cond, now, market_price, line_price, distance_atr)
            _append_event(cond, ConditionalEvent(
                ts=now, kind="line_broken",
                price=market_price, line_price=line_price, distance_atr=distance_atr,
                message=f"bounce: line broken by {distance_atr:.2f} ATR, cancelling",
            ))
            _store.set_status(
                cond.conditional_id, "cancelled",
                reason=f"bounce: line broken by {distance_atr:.1f} ATR",
            )
            return

        # Trigger on touch
        if touched:
            _update_last_poll(cond, now, market_price, line_price, distance_atr)
            _append_event(cond, ConditionalEvent(
                ts=now, kind="triggered",
                price=market_price, line_price=line_price, distance_atr=distance_atr,
                message=f"BOUNCE triggered: dist {distance_atr:.3f} ATR ≤ {cond.trigger.tolerance_atr}",
            ))
            _store.set_status(cond.conditional_id, "triggered", reason="bounce_touched")
            await _handle_trigger(cond, market_price, line_price, atr)
            return

        # Normal poll (still waiting for touch)
        _update_last_poll(cond, now, market_price, line_price, distance_atr)
        _append_event(cond, ConditionalEvent(
            ts=now, kind="tolerance_check",
            price=market_price, line_price=line_price, distance_atr=distance_atr,
            message=event_msg,
        ))
        return

    # ── BREAKOUT kind ──────────────────────────────────────────
    if cond.order.order_kind == "breakout":
        # Trigger when the line IS broken (close-through beyond break_thresh)
        if broken:
            _update_last_poll(cond, now, market_price, line_price, distance_atr)
            _append_event(cond, ConditionalEvent(
                ts=now, kind="triggered",
                price=market_price, line_price=line_price, distance_atr=distance_atr,
                message=f"BREAKOUT triggered: close through {cond.side} by {distance_atr:.2f} ATR",
            ))
            _store.set_status(cond.conditional_id, "triggered", reason="breakout_confirmed")
            await _handle_trigger(cond, market_price, line_price, atr)
            return

        # Not yet broken — keep watching
        _update_last_poll(cond, now, market_price, line_price, distance_atr)
        _append_event(cond, ConditionalEvent(
            ts=now, kind="tolerance_check",
            price=market_price, line_price=line_price, distance_atr=distance_atr,
            message=event_msg,
        ))
        return

    # Unknown kind — defensive fallback
    _update_last_poll(cond, now, market_price, line_price, distance_atr)


# ─────────────────────────────────────────────────────────────
# Trigger handler — Telegram alert + optional exchange submit
# ─────────────────────────────────────────────────────────────
async def _handle_trigger(
    cond: ConditionalOrder,
    market_price: float,
    line_price: float,
    atr: float,
) -> None:
    # 1. Telegram alert — reuses existing subscriber
    try:
        await _send_tg_alert(cond, market_price, line_price, atr)
    except Exception as e:
        print(f"[conditional_watcher] TG alert failed: {e}", flush=True)

    # 2. Optional exchange submit
    if cond.order.submit_to_exchange:
        try:
            await _submit_exchange(cond, market_price, line_price, atr)
        except Exception as e:
            print(f"[conditional_watcher] exchange submit failed: {e}", flush=True)
            traceback.print_exc()
            _append_event(cond, ConditionalEvent(
                ts=now_ts(), kind="exchange_error",
                message=f"submit failed: {e}",
            ))
            _store.set_status(cond.conditional_id, "failed", reason=f"submit error: {e}")

    # 3. Write a snapshot for ML/LLM dataset. Best-effort; failures don't
    #    affect the actual order flow.
    try:
        await _write_trade_snapshot(cond, market_price, line_price, atr)
    except Exception as e:
        print(f"[conditional_watcher] snapshot write failed: {e}", flush=True)


# ─────────────────────────────────────────────────────────────
# Trade snapshot — for ML / LLM dataset
# ─────────────────────────────────────────────────────────────
async def _write_trade_snapshot(
    cond: ConditionalOrder,
    market_price: float,
    line_price: float,
    atr: float,
) -> None:
    """Append one JSONL record per trigger.

    Layered for downstream analysis:
      - Raw context: line geometry, pattern stats, bar window
      - Trade params: entry/stop/tp/qty/leverage/direction
      - Line history: touch_number, prior outcomes on this line
      - Market snapshot: ATR, prices, timeframe
      - Outcome: populated later when the trade closes (pending at write time)

    Path: data/logs/trade_snapshots/{symbol}/{yyyymm}.jsonl
    Also writes a PNG chart snapshot if a renderer is available (best-effort).
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone

    now = now_ts()
    # Count how many times this line has been "triggered" before this one
    prior = _store.list_all(manual_line_id=cond.manual_line_id) if hasattr(_store, "list_all") else []
    prior_triggered = [c for c in prior
                       if c.conditional_id != cond.conditional_id
                       and c.triggered_at is not None]
    touch_number = len(prior_triggered) + 1

    # Compute entry/stop/tp from the same function the order submit uses,
    # so the snapshot reflects exactly what the trader committed to.
    try:
        entry_p, stop_p, tp_p = _compute_trade_prices(cond, line_price, atr)
    except Exception:
        entry_p, stop_p, tp_p = None, None, None

    # Reverse chain tracking: if this cond was auto-spawned as the reverse
    # of another cond, the source is in pattern_stats_at_create['_reversed_from'].
    reversed_from = (cond.pattern_stats_at_create or {}).get("_reversed_from")

    snapshot = {
        "snapshot_version": 1,
        "ts": now,
        "ts_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "trade_id": f"trade_{cond.conditional_id}_{now}",
        "conditional_id": cond.conditional_id,
        "manual_line_id": cond.manual_line_id,
        "symbol": cond.symbol,
        "timeframe": cond.timeframe,
        "side": cond.side,  # support/resistance
        "direction": cond.order.direction,  # long/short
        "order_kind": cond.order.order_kind,
        "touch_number": touch_number,
        "reversed_from": reversed_from,

        # Layer 1 / 2: line geometry
        "line_geometry": {
            "t_start": cond.t_start,
            "t_end": cond.t_end,
            "price_start": cond.price_start,
            "price_end": cond.price_end,
            "extend_left": cond.extend_left,
            "extend_right": cond.extend_right,
            "slope_per_sec": (
                (cond.price_end - cond.price_start) / max(1, cond.t_end - cond.t_start)
            ),
            "slope_pct_per_bar": None,  # filled later if needed
            "length_sec": cond.t_end - cond.t_start,
        },
        "line_price_at_trigger": line_price,

        # Layer 3: market context
        "market": {
            "price": market_price,
            "atr": atr,
            "atr_pct_of_price": (atr / market_price) if market_price else None,
        },

        # Trade params (committed)
        "trade_params": {
            "entry_price": entry_p,
            "stop_price": stop_p,
            "tp_price": tp_p,
            "leverage": cond.order.leverage,
            "tolerance_pct_of_line": cond.order.tolerance_pct_of_line,
            "stop_offset_pct_of_line": cond.order.stop_offset_pct_of_line,
            "rr_target": cond.order.rr_target,
            "qty": cond.fill_qty,
            "notional_usd": cond.order.notional_usd,
            "reverse_enabled": cond.order.reverse_enabled,
        },

        # Pre-existing pattern analysis from the /analyze endpoint
        "pattern_stats_at_create": cond.pattern_stats_at_create,

        # Layer 4: outcome (filled later via _update_snapshot_outcome)
        "outcome": {
            "status": "pending",
            "exit_price": None,
            "exit_ts": None,
            "exit_reason": None,
            "pnl": None,
            "pnl_pct": None,
            "mae_pct": None,  # max adverse excursion
            "mfe_pct": None,  # max favorable excursion
        },
    }

    # Persist via snapshots module (trades.jsonl append-only; outcomes
    # live in a separate file populated by backfill_pending_outcomes).
    from .snapshots import write_trade
    write_trade(snapshot)
    # Best-effort PNG snapshot for visual complement / future VLA training.
    try:
        from .chart_renderer import render_trade_snapshot_png
        await render_trade_snapshot_png(cond, snapshot)
    except Exception as e:
        print(f"[snapshot] png render skipped: {e}", flush=True)


# ─────────────────────────────────────────────────────────────
# Auto-reverse on stop-loss
# ─────────────────────────────────────────────────────────────
async def _spawn_reverse_conditional(src: ConditionalOrder, reason: str) -> None:
    """When a cond's stop-loss fires (detected via reconcile), spawn a NEW
    cond on the SAME manual_line_id with direction flipped. The user's
    pre-configured reverse_* fields on the source OrderConfig become the
    new cond's entry/stop pct.

    The line does NOT die on stop-out under the user's strategy — stop-out
    means the line "flipped polarity" (support→resistance or vice versa)
    and a reverse trade fires automatically.
    """
    from .types import OrderConfig, TriggerConfig
    if not src.order.reverse_enabled:
        return
    if src.order.reverse_entry_offset_pct <= 0:
        print(
            f"[reverse] src={src.conditional_id} reverse_enabled but "
            f"entry/stop pct not set — skipping",
            flush=True,
        )
        return

    flipped_dir = "short" if src.order.direction == "long" else "long"
    flipped_side = "resistance" if src.side == "support" else "support"

    new_cond = ConditionalOrder(
        conditional_id=f"rev_{src.conditional_id}_{now_ts()}",
        manual_line_id=src.manual_line_id,
        symbol=src.symbol,
        timeframe=src.timeframe,
        side=flipped_side,  # type: ignore
        t_start=src.t_start,
        t_end=src.t_end,
        price_start=src.price_start,
        price_end=src.price_end,
        extend_left=src.extend_left,
        extend_right=src.extend_right,
        pattern_stats_at_create={
            **(src.pattern_stats_at_create or {}),
            "_reversed_from": src.conditional_id,
            "_reverse_reason": reason,
        },
        trigger=TriggerConfig(
            tolerance_atr=src.trigger.tolerance_atr,
            poll_seconds=src.trigger.poll_seconds,
            max_age_seconds=0,
            max_distance_atr=0.0,
            break_threshold_atr=src.trigger.break_threshold_atr,
        ),
        order=OrderConfig(
            direction=flipped_dir,  # type: ignore
            order_kind=src.order.order_kind,
            entry_offset_points=None,
            entry_offset_atr=0.0,
            stop_points=None,
            stop_atr=0.3,
            rr_target=src.order.reverse_rr_target or src.order.rr_target,
            tp_price=None,
            notional_usd=src.order.notional_usd,
            equity_pct=None,
            risk_pct=None,
            submit_to_exchange=src.order.submit_to_exchange,
            exchange_mode=src.order.exchange_mode,
            tolerance_pct_of_line=src.order.reverse_entry_offset_pct,
            # Bug 6 fix (2026-04-22): was hardcoded 0.0, now respects the
            # user's `reverse_stop_offset_pct` config. Fallback to the
            # source cond's stop_offset_pct_of_line if reverse field is 0
            # so the reverse inherits a sane stop instead of snapping to
            # the 0.3% fail-safe default in the line-broken check.
            stop_offset_pct_of_line=(
                src.order.reverse_stop_offset_pct
                if src.order.reverse_stop_offset_pct and src.order.reverse_stop_offset_pct > 0
                else src.order.stop_offset_pct_of_line
            ),
            leverage=src.order.reverse_leverage or src.order.leverage,
            # Don't chain auto-reverse infinitely by default
            reverse_enabled=False,
            reverse_entry_offset_pct=0.0,
            reverse_stop_offset_pct=0.0,
            reverse_rr_target=None,
            reverse_leverage=None,
        ),
        status="pending",
        created_at=now_ts(),
        updated_at=now_ts(),
    )
    try:
        _store.create(new_cond)
        _append_event(new_cond, ConditionalEvent(
            ts=now_ts(), kind="created",
            message=f"auto-reverse spawned from {src.conditional_id} ({reason})",
        ))
        print(f"[reverse] spawned {new_cond.conditional_id} from {src.conditional_id}", flush=True)
    except Exception as e:
        print(f"[reverse] failed to create reverse cond: {e}", flush=True)


async def _send_tg_alert(cond: ConditionalOrder, market_price: float, line_price: float, atr: float) -> None:
    """Push a Telegram notification about the trigger."""
    try:
        import os
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            print("[conditional_watcher] Telegram token/chat_id not set, skipping alert", flush=True)
            return
        import httpx
        text = (
            f"🎯 CONDITIONAL TRIGGERED\n"
            f"{cond.symbol} {cond.timeframe} {cond.side}\n"
            f"market={market_price:.4f}  line={line_price:.4f}\n"
            f"direction={cond.order.direction}  "
            f"size={cond.order.notional_usd or '?'}USDT\n"
            f"submit_to_exchange={cond.order.submit_to_exchange}\n"
            f"id={cond.conditional_id}"
        )
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
            )
    except Exception as e:
        print(f"[conditional_watcher] tg send err: {e}", flush=True)


async def _submit_exchange(cond: ConditionalOrder, market_price: float, line_price: float, atr: float) -> None:
    """Actually place the order on the exchange (paper or live).

    Entry + stop are computed RELATIVE TO THE LINE PRICE at trigger time,
    not the market price. Because the line's slope moves the projected
    price on every poll, "the line" is the reference the user drew
    relative to — not whatever random tick happened to be when the
    threshold was crossed.
    """
    # Compute reference prices (relative to the LINE, not market)
    entry_price, stop_price, tp_price = _compute_trade_prices(cond, line_price, atr)
    qty = await _compute_qty(cond, entry_price, atr)
    if qty is None or qty <= 0:
        raise ValueError(f"invalid qty: {qty}")

    _append_event(cond, ConditionalEvent(
        ts=now_ts(), kind="exchange_submitted",
        price=market_price, line_price=line_price,
        message=(
            f"submitting {cond.order.direction} {qty:.6f} {cond.symbol} "
            f"entry={entry_price:.4f} stop={stop_price:.4f} tp={tp_price:.4f} "
            f"(line={line_price:.4f} mkt={market_price:.4f})"
        ),
    ))

    # Route to appropriate engine
    if cond.order.exchange_mode == "live":
        await _submit_live(cond, qty, entry_price, stop_price, tp_price, market_price, line_price, atr)
    else:
        await _submit_paper(cond, qty, entry_price, stop_price, tp_price, market_price, line_price, atr)


def _compute_trade_prices(cond: ConditionalOrder, line_price: float, atr: float) -> tuple[float, float, float]:
    """Entry/stop/tp, all computed relative to the LINE PRICE using
    line-relative percentages (the only supported mode).

    Entry sign convention:
      - Long:  entry = line * (1 + tol_pct/100)  — buffer on the upside
               stop  = line * (1 - stop_pct/100) — fail on the downside
      - Short: entry = line * (1 - tol_pct/100)  — buffer on the downside
               stop  = line * (1 + stop_pct/100) — fail on the upside

    The "buffer" (tolerance_pct_of_line) is a safety margin that
    guarantees the order fills on a real touch, because price rarely
    kisses the line with zero error.

    tp_price falls through: absolute tp_price > rr_target on risk distance.
    """
    oc = cond.order
    dir_sign = 1 if oc.direction == "long" else -1

    tol_pct = float(oc.tolerance_pct_of_line or 0.0)
    stop_pct = max(0.0, float(oc.stop_offset_pct_of_line or 0.0))
    if tol_pct <= 0:
        raise ValueError(
            f"tolerance_pct_of_line ({tol_pct}) and stop_offset_pct_of_line "
            f"({stop_pct}) must both be > 0 — legacy absolute-offset mode is "
            f"deleted. Cond: {cond.conditional_id}"
        )

    entry_price = line_price * (1.0 + (tol_pct / 100.0) * dir_sign)
    stop_price = line_price * (1.0 - (stop_pct / 100.0) * dir_sign)

    # TP: absolute > rr
    risk = abs(entry_price - stop_price)
    if oc.tp_price is not None:
        tp_price = float(oc.tp_price)
    elif oc.rr_target is not None:
        tp_price = entry_price + oc.rr_target * risk * dir_sign
    else:
        tp_price = 0.0

    return entry_price, stop_price, tp_price


async def _submit_live(cond, qty, entry_price, stop_price, tp_price, market_price, line_price, atr):
    """Submit to Bitget via the existing LiveExecutionAdapter.submit_live_entry.

    All prices are already computed by _compute_trade_prices() relative to
    the LINE at trigger time (not market).
    """
    from server.execution.live_adapter import LiveExecutionAdapter
    from server.execution.types import OrderIntent, stable_execution_id

    # Build intent — use conditional_id as the stable lineage so duplicate
    # triggers can't double-submit (adapter dedupes by clientOid).
    intent_id = stable_execution_id("cond", cond.conditional_id, "entry")
    signal_id = stable_execution_id("cond_sig", cond.conditional_id)
    client_oid = stable_execution_id("cond_coid", cond.conditional_id)
    intent = OrderIntent(
        order_intent_id=intent_id,
        signal_id=signal_id,
        line_id=cond.manual_line_id,
        client_order_id=client_oid,
        symbol=cond.symbol,
        timeframe=cond.timeframe,
        side=cond.order.direction,  # type: ignore
        order_type="limit",
        trigger_mode="conditional_touch",
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        tp_price=float(tp_price),
        quantity=float(qty),
        status="approved",
        reason=f"conditional {cond.conditional_id} triggered",
        created_at_bar=-1,
        created_at_ts=now_ts(),
        post_only=True,
    )

    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        raise RuntimeError(
            "Bitget API keys not configured (.env BITGET_API_KEY / BITGET_SECRET / BITGET_PASSPHRASE). "
            "Cannot submit to live exchange. Use exchange_mode='paper' or configure keys."
        )

    mode = cond.order.exchange_mode  # "paper" was already routed to _submit_paper
    resp = await adapter.submit_live_entry(intent, mode=mode)  # type: ignore

    if resp.get("ok"):
        exchange_id = resp.get("exchange_order_id") or ""
        # Mirror status to in-memory so the subsequent _store.update doesn't
        # accidentally REVERT the disk status from 'triggered' back to
        # 'pending' (which would cause the watcher to re-trigger and double-
        # submit, hitting Bitget's duplicate-clientOid guard).
        cond.status = "triggered"
        cond.triggered_at = now_ts()
        cond.exchange_order_id = exchange_id
        cond.fill_price = float(resp.get("submitted_price") or market_price)
        cond.fill_qty = float(qty)
        try:
            # Re-load latest events so we don't clobber the trigger event
            latest = _store.get(cond.conditional_id)
            if latest is not None:
                cond.events = latest.events
            _store.update(cond)
        except ValueError:
            pass
        _append_event(cond, ConditionalEvent(
            ts=now_ts(), kind="exchange_acked",
            price=market_price,
            message=f"Bitget ack: order_id={exchange_id}",
            extra={"exchange_response": resp},
        ))
    else:
        reason = resp.get("reason") or "unknown"
        _append_event(cond, ConditionalEvent(
            ts=now_ts(), kind="exchange_error",
            price=market_price,
            message=f"Bitget rejected: {reason}",
            extra={"exchange_response": resp},
        ))
        _store.set_status(cond.conditional_id, "failed", reason=f"bitget rejected: {reason}")


async def _submit_paper(cond, qty, entry_price, stop_price, tp_price, market_price, line_price, atr):
    """Record a 'would-be paper fill' event with all the trade params.

    We don't push into the existing PaperExecutionEngine (that's bar-driven,
    not intent-driven). This path just persists a high-fidelity log so the
    user can audit what WOULD have happened if it were live.
    """
    _append_event(cond, ConditionalEvent(
        ts=now_ts(), kind="exchange_acked",
        price=market_price, line_price=line_price,
        message=(
            f"[paper] {cond.order.direction} {qty:.6f} {cond.symbol} "
            f"entry={entry_price:.4f} stop={stop_price:.4f} tp={tp_price:.4f} "
            f"(line at trigger={line_price:.4f}, mkt={market_price:.4f})"
        ),
        extra={
            "intent": {
                "direction": cond.order.direction,
                "kind": cond.order.order_kind,
                "qty": qty,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "tp_price": tp_price,
                "line_price_at_trigger": line_price,
                "market_price_at_trigger": market_price,
                "rr_target": cond.order.rr_target,
            }
        },
    ))
    # Persist fill fields on the cond so the UI row shows them
    cond.fill_price = entry_price
    cond.fill_qty = float(qty)
    try:
        _store.update(cond)
    except ValueError:
        pass


async def _compute_qty(cond: ConditionalOrder, market_price: float, atr: float) -> float | None:
    """Derive qty from order config.

    Priority: leverage (preferred for cross-margin) → notional_usd → fail.

    With leverage set, the user's cross-margin account equity is fetched
    live and notional = equity * leverage. This matches the modal's UX
    where the user sets "10x leverage + 0.1% stop" and sees "account risk
    = 1%" in real time.
    """
    oc = cond.order
    if oc.leverage and oc.leverage > 0:
        if market_price <= 0:
            return None
        try:
            from server.execution.live_adapter import LiveExecutionAdapter
            adapter = LiveExecutionAdapter()
            if not adapter.has_api_keys():
                print(f"[_compute_qty] leverage set but no API keys", flush=True)
                return None
            acct = await adapter.get_live_account_status(mode=cond.order.exchange_mode)
            equity = float(
                acct.get("equity")
                or acct.get("totalEquity")
                or acct.get("usdtEquity")
                or 0
            )
            if equity <= 0:
                print(f"[_compute_qty] equity={equity} invalid for leverage sizing", flush=True)
                return None
            notional = equity * float(oc.leverage)
            return notional / market_price
        except Exception as e:
            print(f"[_compute_qty] leverage path failed: {e}", flush=True)
            return None

    if oc.notional_usd and oc.notional_usd > 0:
        if market_price <= 0:
            return None
        return oc.notional_usd / market_price

    print(
        f"[_compute_qty] cond {cond.conditional_id}: neither leverage nor "
        f"notional_usd set — cannot size order.",
        flush=True,
    )
    return None


# ─────────────────────────────────────────────────────────────
# Price + ATR fetchers
# ─────────────────────────────────────────────────────────────
def _tf_seconds(tf: str) -> int:
    return _TF_REPLAN_SECONDS.get(tf, 900)


def _manual_slope_per_bar(cond: ConditionalOrder) -> float:
    span = int(cond.t_end) - int(cond.t_start)
    if span <= 0:
        return 0.0
    return (float(cond.price_end) - float(cond.price_start)) / span * _tf_seconds(cond.timeframe)


def _position_size(row: dict[str, Any]) -> float:
    for key in ("total", "available", "size", "pos", "position"):
        raw = row.get(key)
        if raw in (None, "", 0, "0"):
            continue
        try:
            value = abs(float(raw))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def _position_open_ts(row: dict[str, Any]) -> int:
    for key in ("openTime", "openTimestamp", "cTime", "ctime", "createdTime", "createTime"):
        raw = row.get(key)
        if raw in (None, "", 0, "0"):
            continue
        try:
            ts = int(float(raw))
        except (TypeError, ValueError):
            continue
        if ts > 10_000_000_000:
            ts //= 1000
        if ts > 0:
            return ts
    return 0


def _position_entry_price(row: dict[str, Any]) -> float:
    for key in ("openPriceAvg", "averageOpenPrice", "openAvgPrice", "entryPrice"):
        raw = row.get(key)
        if raw in (None, "", 0, "0"):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def _find_position_for_cond(rows: list[dict[str, Any]], cond: ConditionalOrder) -> dict[str, Any] | None:
    symbol = cond.symbol.upper()
    want_side = cond.order.direction.lower()
    fallback: dict[str, Any] | None = None
    for row in rows:
        if str(row.get("symbol") or "").upper() != symbol:
            continue
        if _position_size(row) <= 0:
            continue
        side = str(row.get("holdSide") or row.get("posSide") or "").lower()
        if side == want_side:
            return row
        if fallback is None:
            fallback = row
    return fallback


async def _sync_sl_to_line_now(
    cond: ConditionalOrder,
    pos: dict[str, Any],
    target_sl_price: float,
) -> None:
    """Immediately cancel Bitget's old preset SL and place a new one at
    line@fill_time. Runs synchronously on fill detection so we don't wait
    for mar_bb_runner's 60s trailing loop. Between fill and first trailing
    scan the old bar_open-based preset SL could be *hundreds* of bps below
    fill_price (breakout fills blow far past the bar_open trigger), and a
    wrong-way price wick within 60s hits the stale SL for a huge loss.

    User 2026-04-22: BAS entered at 0.01693 with preset SL at 0.01634
    (bar_open basis, 3.5% below fill). Stop hit in 3 min 34 sec, before
    first trailing scan. This function closes that window.
    """
    import httpx  # noqa: F401 — ensure imported
    from server.execution.live_adapter import LiveExecutionAdapter
    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return
    mode = cond.order.exchange_mode
    if mode not in ("demo", "live"):
        return
    sym = cond.symbol.upper()
    direction = cond.order.direction
    hold_side = "long" if direction == "long" else "short"

    # 1. Query current position SL plans on Bitget, cancel them all.
    try:
        resp = await adapter._bitget_request(
            "GET", "/api/v2/mix/order/orders-plan-pending",
            mode=mode,
            params={"symbol": sym, "productType": "USDT-FUTURES", "planType": "profit_loss"},
        )
        if resp.get("code") == "00000":
            rows = ((resp.get("data") or {}).get("entrustedList") or
                    (resp.get("data") or {}).get("orderList") or [])
            for row in rows:
                oid = str(row.get("orderId") or "")
                plan_subtype = str(row.get("planType") or "")
                # Cancel existing SL plans (pos_loss / loss_plan) for this side
                if oid and "loss" in plan_subtype.lower():
                    await adapter._bitget_request(
                        "POST", "/api/v2/mix/order/cancel-plan-order",
                        mode=mode,
                        body={
                            "symbol": sym,
                            "productType": "USDT-FUTURES",
                            "marginCoin": "USDT",
                            "orderId": oid,
                            "planType": plan_subtype,
                        },
                    )
    except Exception as exc:
        print(f"[sync_sl] cancel old SL err {sym}: {exc}", flush=True)

    # 2. Place a fresh pos_loss plan at target_sl_price.
    try:
        body = {
            "symbol": sym,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "planType": "pos_loss",
            "triggerPrice": f"{target_sl_price:.8f}".rstrip("0").rstrip("."),
            "triggerType": "mark_price",
            "holdSide": hold_side,
            "executePrice": "0",      # market close
            "clientOid": f"sync_sl_{cond.conditional_id}_{int(now_ts())}",
        }
        r = await adapter._bitget_request(
            "POST", "/api/v2/mix/order/place-tpsl-order",
            mode=mode, body=body,
        )
        if r.get("code") == "00000":
            print(f"[sync_sl] {sym} new SL @ {target_sl_price:.6f} "
                  f"oid={(r.get('data') or {}).get('orderId','')}", flush=True)
        else:
            print(f"[sync_sl] {sym} place err: code={r.get('code')} msg={r.get('msg')}", flush=True)
    except Exception as exc:
        print(f"[sync_sl] place new SL err {sym}: {exc}", flush=True)


def _register_manual_trailing_if_position_open(cond: ConditionalOrder, pos: dict[str, Any], now: int) -> None:
    from server.strategy.mar_bb_runner import register_trendline_params

    opened_ts = _position_open_ts(pos) or int(cond.triggered_at or now)
    line_ref_price = float(cond.line_price_at(opened_ts))
    entry_price = _position_entry_price(pos) or float(cond.fill_price or line_ref_price)
    if line_ref_price <= 0 or entry_price <= 0:
        return
    stop_offset_pct = max(0.0, float(cond.order.stop_offset_pct_of_line or 0.0))
    last_sl_set = (
        line_ref_price * (1.0 - stop_offset_pct / 100.0)
        if cond.order.direction == "long"
        else line_ref_price * (1.0 + stop_offset_pct / 100.0)
    )
    # SL distance cap: keep real stop ≤ user's configured (buffer+stop) %.
    # When breakout fill puts entry far past trigger, line-based SL
    # becomes far from entry; cap it so risk matches user's intended
    # setup. Uses cond.order.tolerance_pct_of_line + stop_offset_pct_of_line.
    tol_pct = max(0.0, float(cond.order.tolerance_pct_of_line or 0.0))
    stp_pct = max(0.0, float(cond.order.stop_offset_pct_of_line or 0.0))
    max_sl_pct = (tol_pct + stp_pct) / 100.0
    if max_sl_pct > 0:
        if cond.order.direction == "long":
            min_sl_cap = entry_price * (1.0 - max_sl_pct)
            if last_sl_set < min_sl_cap:
                print(f"[trailing_register] {cond.symbol} SL capped to setup total: "
                      f"line-based {last_sl_set:.6f} -> {min_sl_cap:.6f} "
                      f"(entry {entry_price:.6f}, cap {max_sl_pct*100:.2f}%)", flush=True)
                last_sl_set = min_sl_cap
        else:
            max_sl_cap = entry_price * (1.0 + max_sl_pct)
            if last_sl_set > max_sl_cap:
                print(f"[trailing_register] {cond.symbol} SL capped to setup total: "
                      f"line-based {last_sl_set:.6f} -> {max_sl_cap:.6f} "
                      f"(entry {entry_price:.6f}, cap {max_sl_pct*100:.2f}%)", flush=True)
                last_sl_set = max_sl_cap
    register_trendline_params(
        cond.symbol.upper(),
        slope=_manual_slope_per_bar(cond),
        intercept=0.0,
        entry_bar=0,
        entry_price=entry_price,
        side=cond.order.direction,
        tf=cond.timeframe,
        created_ts=opened_ts,
        tp_price=float(cond.order.tp_price or 0.0),
        last_sl_set=last_sl_set,
        line_ref_ts=opened_ts,
        line_ref_price=line_ref_price,
        stop_offset_pct=stop_offset_pct,
    )
    # 2026-04-22 fix: don't wait 60s for mar_bb scan to update SL. The
    # preset SL from the original plan was based on bar_open line, which
    # may be far below actual fill price. Immediately sync Bitget SL to
    # line_ref_price ± stop_offset_pct (= what trailing would do on
    # next scan). User lost $11 on BAS because SL was at 0.01634
    # (bar_open 12:00 basis) while fill was at 0.01693 — got stopped
    # in 3.5 minutes before scan caught up.
    try:
        import asyncio as _asyncio
        _asyncio.create_task(_sync_sl_to_line_now(cond, pos, last_sl_set))
    except Exception as exc:
        print(f"[reconcile] sync_sl schedule err {cond.symbol}: {exc}", flush=True)


async def _reconcile_against_bitget() -> None:
    """Reconcile local triggered conds with Bitget's authoritative state.

    Hard design invariant (PRINCIPLES.md P9 + P9a, 2026-04-23):
      Local state transitions (triggered → cancelled / filled) happen ONLY
      on affirmative Bitget evidence. Never on "my oid was absent from a
      list". If we can't tell for sure, we keep the cond triggered and
      retry next cycle.

    Flow:
      1. Fetch pending oids + positions for each mode. Both can fail; the
         classifier treats "fetch failed" as "no info from that source".
      2. For each triggered cond, call classify_cond_state() which:
            pending has my oid    → LIVE
            position matches me   → FILLED
            otherwise             → query orders-plan-history for my oid
                                       row state=cancelled → CANCELLED
                                       row state=filled    → FILLED
                                       row state=live      → LIVE
                                       row missing / error → UNKNOWN
      3. Act only on LIVE / FILLED / CANCELLED. UNKNOWN → stay triggered.

    Why this beats the old "not-in-pending → cancel" logic: a 429 on
    orders-plan-pending used to cascade-cancel every triggered plan
    cond in one tick. Now pending-list absence is just a hint that
    forces a history lookup; only history can confirm a cancel.
    """
    triggered = _store.list_all(status="triggered")
    if not triggered:
        return
    try:
        from server.execution.live_adapter import LiveExecutionAdapter
        from .oid_state import classify_cond_state, CondRemoteState

        adapter = LiveExecutionAdapter()
        if not adapter.has_api_keys():
            return

        modes = {
            cond.order.exchange_mode
            for cond in triggered
            if cond.order.exchange_mode in {"demo", "live"}
        }

        # Per-mode caches of pending + positions. A failed fetch just means
        # classify_cond_state must fall back to its history query — it
        # does NOT block the reconcile tick.
        pending_by_mode: dict[str, tuple[set[str], bool]] = {}   # (oids, ok)
        positions_by_mode: dict[str, tuple[list[dict[str, Any]], bool]] = {}

        for mode in modes:
            pending_oids: set[str] = set()
            ok_regular = False
            ok_plan = False
            try:
                for row in await adapter.get_pending_orders(mode):  # type: ignore[arg-type]
                    oid = str(row.get("orderId") or row.get("order_id") or "")
                    if oid:
                        pending_oids.add(oid)
                ok_regular = True
            except Exception as e:
                print(f"[reconcile] regular pending fetch failed mode={mode}: {e}", flush=True)
            try:
                for row in await adapter.get_pending_plan_orders(mode, plan_type="normal_plan"):  # type: ignore[arg-type]
                    oid = str(row.get("orderId") or row.get("order_id") or "")
                    if oid:
                        pending_oids.add(oid)
                ok_plan = True
            except Exception as e:
                print(f"[reconcile] plan pending fetch failed mode={mode}: {e}", flush=True)
            # "pending_fetch_ok" is the AND — the classifier only trusts
            # pending-list absence as a hint (not evidence) when both paths
            # succeeded, since we cannot know which list a cond's oid
            # should live in without the other also being complete.
            pending_by_mode[mode] = (pending_oids, ok_regular and ok_plan)

            try:
                pos_resp = await adapter._bitget_request(
                    "GET", "/api/v2/mix/position/all-position",
                    mode=mode,
                    params={"productType": "USDT-FUTURES", "marginCoin": "USDT"},
                )
                if pos_resp.get("code") == "00000":
                    positions_by_mode[mode] = (adapter._as_rows(pos_resp.get("data")), True)
                else:
                    positions_by_mode[mode] = ([], False)
            except Exception as e:
                print(f"[reconcile] position fetch failed mode={mode}: {e}", flush=True)
                positions_by_mode[mode] = ([], False)

        for cond in triggered:
            mode = cond.order.exchange_mode
            if mode not in {"demo", "live"}:
                continue
            pending_oids, pending_ok = pending_by_mode.get(mode, (set(), False))
            positions, positions_ok = positions_by_mode.get(mode, ([], False))

            result = await classify_cond_state(
                cond, adapter,
                pending_oids=pending_oids,
                pending_fetch_ok=pending_ok,
                positions=positions,
                positions_fetch_ok=positions_ok,
            )

            # Observability (PRINCIPLES.md P11): always log the decision
            # with path so we can audit any surprise on the user's side.
            print(
                f"[reconcile] cond={cond.conditional_id} oid={cond.exchange_order_id} "
                f"state={result.state.value} path={result.path}"
                + (f" err={result.error}" if result.error else ""),
                flush=True,
            )

            if result.state == CondRemoteState.LIVE:
                _clear_unknown_streak(cond.conditional_id)
                continue

            if result.state == CondRemoteState.FILLED:
                _clear_unknown_streak(cond.conditional_id)
                now = now_ts()
                pos = result.matched_row if (result.matched_row and "holdSide" in (result.matched_row or {})) else None
                # matched_row may be a history row (filled via history) rather
                # than a position row; still fine — use cond.fill_price as fallback.
                if pos is not None:
                    _register_manual_trailing_if_position_open(cond, pos, now)
                    entry = _position_entry_price(pos) or float(cond.fill_price or 0.0)
                    qty = _position_size(pos) or float(cond.fill_qty or 0.0)
                    opened_ts = _position_open_ts(pos) or now
                else:
                    entry = float(cond.fill_price or 0.0)
                    qty = float(cond.fill_qty or 0.0)
                    opened_ts = now
                cond.status = "filled"
                cond.fill_price = entry or cond.fill_price
                cond.fill_qty = qty or cond.fill_qty
                try:
                    latest = _store.get(cond.conditional_id)
                    if latest is not None:
                        cond.events = latest.events
                    _store.update(cond)
                    _store.append_event(cond.conditional_id, ConditionalEvent(
                        ts=now, kind="exchange_acked",
                        price=entry or None,
                        line_price=cond.line_price_at(opened_ts),
                        message=f"reconcile: classifier={result.path}; fill captured",
                    ))
                except Exception as e:
                    print(f"[reconcile] fill status update failed for {cond.conditional_id}: {e}", flush=True)
                print(f"[reconcile] cond {cond.conditional_id} transitioned triggered→filled ({result.path})", flush=True)
                continue

            if result.state == CondRemoteState.CANCELLED:
                _clear_unknown_streak(cond.conditional_id)
                # Affirmative cancel from Bitget history. Safe to transition.
                winner = _store.set_status_if(
                    cond.conditional_id,
                    from_status="triggered",
                    to_status="cancelled",
                    reason=f"reconcile history-confirmed cancel: {result.path}",
                )
                if winner is None:
                    print(f"[reconcile] cond {cond.conditional_id} CAS lost — another path already handled it, skipping reverse", flush=True)
                    continue
                try:
                    _store.append_event(cond.conditional_id, ConditionalEvent(
                        ts=now_ts(), kind="cancelled",
                        message=f"reconcile: Bitget history confirms oid {cond.exchange_order_id} cancelled ({result.path})",
                    ))
                except Exception:
                    pass
                try:
                    await _spawn_reverse_conditional(cond, reason="reconcile_history_cancelled")
                except Exception as e:
                    print(f"[reconcile] reverse spawn failed for {cond.conditional_id}: {e}", flush=True)
                continue

            # UNKNOWN. The whole point of this branch: do NOTHING. Keep
            # the cond triggered; next reconcile cycle will re-classify.
            # The log line above already reports the 'path' so ops can see
            # which reason the classifier couldn't resolve.
            _record_unknown_tick(cond, result)
    except Exception as e:
        print(f"[reconcile] err: {e}", flush=True)


# ─── UNKNOWN-tick escalation ───────────────────────────────────────────
#
# If the same cond keeps classifying as UNKNOWN for many consecutive
# reconcile cycles, something non-transient is happening (Bitget API
# persistently broken / the oid is so old it's off the history window /
# a bug in our classifier). We track consecutive unknowns per cond and
# emit ONE Telegram notification on crossing a threshold so the user
# isn't left flying blind in silence.
_UNKNOWN_STREAK: dict[str, int] = {}
_UNKNOWN_ALERT_THRESHOLD = 10   # ~5 min at 30s reconcile cadence
_UNKNOWN_ALERT_FIRED: set[str] = set()


def _record_unknown_tick(cond: ConditionalOrder, result: Any) -> None:
    cid = cond.conditional_id
    streak = _UNKNOWN_STREAK.get(cid, 0) + 1
    _UNKNOWN_STREAK[cid] = streak
    if streak >= _UNKNOWN_ALERT_THRESHOLD and cid not in _UNKNOWN_ALERT_FIRED:
        _UNKNOWN_ALERT_FIRED.add(cid)
        print(
            f"[reconcile] cond {cid} has been UNKNOWN for {streak} cycles "
            f"(~{streak * 30}s). Last path={getattr(result, 'path', '?')}. "
            f"Check Bitget manually — this oid may need intervention.",
            flush=True,
        )
        # Telegram notify so the user knows something's off without
        # having to watch the log.
        try:
            from server.core.bus import get_event_bus
            bus = get_event_bus()
            if bus is not None:
                bus.publish(
                    "conditional.reconcile_unknown_streak",
                    {
                        "conditional_id": cid,
                        "symbol": cond.symbol,
                        "oid": cond.exchange_order_id,
                        "streak": streak,
                        "last_path": getattr(result, "path", "?"),
                    },
                )
        except Exception:
            pass


def _clear_unknown_streak(conditional_id: str) -> None:
    """Called by non-UNKNOWN resolution paths (LIVE/FILLED/CANCELLED)
    so a future UNKNOWN flash doesn't carry over the old streak."""
    _UNKNOWN_STREAK.pop(conditional_id, None)
    _UNKNOWN_ALERT_FIRED.discard(conditional_id)


async def _fetch_market_price(symbol: str) -> float | None:
    """Latest price for a symbol. Tries Bitget ticker first (fast, single
    HTTP call), falls back to /api/ohlcv tail (heavy, builds polars df).

    Was previously calling get_ohlcv_with_df on every poll which loads
    1 day of 1m candles per call — significant CPU + cache thrash.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                "https://api.bitget.com/api/v2/mix/market/ticker",
                params={"symbol": symbol.upper(), "productType": "USDT-FUTURES"},
            )
            if r.status_code == 200:
                data = (r.json() or {}).get("data") or []
                if data:
                    row = data[0] if isinstance(data, list) else data
                    mp = row.get("markPrice") or row.get("lastPr")
                    if mp:
                        return float(mp)
    except Exception as e:
        print(f"[conditional_watcher] ticker fetch err {symbol}: {e}", flush=True)

    # Fallback: ohlcv tail (slower)
    try:
        from server.data_service import get_ohlcv_with_df
        polars_df, _ = await get_ohlcv_with_df(
            symbol, "1m", None, days=1,
            history_mode="fast_window",
            include_price_precision=False,
            include_render_payload=False,
        )
        if polars_df is None or polars_df.is_empty():
            return None
        last = polars_df.tail(1).to_pandas().iloc[0]
        return float(last["close"])
    except Exception as e:
        print(f"[conditional_watcher] price fetch fallback err {symbol}: {e}", flush=True)
        return None


async def _fetch_mark_price_strict(symbol: str) -> float | None:
    """Strict mark-price-only fetch for invalidation / line-broken checks.

    Unlike `_fetch_market_price` which falls back to `lastPr` when
    `markPrice` is missing, this returns None on ANY missing-mark case.
    This is critical for semantic consistency with Bitget's trigger
    engine — our plan orders are placed with triggerType="mark_price",
    so the ONLY price that matters for "did the trigger fire / should
    we cancel" is markPrice. Using lastPr here can cause:
      - False cancels (lastPr spikes while markPrice stays tame)
      - Missed real breaks (markPrice breaks while lastPr lags)

    Returns None on any failure mode. Callers MUST treat None as
    "cannot safely judge" — do NOT cancel, do NOT replan.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                "https://api.bitget.com/api/v2/mix/market/ticker",
                params={"symbol": symbol.upper(), "productType": "USDT-FUTURES"},
            )
            if r.status_code == 200:
                data = (r.json() or {}).get("data") or []
                if data:
                    row = data[0] if isinstance(data, list) else data
                    mp = row.get("markPrice")
                    if mp:
                        try:
                            val = float(mp)
                            if val > 0:
                                return val
                        except (TypeError, ValueError):
                            pass
    except Exception as e:
        print(f"[watcher] strict mark fetch err {symbol}: {e}", flush=True)
    return None


async def _handle_line_broken_outcome(
    cond: ConditionalOrder,
    outcome: str,
    reason_text: str,
    reverse_reason: str,
    *,
    mark_for_check: float,
    line_now: float,
    now: int,
) -> None:
    """Apply the correct local transition for a line-broken cancel attempt.

    Semantics per outcome:
      - CANCELLED: set_status('cancelled') + spawn reverse if enabled.
      - FILLED:    set_status('filled') — the order fired before cancel
                   landed. No reverse spawn; the fill IS the position.
      - RETRY:     keep cond triggered, update last_poll_ts so bar-gate
                   holds until next bar.

    Without the FILLED branch the old code would orphan a fresh fill
    by marking it cancelled (and spawn a reverse trade against a real
    open position). Without the CANCELLED-via-classifier branch, a
    Bitget 'not found' on cancel would loop forever.
    """
    if outcome == CancelOutcome.RETRY:
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_error",
            price=mark_for_check, line_price=line_now,
            message=f"{reason_text} — cancel APIs failed AND classifier could not confirm terminal state; plan may still be live. Leaving cond=triggered for next-bar retry.",
        ))
        cond.last_poll_ts = now
        try:
            _store.update(cond)
        except Exception:
            pass
        return

    if outcome == CancelOutcome.FILLED:
        winner = _store.set_status_if(
            cond.conditional_id,
            from_status="triggered",
            to_status="filled",
            reason=f"{reason_text} — but classifier shows FILLED before cancel landed; transitioning to filled (no reverse)",
        )
        if winner is None:
            print(f"[watcher] {cond.symbol} line-broken→filled CAS lost, skipping", flush=True)
            return
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_acked",
            price=mark_for_check, line_price=line_now,
            message=f"{reason_text} — order already FILLED per classifier; cond transitioned to filled",
        ))
        return

    # CANCELLED
    winner = _store.set_status_if(
        cond.conditional_id,
        from_status="triggered",
        to_status="cancelled",
        reason=reason_text,
    )
    if winner is None:
        print(f"[watcher] {cond.symbol} line-broken CAS lost — another path already handled this cond, skipping reverse", flush=True)
        return
    _append_event(cond, ConditionalEvent(
        ts=now, kind="line_broken",
        price=mark_for_check, line_price=line_now,
        message=reason_text,
    ))
    if cond.order.reverse_enabled:
        try:
            await _spawn_reverse_conditional(cond, reason=reverse_reason)
        except Exception as exc:
            print(f"[watcher] line-broken reverse err: {exc}", flush=True)


class CancelOutcome:
    """Outcome tag returned by _cancel_bitget_plan_safely. Callers use
    these to pick the correct local transition — cancelled (dead, may
    reverse), filled (order fired before we cancelled, keep as open
    position), or retry (state unclear, do nothing this bar).

    Using a named tag instead of bool so callers can't silently confuse
    'cancelled' with 'filled' — which would have been a real bug in the
    pre-2026-04-23 line-broken path if a cancel race happened against
    a fill."""
    CANCELLED = "cancelled"   # Bitget confirmed or classifier shows terminal cancel
    FILLED    = "filled"      # Order already fired; caller should treat as filled
    RETRY     = "retry"       # Can't confirm either way; stay triggered


async def _cancel_bitget_plan_safely(
    cond: ConditionalOrder,
    reason_tag: str,
    *,
    adapter: Any | None = None,
) -> str:
    """Attempt to cancel a Bitget plan order and tell the caller what
    actually happened.

    Returns one of CancelOutcome.{CANCELLED, FILLED, RETRY}.

    Decision flow:
      1. Try regular cancel. ok → CANCELLED.
      2. Try plan cancel. ok → CANCELLED.
      3. Both cancel APIs failed. Classify via orders-plan-history:
           CANCELLED → CANCELLED (already cancelled on Bitget)
           FILLED    → FILLED (order fired between our cancel attempt and now)
           LIVE      → RETRY
           UNKNOWN   → RETRY
      4. Fall through → RETRY.

    Contract:
      - CANCELLED: safe to set_status('cancelled') + spawn reverse.
      - FILLED:    safe to set_status('filled'); MUST NOT spawn reverse
                   (the fill is a real position).
      - RETRY:     leave cond.status='triggered', wait for next cycle.

    The classifier fallback fixes the mirror-of-reconcile-bug: before,
    a cancel call that returned non-ok (e.g. 'order already cancelled')
    returned False and locked us in retry forever even though Bitget's
    history clearly showed the order was gone. That blocked
    line-broken → reverse spawn and replan → resubmit indefinitely.
    """
    if adapter is None:
        from server.execution.live_adapter import LiveExecutionAdapter
        adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys() or not cond.exchange_order_id:
        # Nothing to cancel on Bitget — paper mode or order never placed.
        return CancelOutcome.CANCELLED
    try:
        _cr = await adapter.cancel_order(
            cond.symbol.upper(),
            cond.exchange_order_id,
            cond.order.exchange_mode,
        )
        if (_cr or {}).get("ok"):
            return CancelOutcome.CANCELLED
    except Exception as exc:
        print(f"[watcher] {reason_tag} regular-cancel err: {exc} (falling back to plan-cancel)", flush=True)
    try:
        _cr2 = await adapter.cancel_plan_order_any_type(
            cond.symbol.upper(),
            cond.exchange_order_id,
            cond.order.exchange_mode,
        )
        if (_cr2 or {}).get("ok"):
            return CancelOutcome.CANCELLED
    except Exception as exc:
        print(f"[watcher] {reason_tag} plan-cancel err: {exc}", flush=True)

    # Both cancel APIs failed. Go straight to orders-plan-history for
    # authoritative state. We intentionally SKIP the pending-cache fast
    # path here because: (a) cancel just returned non-ok, so pending may
    # be racing with the cancel; (b) if oid is still in pending we'd
    # conclude LIVE and retry, but that's what we're already doing via
    # the per-bar gate, so pending gives us no new signal. History is
    # the only source that can distinguish "cancel failed, order still
    # live" from "cancel failed, order is actually CANCELLED/FILLED".
    try:
        from .oid_state import classify_cond_state, CondRemoteState
        result = await classify_cond_state(
            cond, adapter,
            pending_oids=None, pending_fetch_ok=False,
            positions=None, positions_fetch_ok=False,
        )
        if result.state == CondRemoteState.CANCELLED:
            print(f"[watcher] {reason_tag} cancel APIs failed but history confirms CANCELLED ({result.path})", flush=True)
            return CancelOutcome.CANCELLED
        if result.state == CondRemoteState.FILLED:
            print(f"[watcher] {reason_tag} cancel APIs failed but history shows FILLED ({result.path})", flush=True)
            return CancelOutcome.FILLED
        if result.state == CondRemoteState.LIVE:
            print(f"[watcher] {reason_tag} cancel APIs failed, history says LIVE ({result.path}); will retry next bar", flush=True)
        else:
            print(f"[watcher] {reason_tag} cancel APIs failed, classifier=UNKNOWN ({result.path}); will retry next bar", flush=True)
    except Exception as exc:
        print(f"[watcher] {reason_tag} classifier fallback err: {exc}", flush=True)
    return CancelOutcome.RETRY


async def _fetch_current_atr(symbol: str, timeframe: str, period: int = 14) -> float | None:
    """Compute ATR for the current timeframe from the last 20 bars."""
    try:
        from server.data_service import get_ohlcv_with_df
        polars_df, _ = await get_ohlcv_with_df(
            symbol, timeframe, None, days=7,
            history_mode="fast_window",
            include_price_precision=False,
            include_render_payload=False,
        )
        if polars_df is None or polars_df.is_empty():
            return None
        pdf = polars_df.tail(period + 5).to_pandas()
        h = pdf["high"].astype(float).values
        l = pdf["low"].astype(float).values
        c = pdf["close"].astype(float).values
        import numpy as _np
        prev_c = _np.roll(c, 1)
        prev_c[0] = c[0]
        tr = _np.maximum(h - l, _np.maximum(_np.abs(h - prev_c), _np.abs(l - prev_c)))
        return float(_np.mean(tr[-period:]))
    except Exception as e:
        print(f"[conditional_watcher] atr fetch err {symbol}/{timeframe}: {e}", flush=True)
        return None


# ─────────────────────────────────────────────────────────────
# Mutation helpers
# ─────────────────────────────────────────────────────────────

# Kinds worth forwarding to Telegram / global bus. Everything else
# (poll, tolerance_check, created, drifted_far, expired) is internal
# bookkeeping that would spam the user's phone.
_TG_BROADCAST_KINDS = frozenset({
    "exchange_submitted",  # plan order placed on Bitget
    "exchange_acked",      # position opened (plan filled)
    "exchange_error",      # Bitget rejected / error returned
    "triggered",           # local trigger conditions met
    "breakout",            # breakout variant of triggered
    "cancelled",           # cond cancelled (manual / line_broken / reconcile)
    "line_broken",         # line invalidated
})


def _append_event(cond: ConditionalOrder, event: ConditionalEvent) -> None:
    _store.append_event(cond.conditional_id, event)
    # Bridge to global event bus (Telegram subscriber, SSE broadcast,
    # anything else listening). User 2026-04-22: "Bitget doesn't notify
    # me when plans fire; I need Telegram for every order event." Fire-
    # and-forget via asyncio so _append_event stays sync.
    kind = getattr(event, "kind", None)
    if kind not in _TG_BROADCAST_KINDS:
        return
    try:
        from ..core.events import bus, Event
        bus_event = Event(
            type=f"conditional.{kind}",
            payload={
                "kind": kind,
                "conditional_id": cond.conditional_id,
                "symbol": cond.symbol,
                "timeframe": cond.timeframe,
                "side": cond.side,
                "direction": cond.order.direction,
                "manual_line_id": cond.manual_line_id,
                "exchange_order_id": cond.exchange_order_id,
                "exchange_mode": cond.order.exchange_mode,
                "message": getattr(event, "message", "") or "",
                "price": getattr(event, "price", None),
                "line_price": getattr(event, "line_price", None),
            },
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(bus.publish(bus_event))
        except RuntimeError:
            # Not in async context — drop the broadcast silently rather
            # than spinning up a loop (which could deadlock on Windows).
            pass
    except Exception as exc:
        print(f"[watcher] telegram bridge err: {exc}", flush=True)


def _update_last_poll(
    cond: ConditionalOrder,
    now: int,
    market_price: float | None,
    line_price: float | None = None,
    distance_atr: float | None = None,
) -> None:
    # Mutate in-memory then persist via store.update.
    # IMPORTANT: re-load events from disk first so we don't clobber any
    # events appended by _append_event between the load at the start of
    # this poll and now. Otherwise every tick wipes the events list.
    cond.last_poll_ts = now
    cond.last_market_price = market_price
    cond.last_line_price = line_price
    cond.last_distance_atr = distance_atr
    try:
        latest = _store.get(cond.conditional_id)
        if latest is not None:
            cond.events = latest.events
        _store.update(cond)
    except ValueError:
        pass


async def _maybe_replan(cond: ConditionalOrder, now: int) -> None:
    """For a triggered cond holding a Bitget plan order, periodically
    cancel the existing plan and re-place it with the line's UPDATED
    trigger price. Frequency tied to the line's timeframe."""
    # Skip horizontal lines — their trigger price never changes.
    if cond.price_start == cond.price_end:
        return

    # Bar-gate FIRST (cheap, no API). This cuts the vast majority of
    # per-tick calls before we ever touch Bitget. Plan-existence check
    # moved AFTER the gate so it only runs when we're actually about
    # to replan. Perf 2026-04-21: old ordering hit Bitget 2x/sec per
    # triggered cond (per user "之前挂单都是几秒就好了 为什么现在需要这么久").
    global _force_replan_set
    forced = cond.conditional_id in _force_replan_set
    if forced:
        _force_replan_set.discard(cond.conditional_id)
    else:
        last = cond.last_poll_ts or cond.triggered_at or cond.created_at
        if _bar_open_ts(now, cond.timeframe) <= _bar_open_ts(last, cond.timeframe):
            return

    # NOW (post-gate) check that our plan is still alive on Bitget
    # before spending effort on cancel+replace. Cached pending list
    # lets multiple conds in the same tick share one fetch.
    try:
        pending_oids = await _get_cached_pending_oids(cond.order.exchange_mode)
        our_oid = str(cond.exchange_order_id or "")
        if our_oid and our_oid not in pending_oids:
            # Our plan is gone — filled or user cancelled. Let the
            # reconcile path handle the status transition; we do
            # NOT replan a vanished order.
            return
    except Exception as exc:
        print(f"[watcher] plan-existence check failed {cond.symbol}: {exc}", flush=True)
        # Fall through — better to attempt replan than silently stall.

    # Project the line's price at NOW. Was bar-open snap (2026-04-21
    # unification) — removed 2026-04-23 after ZEC/1d incident where
    # bar-open was 22h stale and diverged ~1.5% from the user's
    # visual reading. Same principle as place-line-order: reference
    # point = user's eye = right edge of live bar ≈ now.
    line_now = _line_price_for_replan(cond, now)

    # Line-broken invalidation (user 2026-04-22, threshold fixed 2026-04-22):
    #   If price has moved to the WRONG side of the line for the chosen
    #   direction PAST where SL would sit, the thesis is dead.
    #
    #   Mental model (from user memory): "line IS the stop". The user's
    #   own `stop_offset_pct_of_line` IS their definition of "how far past
    #   the line counts as broken". Using it as the threshold makes the
    #   cancel tight for tight setups and wide for wide setups — matches
    #   user intent. No hardcoded buffer.
    #
    #   Fallback: if stop_pct is 0 / missing, use 0.3% as a conservative
    #   default (one typical tick past the line).
    #
    #   Called only at bar-boundary (via _trailing_scheduler_loop) so
    #   intrabar wicks are already filtered.
    _user_stop_pct = max(0.0, float(cond.order.stop_offset_pct_of_line or 0.0)) / 100.0
    LINE_BROKEN_INVALID_PCT = _user_stop_pct if _user_stop_pct > 0 else 0.003

    # FAIL-CLOSED: strict mark-only fetch. If we can't get a trustworthy
    # markPrice, we can't judge line integrity safely. In that case we
    # MUST abort the whole _maybe_replan (not just skip the line-broken
    # check) — falling through to replan would re-create the exact BAS
    # 2026-04-22 failure mode where a transient ticker outage caused the
    # code to place a new trigger at the wrong semantic direction.
    mark_for_check = await _fetch_mark_price_strict(cond.symbol)
    if mark_for_check is None or mark_for_check <= 0 or line_now <= 0:
        _append_event(cond, ConditionalEvent(
            ts=now, kind="poll",
            message=f"_maybe_replan ABORTED: mark={mark_for_check} line={line_now} "
                    f"— cannot safely judge line integrity, retry next bar",
        ))
        # Update last_poll_ts so bar-gate holds until next bar
        cond.last_poll_ts = now
        try:
            _store.update(cond)
        except Exception:
            pass
        return

    if cond.order.direction == "short" and mark_for_check > line_now * (1 + LINE_BROKEN_INVALID_PCT):
        # Price above where SL would sit → short thesis invalidated.
        _break_level = line_now * (1 + LINE_BROKEN_INVALID_PCT)
        print(f"[watcher] {cond.symbol} short line-broken: mark={mark_for_check:.6f} > line*(1+stop_pct)={_break_level:.6f} (stop_pct={LINE_BROKEN_INVALID_PCT*100:.3f}%), attempting cancel", flush=True)
        outcome = await _cancel_bitget_plan_safely(cond, "line-broken-short")
        reason_text = (
            f"line broken upward, short thesis dead "
            f"(mark {mark_for_check:.6f} > line {line_now:.6f} × "
            f"(1+{LINE_BROKEN_INVALID_PCT*100:.3f}%))"
        )
        await _handle_line_broken_outcome(
            cond, outcome, reason_text, "line_broken_upward",
            mark_for_check=mark_for_check, line_now=line_now, now=now,
        )
        return
    elif cond.order.direction == "long" and mark_for_check < line_now * (1 - LINE_BROKEN_INVALID_PCT):
        _break_level = line_now * (1 - LINE_BROKEN_INVALID_PCT)
        print(f"[watcher] {cond.symbol} long line-broken: mark={mark_for_check:.6f} < line*(1-stop_pct)={_break_level:.6f} (stop_pct={LINE_BROKEN_INVALID_PCT*100:.3f}%), attempting cancel", flush=True)
        outcome = await _cancel_bitget_plan_safely(cond, "line-broken-long")
        reason_text = (
            f"line broken downward, long thesis dead "
            f"(mark {mark_for_check:.6f} < line {line_now:.6f} × "
            f"(1-{LINE_BROKEN_INVALID_PCT*100:.3f}%))"
        )
        await _handle_line_broken_outcome(
            cond, outcome, reason_text, "line_broken_downward",
            mark_for_check=mark_for_check, line_now=line_now, now=now,
        )
        return

    # Re-apply the original LINE-RELATIVE percentages so each replan tracks
    # the new line price exactly. pct-only — legacy points path deleted.
    direction = cond.order.direction
    tol_pct = cond.order.tolerance_pct_of_line or 0.0
    if tol_pct <= 0:
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_error",
            message=f"replan aborted: tolerance_pct_of_line={tol_pct} must be > 0",
        ))
        return
    new_trigger = (
        line_now * (1 + tol_pct / 100.0) if direction == "long"
        else line_now * (1 - tol_pct / 100.0)
    )

    old_trigger = cond.fill_price or 0
    if old_trigger > 0:
        drift_pct = abs(new_trigger - old_trigger) / old_trigger * 100
        if drift_pct <= 0.000001:
            cond.last_poll_ts = now
            try:
                latest = _store.get(cond.conditional_id)
                if latest is not None:
                    cond.events = latest.events
                _store.update(cond)
            except Exception:
                pass
            return

    # Cancel the old Bitget entry order. Current manual-line entries must be
    # normal_plan trigger-market orders; older records may still be regular
    # limit orders, so try both cancel paths.
    from server.execution.live_adapter import LiveExecutionAdapter
    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return
    sym = cond.symbol.upper()
    # Use _cancel_bitget_plan_safely so cancel path + classifier fallback
    # are centralized. For replan we need to distinguish CANCELLED (ok,
    # proceed with new plan) from FILLED (abort, reconcile will transition).
    # Pass the adapter we just created so tests that monkeypatch
    # LiveExecutionAdapter see a single instance with both cancel + submit.
    cancel_outcome = await _cancel_bitget_plan_safely(cond, "replan", adapter=adapter)
    if cancel_outcome == CancelOutcome.RETRY:
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_error",
            message="replan cancel failed AND classifier could not confirm terminal state; retry next bar",
        ))
        return
    if cancel_outcome == CancelOutcome.FILLED:
        # Order fired between tick and cancel. Reconcile's FILLED path
        # will transition status=filled next tick; submitting a new plan
        # here would leak a stray plan on top of an open position.
        print(f"[watcher] {sym} replan: old oid {cond.exchange_order_id} already FILLED per classifier; skipping replan (reconcile will transition)", flush=True)
        return
    # cancel_outcome == CancelOutcome.CANCELLED → ok to place new plan

    # Recompute SL / TP from the new LINE projection using line-relative pct.
    rr = cond.order.rr_target or 2.0
    stop_pct = max(0.0, float(cond.order.stop_offset_pct_of_line or 0.0))
    if direction == "long":
        new_stop = line_now * (1.0 - stop_pct / 100.0)
        risk = new_trigger - new_stop
        new_tp = new_trigger + risk * rr
    else:
        new_stop = line_now * (1.0 + stop_pct / 100.0)
        risk = new_stop - new_trigger
        new_tp = new_trigger - risk * rr

    # qty logic — user 2026-04-22 spec: "百分比策略的仓位 USD 应该跟着账户
    # 余额变化. 比如我平仓, 账户余额变了, 挂单的 qty 就应跟着涨/跌."
    #
    # - equity_pct mode: recompute qty from CURRENT account equity every
    #   replan, so size tracks equity as positions close or open elsewhere.
    # - fixed notional_usd mode: keep original qty (notional/new_trigger).
    #
    # Leverage: if set, notional = equity × pct × leverage.
    new_qty = 0.0
    equity_pct = float(cond.order.equity_pct or 0)
    leverage = float(cond.order.leverage or 0)
    if equity_pct > 0:
        # Dynamic resize on every replan
        try:
            from server.execution.live_adapter import LiveExecutionAdapter
            adapter = LiveExecutionAdapter()
            if adapter.has_api_keys():
                acct = await adapter.get_live_account_status(mode=cond.order.exchange_mode)
                equity = float(acct.get("total_equity") or acct.get("equity") or 0)
                if equity > 0:
                    lev = leverage if leverage > 0 else 1.0
                    notional = equity * (equity_pct / 100.0) * lev
                    new_qty = notional / new_trigger
                    print(f"[replan] {cond.symbol} dynamic resize: equity=${equity:.2f} "
                          f"x {equity_pct}% x {lev}x = ${notional:.2f} -> qty {new_qty:.6f}", flush=True)
        except Exception as exc:
            print(f"[replan] equity fetch err {cond.symbol}: {exc}", flush=True)

    if new_qty <= 0:
        # Fallback: preserve original qty
        if cond.fill_qty and cond.fill_qty > 0:
            new_qty = cond.fill_qty
        elif cond.order.notional_usd and cond.order.notional_usd > 0:
            new_qty = cond.order.notional_usd / new_trigger
        else:
            _append_event(cond, ConditionalEvent(
                ts=now, kind="exchange_error",
                message="replan aborted: no fill_qty/notional_usd/equity_pct to re-derive size",
            ))
            return

    # Build a fresh OrderIntent with new prices. Keep it as a Bitget
    # trigger-market plan order so it does not reserve regular order margin
    # before the line is touched.
    from server.execution.types import OrderIntent
    intent = OrderIntent(
        order_intent_id=f"replan_{cond.conditional_id}_{now}",
        signal_id="replan",
        line_id=cond.manual_line_id,
        client_order_id=f"replan_{cond.conditional_id}_{now}",
        symbol=sym,
        timeframe=cond.timeframe,
        side=direction,  # type: ignore
        order_type="market",
        trigger_mode="manual_replan",
        entry_price=float(new_trigger),
        stop_price=float(new_stop),
        tp_price=float(new_tp),
        quantity=float(new_qty),
        status="approved",
        reason="line_slope_replan",
        created_at_bar=-1,
        created_at_ts=now,
        post_only=False,
    )
    place = await adapter.submit_live_plan_entry(intent, mode=cond.order.exchange_mode, trigger_price=float(new_trigger))
    if place.get("ok"):
        new_oid = place.get("exchange_order_id") or ""
        cond.exchange_order_id = new_oid
        cond.fill_price = float(new_trigger)
        cond.last_poll_ts = now
        # Seed the cache with new_oid so a force-replan firing within
        # 3s (TTL) sees our plan as pending. We do NOT invalidate the
        # whole cache — other conds' oids on the same mode would be
        # lost until next refetch, causing false "plan gone" skips for
        # them. Old oid may linger 3s but it's harmless: cond.exchange
        # _order_id is now new_oid, so the check targets new_oid only.
        _add_pending_oid_to_cache(cond.order.exchange_mode, new_oid)
        try:
            latest = _store.get(cond.conditional_id)
            if latest is not None:
                cond.events = latest.events
            _store.update(cond)
        except Exception:
            pass
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_submitted",
            line_price=float(new_trigger),
            message=(
                f"replanned: new trigger {new_trigger:.4f} (was {old_trigger:.4f}, "
                f"drift {(abs(new_trigger-old_trigger)/old_trigger*100 if old_trigger > 0 else 0.0):.3f}%) "
                f"trigger-market plan new orderId={cond.exchange_order_id}"
            ),
        ))
    else:
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_error",
            message=f"replan submit failed: {place.get('reason','?')}",
        ))


def _expire(cond: ConditionalOrder, reason: str) -> None:
    _append_event(cond, ConditionalEvent(
        ts=now_ts(), kind="expired", message=reason,
    ))
    _store.set_status(cond.conditional_id, "cancelled", reason=f"expired: {reason}")


__all__ = ["start_watcher", "stop_watcher"]
