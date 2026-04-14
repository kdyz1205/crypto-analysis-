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
_RECONCILE_EVERY_TICKS = 30   # 30s with 1s tick
_reconcile_counter = 0

# Outcome backfill: scan pending trade snapshots and try to fill in
# exit_price / PnL / MAE / MFE from 1m K-line history. Runs much less
# often than reconcile because it loads 1m data per symbol.
_OUTCOME_BACKFILL_EVERY_TICKS = 300   # 5 minutes with 1s tick
_outcome_backfill_counter = 0

# Set of cond IDs flagged for immediate replan (e.g. user dragged a
# line's anchor). The watcher checks this set on every tick and clears
# it as entries are processed. See `force_replan(conditional_id)`.
_force_replan_set: set[str] = set()


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
    """
    global _force_replan_set
    n = 0
    try:
        for cond in _store.list_all(status="triggered", manual_line_id=manual_line_id):
            _force_replan_set.add(cond.conditional_id)
            n += 1
            # Also mirror the line anchors from the drawings store so the
            # replan actually uses the new geometry, not the snapshot that
            # was captured at cond creation time.
            try:
                from server.drawings.store import ManualTrendlineStore
                drawing = ManualTrendlineStore().get(manual_line_id)
                if drawing is not None:
                    cond.t_start = drawing.t_start
                    cond.t_end = drawing.t_end
                    cond.price_start = drawing.price_start
                    cond.price_end = drawing.price_end
                    _store.update(cond)
            except Exception as e:
                print(f"[force_replan] failed to refresh anchors for {cond.conditional_id}: {e}", flush=True)
    except Exception as e:
        print(f"[force_replan] {manual_line_id}: {e}", flush=True)
    return n


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
    if src.order.reverse_entry_offset_pct <= 0 or src.order.reverse_stop_offset_pct <= 0:
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
            stop_offset_pct_of_line=src.order.reverse_stop_offset_pct,
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
    stop_pct = float(oc.stop_offset_pct_of_line or 0.0)
    if tol_pct <= 0 or stop_pct <= 0:
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
        order_type="market",
        trigger_mode="conditional_touch",
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        tp_price=float(tp_price),
        quantity=float(qty),
        status="approved",
        reason=f"conditional {cond.conditional_id} triggered",
        created_at_bar=-1,
        created_at_ts=now_ts(),
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
async def _reconcile_against_bitget() -> None:
    """Compare local triggered conds against actual Bitget plan orders.
    Any local cond whose exchange_order_id is NOT on Bitget gets marked
    cancelled — covers user manual cancel from Bitget app, fills, expiry."""
    triggered = _store.list_all(status="triggered")
    if not triggered:
        return
    try:
        from server.execution.live_adapter import LiveExecutionAdapter
        adapter = LiveExecutionAdapter()
        if not adapter.has_api_keys():
            return
        # Fetch all live plan orders across all symbols for live mode
        resp = await adapter._bitget_request(
            "GET", "/api/v2/mix/order/orders-plan-pending",
            mode="live",
            params={"productType": "USDT-FUTURES", "planType": "normal_plan"},
        )
        if resp.get("code") != "00000":
            return
        data = resp.get("data") or {}
        rows = data.get("entrustedList") or data.get("orderList") or []
        live_oids = {str(r.get("orderId") or "") for r in rows}
        for cond in triggered:
            oid = str(cond.exchange_order_id or "")
            if not oid:
                continue
            if oid in live_oids:
                continue
            # Local says triggered with this oid but Bitget doesn't have it.
            # Could mean: user manually cancelled, filled, or (we assume)
            # the position opened and stop-loss closed it. Under the user's
            # strategy a stop-out = auto-reverse on the same line.
            print(f"[reconcile] cond {cond.conditional_id} oid={oid} not on Bitget — marking closed", flush=True)
            _store.set_status(
                cond.conditional_id, "cancelled",
                reason="reconcile: bitget order no longer exists",
            )
            try:
                _store.append_event(cond.conditional_id, ConditionalEvent(
                    ts=now_ts(), kind="cancelled",
                    message=f"reconcile: order {oid} not on Bitget anymore",
                ))
            except Exception:
                pass
            # Auto-reverse (if pre-configured).
            # NOTE: we cannot distinguish SL-fill from manual cancel here.
            # The user's policy: if they pre-set reverse_enabled on the
            # original cond, assume "gone = SL hit" and spawn the reverse.
            # They can kill the reverse manually if it was actually a cancel.
            try:
                await _spawn_reverse_conditional(cond, reason="reconcile_order_gone")
            except Exception as e:
                print(f"[reconcile] reverse spawn failed for {cond.conditional_id}: {e}", flush=True)
    except Exception as e:
        print(f"[reconcile] err: {e}", flush=True)


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
def _append_event(cond: ConditionalOrder, event: ConditionalEvent) -> None:
    _store.append_event(cond.conditional_id, event)


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

    # Force-replan flag bypasses the bar-interval gate and the drift
    # threshold — used when the user drags an anchor and we need the
    # order to immediately reflect the new geometry.
    global _force_replan_set
    forced = cond.conditional_id in _force_replan_set
    if forced:
        _force_replan_set.discard(cond.conditional_id)
    else:
        interval = _TF_REPLAN_SECONDS.get(cond.timeframe, 900)
        last = cond.last_poll_ts or cond.triggered_at or cond.created_at
        if (now - last) < interval:
            return

    # Project the line's price at NOW (clamped to anchor window)
    span = cond.t_end - cond.t_start
    if span <= 0:
        return
    if now <= cond.t_start:
        line_now = cond.price_start
    elif now >= cond.t_end:
        line_now = cond.price_end
    else:
        slope = (cond.price_end - cond.price_start) / span
        line_now = cond.price_start + slope * (now - cond.t_start)

    # Re-apply the original LINE-RELATIVE percentages so each replan tracks
    # the new line price exactly. pct-only — legacy points path deleted.
    direction = cond.order.direction
    tol_pct = cond.order.tolerance_pct_of_line or 0.0
    stop_pct = cond.order.stop_offset_pct_of_line or 0.0
    if tol_pct <= 0 or stop_pct <= 0:
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_error",
            message=f"replan aborted: tolerance_pct_of_line={tol_pct} stop_offset_pct_of_line={stop_pct} "
                    "(both must be > 0; legacy absolute-offset path is deleted)",
        ))
        return
    new_trigger = (
        line_now * (1 + tol_pct / 100.0) if direction == "long"
        else line_now * (1 - tol_pct / 100.0)
    )

    # Skip if drift below threshold (volatility-aware: wide regimes
    # tolerate larger drift before a replan, avoiding cancel/replace churn).
    try:
        from server.routers.conditionals import volatility_context
        vol_ctx = await volatility_context(cond.symbol, cond.timeframe)
        atr_pct_vol = float(vol_ctx.get("atr_pct") or 0.0)
    except Exception:
        atr_pct_vol = 0.0
    drift_threshold = max(_REPLAN_DRIFT_PCT, atr_pct_vol * 0.20)

    old_trigger = cond.fill_price or 0
    if old_trigger > 0 and not forced:
        drift_pct = abs(new_trigger - old_trigger) / old_trigger * 100
        if drift_pct < drift_threshold:
            cond.last_poll_ts = now
            try:
                latest = _store.get(cond.conditional_id)
                if latest is not None:
                    cond.events = latest.events
                _store.update(cond)
            except Exception:
                pass
            return

    # Cancel old Bitget plan
    from server.execution.live_adapter import LiveExecutionAdapter
    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return
    sym = cond.symbol.upper()
    cancel_body = {
        "symbol": sym,
        "productType": "USDT-FUTURES",
        "marginCoin": "USDT",
        "orderIdList": [{"orderId": cond.exchange_order_id}],
    }
    cancel_resp = await adapter._bitget_request(
        "POST", "/api/v2/mix/order/cancel-plan-order",
        mode=cond.order.exchange_mode, body=cancel_body,
    )
    if cancel_resp.get("code") != "00000":
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_error",
            message=f"replan cancel failed: {cancel_resp.get('msg','?')}",
        ))
        return

    # Recompute SL / TP from the new LINE projection using line-relative pct.
    rr = cond.order.rr_target or 2.0
    if direction == "long":
        new_stop = line_now * (1 - stop_pct / 100.0)
        risk = new_trigger - new_stop
        new_tp = new_trigger + risk * rr
    else:
        new_stop = line_now * (1 + stop_pct / 100.0)
        risk = new_stop - new_trigger
        new_tp = new_trigger - risk * rr

    # Re-use fill_qty from the original submit. Recomputing would require
    # an async call (leverage path fetches equity); at replan time we want
    # to track the SAME position size the trader committed to, not a
    # fresh one sized against moved equity.
    if cond.fill_qty and cond.fill_qty > 0:
        new_qty = cond.fill_qty
    elif cond.order.notional_usd and cond.order.notional_usd > 0:
        new_qty = cond.order.notional_usd / new_trigger
    else:
        _append_event(cond, ConditionalEvent(
            ts=now, kind="exchange_error",
            message="replan aborted: no fill_qty and no notional_usd to re-derive size",
        ))
        return

    # Build a fresh OrderIntent with new prices
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
        trigger_mode="manual",
        entry_price=float(new_trigger),
        stop_price=float(new_stop),
        tp_price=float(new_tp),
        quantity=float(new_qty),
        status="approved",
        reason="line_slope_replan",
        created_at_bar=-1,
        created_at_ts=now,
    )
    place = await adapter.submit_live_plan_entry(
        intent, mode=cond.order.exchange_mode,
        trigger_price=float(new_trigger), trigger_type="mark_price",
    )
    if place.get("ok"):
        cond.exchange_order_id = place.get("exchange_order_id") or ""
        cond.fill_price = float(new_trigger)
        cond.last_poll_ts = now
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
                f"drift {(abs(new_trigger-old_trigger)/old_trigger*100):.3f}%) "
                f"new orderId={cond.exchange_order_id}"
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
