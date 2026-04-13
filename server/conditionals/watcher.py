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
# decide when to actually re-poll based on their own poll_seconds.
_LOOP_TICK_SECONDS = 5


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
    """One pass through all pending conditionals."""
    pending = _store.list_all(status="pending")
    if not pending:
        return
    now = now_ts()
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

    atr = await _fetch_current_atr(cond.symbol, cond.timeframe)
    if atr is None or atr <= 0:
        atr = market_price * 0.005  # 0.5% fallback

    line_price = cond.line_price_at(now)
    distance = abs(market_price - line_price)
    distance_atr = distance / atr if atr > 0 else 0.0

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
            await _submit_exchange(cond, market_price, atr)
        except Exception as e:
            print(f"[conditional_watcher] exchange submit failed: {e}", flush=True)
            traceback.print_exc()
            _append_event(cond, ConditionalEvent(
                ts=now_ts(), kind="exchange_error",
                message=f"submit failed: {e}",
            ))
            _store.set_status(cond.conditional_id, "failed", reason=f"submit error: {e}")


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


async def _submit_exchange(cond: ConditionalOrder, market_price: float, atr: float) -> None:
    """Actually place the order on the exchange (paper or live)."""
    # Compute quantity
    qty = _compute_qty(cond, market_price, atr)
    if qty is None or qty <= 0:
        raise ValueError(f"invalid qty: {qty}")

    _append_event(cond, ConditionalEvent(
        ts=now_ts(), kind="exchange_submitted",
        price=market_price,
        message=f"submitting {cond.order.direction} {qty:.6f} "
                f"{cond.symbol} mode={cond.order.exchange_mode}",
    ))

    # Route to appropriate engine
    if cond.order.exchange_mode == "live":
        await _submit_live(cond, qty, market_price, atr)
    else:
        await _submit_paper(cond, qty, market_price, atr)


async def _submit_live(cond, qty, market_price, atr):
    """Submit to Bitget via the existing LiveExecutionAdapter.submit_live_entry.

    Constructs an OrderIntent from the conditional's order config and
    calls into the SAME adapter the Phase 1 live execution uses, so
    this path shares all the contract-resolution, size-normalization,
    and error-handling logic already validated there.
    """
    from server.execution.live_adapter import LiveExecutionAdapter
    from server.execution.types import OrderIntent, stable_execution_id

    # Direction-adjusted entry / stop / tp prices
    dir_sign = 1 if cond.order.direction == "long" else -1
    entry_offset = cond.order.entry_offset_atr * atr * dir_sign
    entry_price = market_price + entry_offset
    stop_distance = cond.order.stop_atr * atr
    stop_price = entry_price - stop_distance * dir_sign
    if cond.order.tp_price is not None:
        tp_price = float(cond.order.tp_price)
    elif cond.order.rr_target is not None:
        tp_price = entry_price + cond.order.rr_target * stop_distance * dir_sign
    else:
        tp_price = 0.0

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
        # Persist exchange_order_id on the conditional for audit
        cond.exchange_order_id = exchange_id
        cond.fill_price = float(resp.get("submitted_price") or market_price)
        cond.fill_qty = float(qty)
        try:
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


async def _submit_paper(cond, qty, market_price, atr):
    """Submit to the paper engine — reuses the existing runtime."""
    try:
        from server.execution.engine import PaperExecutionEngine  # type: ignore
        # Just log — the paper engine doesn't take external intents;
        # for now we record the "intended fill" as a synthetic event.
        _append_event(cond, ConditionalEvent(
            ts=now_ts(), kind="exchange_acked",
            message=(
                f"[paper] would submit: {cond.order.direction} {qty:.6f} "
                f"{cond.symbol} @ {market_price:.4f} (entry_offset={cond.order.entry_offset_atr} ATR, "
                f"stop={cond.order.stop_atr} ATR, RR={cond.order.rr_target})"
            ),
            extra={
                "intent": {
                    "direction": cond.order.direction,
                    "qty": qty,
                    "entry_price": market_price,
                    "stop_price": market_price - cond.order.stop_atr * atr if cond.order.direction == "long" else market_price + cond.order.stop_atr * atr,
                    "rr_target": cond.order.rr_target,
                }
            },
        ))
    except ImportError:
        _append_event(cond, ConditionalEvent(
            ts=now_ts(), kind="exchange_acked",
            message="[paper] engine not available — trigger logged only",
        ))


def _compute_qty(cond: ConditionalOrder, market_price: float, atr: float) -> float | None:
    """Derive qty from order config."""
    oc = cond.order
    if oc.notional_usd and oc.notional_usd > 0:
        return oc.notional_usd / market_price
    if oc.equity_pct and oc.equity_pct > 0:
        # Default equity estimate if we can't read real balance
        equity = 10_000.0  # TODO: read from paper_execution or live account
        return (equity * oc.equity_pct) / market_price
    if oc.risk_pct and oc.risk_pct > 0:
        equity = 10_000.0
        stop_distance = oc.stop_atr * atr
        if stop_distance <= 0:
            return None
        risk_usd = equity * oc.risk_pct
        return risk_usd / stop_distance
    return None


# ─────────────────────────────────────────────────────────────
# Price + ATR fetchers
# ─────────────────────────────────────────────────────────────
async def _fetch_market_price(symbol: str) -> float | None:
    """Latest price for a symbol. Tries /api/ohlcv tail first, then ticker."""
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
        print(f"[conditional_watcher] price fetch err {symbol}: {e}", flush=True)
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
    # Mutate in-memory then persist via store.update
    cond.last_poll_ts = now
    cond.last_market_price = market_price
    cond.last_line_price = line_price
    cond.last_distance_atr = distance_atr
    try:
        _store.update(cond)
    except ValueError:
        # Conditional may have been deleted mid-poll — safe to ignore
        pass


def _expire(cond: ConditionalOrder, reason: str) -> None:
    _append_event(cond, ConditionalEvent(
        ts=now_ts(), kind="expired", message=reason,
    ))
    _store.set_status(cond.conditional_id, "cancelled", reason=f"expired: {reason}")


__all__ = ["start_watcher", "stop_watcher"]
