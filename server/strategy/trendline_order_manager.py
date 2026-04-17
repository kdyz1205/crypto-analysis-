"""Trendline plan-order manager — persistent lines, bar-boundary updates.

Core logic (per TRENDLINE_TRADING_RULES.md):
  1. Lines are PERSISTENT once drawn. They don't disappear between scans.
  2. Each line has a plan order on Bitget (trigger = projected line + buffer).
  3. At each TF bar boundary, recalculate projection → cancel old → place new.
  4. Between boundaries, orders are untouched.
  5. A line is only REMOVED when BROKEN (price crosses through from wrong side).
  6. When plan order triggers → Bitget opens position with SL/TP attached.

Storage: data/trendline_active_orders.json (persistent across restarts)
"""
from __future__ import annotations

import time
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict

ACTIVE_LINES_FILE = Path("data/trendline_active_orders.json")

_broken_cooldowns: dict[str, float] = {}

TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
TF_PRIORITY = {"4h": 4, "1h": 3, "15m": 2, "5m": 1}


@dataclass
class ActiveLineOrder:
    symbol: str
    timeframe: str
    kind: str                   # support | resistance
    slope: float
    intercept: float
    anchor1_bar: int
    anchor2_bar: int
    bar_count: int              # total bars in data when line was found
    current_projected_price: float
    limit_price: float
    stop_price: float
    tp_price: float
    exchange_order_id: str
    created_ts: float
    last_updated_ts: float
    status: str                 # placed | filled | broken | stale
    line_ref_ts: float = 0.0     # timestamp where line_ref_price was computed
    line_ref_price: float = 0.0  # projected line price at line_ref_ts


def _load_active() -> list[ActiveLineOrder]:
    if not ACTIVE_LINES_FILE.exists():
        return []
    try:
        with open(ACTIVE_LINES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = []
        for d in data:
            d.setdefault("bar_count", 500)
            d.setdefault("line_ref_ts", d.get("last_updated_ts") or d.get("created_ts") or 0.0)
            d.setdefault("line_ref_price", d.get("current_projected_price") or 0.0)
            result.append(ActiveLineOrder(**d))
        return result
    except Exception as exc:
        print(f"[trendline_orders] active load err: {exc}", flush=True)
        traceback.print_exc()
        return []


def _save_active(orders: list[ActiveLineOrder]):
    ACTIVE_LINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_LINES_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(o) for o in orders], f, indent=2)


def _is_bar_boundary(tf: str) -> bool:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    m, s, h = now.minute, now.second, now.hour
    if tf == "5m":
        return (m % 5 == 0) and s < 90
    elif tf == "15m":
        return (m % 15 == 0) and s < 90
    elif tf == "1h":
        return m == 0 and s < 90
    elif tf == "4h":
        return (h % 4 == 0) and m == 0 and s < 90
    elif tf == "1d":
        return h == 0 and m == 0 and s < 90
    return True


def _buffer_fraction_for_tf(tf: str, cfg: dict) -> float:
    """Return buffer as a fraction. Config values are documented percentages."""
    raw_buffer_pct = float(cfg.get("buffer_pct", 0.10))
    tf_buffer_pct = float((cfg.get("tf_buffer") or {}).get(tf, raw_buffer_pct))
    return tf_buffer_pct / 100.0


def _qty_for_risk(
    *,
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    leverage: int,
    max_position_pct: float,
) -> tuple[float, float, bool]:
    stop_distance = abs(entry_price - stop_price)
    if equity <= 0 or risk_pct <= 0 or entry_price <= 0 or stop_distance <= 0:
        return 0.0, 0.0, False

    risk_usd = equity * risk_pct
    raw_qty = risk_usd / stop_distance
    max_notional = equity * max_position_pct * leverage
    if max_notional <= 0:
        return raw_qty, risk_usd, False

    max_qty = max_notional / entry_price
    capped = raw_qty > max_qty
    return min(raw_qty, max_qty), risk_usd, capped


def _would_trigger_immediately(kind: str, current_price: float, limit_price: float) -> bool:
    """A passive line order must wait for price to come to it."""
    if current_price <= 0 or limit_price <= 0:
        return False
    if kind == "support":
        return current_price <= limit_price
    return current_price >= limit_price


def _price_from_cfg(symbol: str, cfg: dict) -> float:
    prices = cfg.get("prices") or {}
    raw = prices.get(symbol) or prices.get(symbol.upper()) or prices.get(symbol.lower()) or 0
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


def _touched_status(kind: str, current_price: float, projected_price: float, limit_price: float) -> str | None:
    """Return the terminal status for a passive order that is no longer ahead of price."""
    if not _would_trigger_immediately(kind, current_price, limit_price):
        return None
    if kind == "support" and current_price <= projected_price:
        return "broken"
    if kind == "resistance" and current_price >= projected_price:
        return "broken"
    return "stale"


def _is_line_broken(kind: str, projected_price: float, current_close: float, atr: float) -> bool:
    """A line is broken when price closes decisively through it."""
    if projected_price <= 0:
        return True
    if kind == "support":
        return current_close < projected_price - 0.5 * atr
    else:  # resistance
        return current_close > projected_price + 0.5 * atr


async def update_trendline_orders(
    new_signals: list[dict],
    current_bar_index: int,
    cfg: dict,
):
    """Called each scan cycle.

    Two jobs:
      A. Add NEW lines (from new_signals) that we don't already track.
      B. For EXISTING lines: at bar boundary → update order coordinates.
         Between boundaries → do nothing. If broken → cancel + remove.
    """
    from server.execution.live_adapter import LiveExecutionAdapter
    from server.execution.types import OrderIntent

    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return {"placed": 0, "updated": 0, "cancelled": 0}

    active = _load_active()
    placed = updated = cancelled = 0
    now = time.time()
    rr = cfg.get("rr", 15.0)
    leverage = int(cfg.get("leverage", 30))
    equity = float(cfg.get("equity", 50.0))
    risk_pct = float(cfg.get("risk_pct", 0.03))
    max_position_pct = float(cfg.get("max_position_pct", 0.50))
    mode = cfg.get("mode", "live")
    held_symbols = {str(s).upper() for s in (cfg.get("held_symbols") or set())}
    position_guard_ok = True
    try:
        held_symbols |= await adapter.get_open_position_symbols(mode)
    except Exception as exc:
        position_guard_ok = False
        print(f"[trendline_orders] position guard failed; skipping new orders: {exc}", flush=True)

    pending_normal_ids_by_symbol: dict[str, set[str]] = {}
    pending_sync_ok = False
    try:
        pending_rows = await adapter.get_pending_plan_orders(mode, plan_type="normal_plan")
        for row in pending_rows:
            row_symbol = str(row.get("symbol") or "").upper()
            row_id = str(row.get("orderId") or row.get("order_id") or "")
            if row_symbol and row_id:
                pending_normal_ids_by_symbol.setdefault(row_symbol, set()).add(row_id)
        pending_sync_ok = True
    except Exception as exc:
        print(f"[trendline_orders] pending normal_plan sync failed: {exc}", flush=True)

    local_placed_ids = {
        str(o.exchange_order_id or "")
        for o in active
        if o.status == "placed" and str(o.exchange_order_id or "")
    }
    if pending_sync_ok:
        for row in pending_rows:
            row_id = str(row.get("orderId") or row.get("order_id") or "")
            row_symbol = str(row.get("symbol") or "").upper()
            client_oid = str(row.get("clientOid") or "")
            if not row_id or not row_symbol or row_id in local_placed_ids:
                continue
            if not client_oid.startswith("tl_"):
                continue
            cancel_resp = await adapter.cancel_plan_order(
                row_symbol,
                row_id,
                mode,
                plan_type="normal_plan",
            )
            if cancel_resp.get("ok"):
                cancelled += 1
                pending_normal_ids_by_symbol.get(row_symbol, set()).discard(row_id)
                print(
                    f"[trendline_orders] ORPHAN-CANCEL {row_symbol}: "
                    f"exchange normal_plan order_id={row_id} not managed locally",
                    flush=True,
                )
            else:
                print(f"[trendline_orders] orphan cancel {row_symbol} failed: {cancel_resp}", flush=True)

    # Index only exchange-live local orders by (symbol, tf, kind). Historical
    # stale/filled records are kept for evidence but must not block fresh lines.
    inactive_orders = [o for o in active if o.status != "placed"]
    existing = {}
    for o in active:
        if o.status == "placed":
            existing[(o.symbol, o.timeframe, o.kind)] = o
    tracked_symbols = {
        o.symbol.upper()
        for o in active
        if o.status == "placed"
    } | held_symbols

    # --- A. Add new lines not already tracked ---
    sorted_signals = sorted(
        new_signals,
        key=lambda s: (TF_PRIORITY.get(str(s.get("timeframe", "")), 0), str(s.get("symbol", ""))),
        reverse=True,
    )
    new_keys: set[tuple[str, str, str]] = set()
    for sig in sorted_signals:
        if not position_guard_ok:
            break
        key = (sig["symbol"], sig["timeframe"], sig["kind"])
        if key in existing:
            continue  # already tracking this line

        sym, tf, kind = key
        sym_upper = sym.upper()
        if sym_upper in tracked_symbols:
            print(f"[trendline_orders] SKIP {sym} {tf}: symbol already has active/order position", flush=True)
            continue

        slope = sig["slope"]
        intercept = sig["intercept"]
        bar_count = sig.get("bar_count", 500)
        proj = slope * (bar_count - 1) + intercept
        if proj <= 0:
            continue

        # No cooldown needed — plan order + preset SL handles broken lines

        # Use per-TF buffer from backtest: 5m=0.05%, 15m=0.10%, 1h=0.20%, 4h=0.30%
        buffer_pct = _buffer_fraction_for_tf(tf, cfg)

        if kind == "support":
            limit_px = proj * (1 + buffer_pct)
            stop_px = proj
            tp_px = limit_px * (1 + buffer_pct * rr)
            direction = "long"
        else:
            limit_px = proj * (1 - buffer_pct)
            stop_px = proj
            tp_px = limit_px * (1 - buffer_pct * rr)
            direction = "short"

        current_close = _price_from_cfg(sym_upper, cfg) or float(sig.get("entry_price", 0) or 0)
        if _would_trigger_immediately(kind, float(current_close or 0), limit_px):
            print(f"[trendline_orders] SKIP {sym} {tf}: current={float(current_close):.8f} would trigger entry={limit_px:.8f}", flush=True)
            continue

        per_tf_risk = float((cfg.get("tf_risk") or {}).get(tf, risk_pct))
        qty, risk_usd, capped = _qty_for_risk(
            equity=equity,
            risk_pct=per_tf_risk,
            entry_price=limit_px,
            stop_price=stop_px,
            leverage=leverage,
            max_position_pct=max_position_pct,
        )
        if qty <= 0:
            continue

        # Place plan order
        try:
            try:
                await adapter._bitget_request(
                    "POST", "/api/v2/mix/account/set-leverage",
                    mode=mode,
                    body={"symbol": sym.upper(), "productType": "USDT-FUTURES",
                          "marginCoin": "USDT", "leverage": str(leverage)},
                )
            except Exception as exc:
                print(f"[trendline_orders] set-leverage {sym} warn: {exc}", flush=True)

            now_ts = int(time.time())
            intent = OrderIntent(
                order_intent_id=f"tl_{sym}_{now_ts}",
                signal_id=f"tl_sig_{now_ts}",
                line_id="",
                client_order_id=f"tl_{sym[:10]}_{now_ts}",
                symbol=sym.upper(), timeframe=tf, side=direction,
                order_type="limit", trigger_mode="plan",
                entry_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                quantity=qty, status="approved",
                reason="trendline_plan_order",
                created_at_bar=bar_count - 1, created_at_ts=now_ts,
            )
            resp = await adapter.submit_live_plan_entry(intent, mode=mode, trigger_price=limit_px)

            if resp.get("ok"):
                order_id = resp.get("exchange_order_id", "")
                existing[key] = ActiveLineOrder(
                    symbol=sym, timeframe=tf, kind=kind,
                    slope=slope, intercept=intercept,
                    anchor1_bar=sig.get("anchor1_bar", 0),
                    anchor2_bar=sig.get("anchor2_bar", 0),
                    bar_count=bar_count,
                    current_projected_price=proj,
                    limit_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                    exchange_order_id=order_id,
                    created_ts=now, last_updated_ts=now, status="placed",
                    line_ref_ts=now, line_ref_price=proj,
                )
                new_keys.add(key)
                tracked_symbols.add(sym_upper)
                placed += 1
                print(f"[trendline_orders] NEW {direction} {sym} {tf} @ {limit_px:.6f} "
                      f"SL={stop_px:.6f} TP={tp_px:.6f} risk=${risk_usd:.4f}"
                      f"{' capped' if capped else ''}", flush=True)
                try:
                    from server.strategy.trade_log import log_trade
                    log_trade(
                        symbol=sym, timeframe=tf, strategy="trendline",
                        direction=direction, entry_price=limit_px,
                        stop_price=stop_px, tp_price=tp_px,
                        size_usd=qty * limit_px, leverage=leverage,
                        order_id=order_id, status="plan_placed",
                        buffer_pct=buffer_pct, risk_usd=risk_usd,
                    )
                    from server.strategy.ml_trade_db import log_plan_placed
                    log_plan_placed(
                        symbol=sym, tf=tf, direction=direction, kind=kind,
                        slope=slope, intercept=intercept, bar_count=bar_count,
                        entry_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                        buffer_pct=buffer_pct, rr=rr, leverage=leverage,
                        quantity=qty, order_id=order_id,
                        pivot1_bar=sig.get("anchor1_bar", 0),
                        pivot2_bar=sig.get("anchor2_bar", 0),
                        risk_usd=risk_usd, capped=capped,
                    )
                except Exception as exc:
                    print(f"[trendline_orders] plan log err {sym}: {exc}", flush=True)
            else:
                print(f"[trendline_orders] REJECTED {sym}: {resp.get('reason')}", flush=True)
        except Exception as e:
            print(f"[trendline_orders] place {sym} err: {e}", flush=True)

    # --- B. Update existing lines at bar boundaries / remove broken ---
    surviving = list(inactive_orders)
    for key, order in existing.items():
        if order.status != "placed":
            surviving.append(order)
            continue
        if key in new_keys:
            surviving.append(order)
            continue

        if order.symbol.upper() in held_symbols:
            order.status = "filled"
            surviving.append(order)
            print(f"[trendline_orders] FILLED-SYNC {order.symbol} {order.timeframe}: held position detected", flush=True)
            continue

        if pending_sync_ok:
            live_ids = pending_normal_ids_by_symbol.get(order.symbol.upper(), set())
            order_id = str(order.exchange_order_id or "")
            if order_id and order_id not in live_ids:
                order.status = "stale"
                surviving.append(order)
                print(
                    f"[trendline_orders] STALE {order.symbol} {order.timeframe}: "
                    f"exchange missing normal_plan order_id={order_id}; local line disabled",
                    flush=True,
                )
                continue

        tf = order.timeframe
        # Recalculate projection (bar_count + 1 for each elapsed bar)
        # Estimate: bars elapsed = time since last update / bar duration
        bar_dur = TF_SECONDS.get(tf, 300)
        bars_elapsed = max(0, int((now - order.last_updated_ts) / bar_dur))
        new_bar_index = order.bar_count - 1 + bars_elapsed
        proj = order.slope * new_bar_index + order.intercept

        if proj <= 0:
            continue

        # No broken-line detection needed — if price crosses through,
        # the plan order triggers and preset SL handles it automatically.

        # Recalculate target coordinates first. Broken/touched orders must be
        # cancelled immediately; only healthy orders wait for the next TF bar
        # before moving.
        buffer_pct = _buffer_fraction_for_tf(tf, cfg)

        if order.kind == "support":
            limit_px = proj * (1 + buffer_pct)
            stop_px = proj
            tp_px = limit_px * (1 + buffer_pct * rr)
            direction = "long"
        else:
            limit_px = proj * (1 - buffer_pct)
            stop_px = proj
            tp_px = limit_px * (1 - buffer_pct * rr)
            direction = "short"

        current_price = _price_from_cfg(order.symbol, cfg)
        touched_status = _touched_status(order.kind, current_price, proj, limit_px)
        if touched_status:
            try:
                cancel_resp = await adapter.cancel_plan_order(
                    order.symbol.upper(),
                    order.exchange_order_id,
                    mode,
                    plan_type="normal_plan",
                )
                if cancel_resp.get("ok"):
                    order.status = touched_status
                    surviving.append(order)
                    cancelled += 1
                    print(
                        f"[trendline_orders] CANCEL {order.symbol} {tf}: "
                        f"current={current_price:.8f} already touched entry={limit_px:.8f}; "
                        f"status={touched_status}",
                        flush=True,
                    )
                    continue
                print(f"[trendline_orders] cancel touched {order.symbol} failed: {cancel_resp}", flush=True)
            except Exception as e:
                print(f"[trendline_orders] cancel touched {order.symbol} err: {e}", flush=True)
            surviving.append(order)
            continue

        # Not touched/broken -> move only after at least one full TF bar elapsed.
        # This avoids missing the 90s wall-clock window and prevents repeated
        # cancel/re-place loops inside the same boundary window.
        if bars_elapsed <= 0:
            surviving.append(order)
            continue

        # Bar boundary -> cancel old + place new at updated coordinates

        # Cancel old
        try:
            cancel_resp = await adapter.cancel_plan_order(
                order.symbol.upper(),
                order.exchange_order_id,
                mode,
                plan_type="normal_plan",
            )
            if not cancel_resp.get("ok"):
                print(f"[trendline_orders] cancel {order.symbol} failed: {cancel_resp}", flush=True)
                surviving.append(order)
                continue
        except Exception as e:
            print(f"[trendline_orders] cancel {order.symbol} err: {e}", flush=True)
            surviving.append(order)
            continue

        # Place new
        try:
            per_tf_risk = float((cfg.get("tf_risk") or {}).get(tf, risk_pct))
            qty, risk_usd, capped = _qty_for_risk(
                equity=equity,
                risk_pct=per_tf_risk,
                entry_price=limit_px,
                stop_price=stop_px,
                leverage=leverage,
                max_position_pct=max_position_pct,
            )
            if qty <= 0:
                continue

            now_ts = int(time.time())
            intent = OrderIntent(
                order_intent_id=f"tl_{order.symbol}_{now_ts}",
                signal_id=f"tl_sig_{now_ts}", line_id="",
                client_order_id=f"tl_{order.symbol[:10]}_{now_ts}",
                symbol=order.symbol.upper(), timeframe=tf, side=direction,
                order_type="limit", trigger_mode="plan",
                entry_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                quantity=qty, status="approved",
                reason="trendline_plan_order",
                created_at_bar=new_bar_index, created_at_ts=now_ts,
            )
            resp = await adapter.submit_live_plan_entry(intent, mode=mode, trigger_price=limit_px)

            if resp.get("ok"):
                order.exchange_order_id = resp.get("exchange_order_id", "")
                order.limit_price = limit_px
                order.stop_price = stop_px
                order.tp_price = tp_px
                order.current_projected_price = proj
                order.last_updated_ts = now
                order.bar_count = new_bar_index + 1
                order.line_ref_ts = now
                order.line_ref_price = proj
                surviving.append(order)
                updated += 1
                print(f"[trendline_orders] MOVED {direction} {order.symbol} {tf} @ {limit_px:.6f} "
                      f"SL={stop_px:.6f} risk=${risk_usd:.4f}{' capped' if capped else ''}", flush=True)
            else:
                print(f"[trendline_orders] MOVE FAILED {order.symbol}: {resp.get('reason')}", flush=True)
                surviving.append(order)  # keep tracking even if place failed
        except Exception as e:
            print(f"[trendline_orders] update {order.symbol} err: {e}", flush=True)
            surviving.append(order)

    _save_active(surviving)
    return {"placed": placed, "updated": updated, "cancelled": cancelled}


async def cancel_all_trendline_plan_orders(cfg: dict, *, status: str = "cancelled") -> dict:
    """Cancel every exchange-live trendline plan order managed by this system.

    Used by risk halts. It cancels local active orders plus exchange orphan
    `tl_` plan orders, but leaves manual non-`tl_` plan orders untouched.
    """
    from server.execution.live_adapter import LiveExecutionAdapter

    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return {"cancelled": 0, "failed": 0, "reason": "api_keys_missing"}

    mode = cfg.get("mode", "live")
    active = _load_active()
    cancelled = failed = 0
    cancelled_ids: set[str] = set()
    pending_rows: list[dict] = []

    try:
        pending_rows = await adapter.get_pending_plan_orders(mode, plan_type="normal_plan")
    except Exception as exc:
        print(f"[trendline_orders] halt pending sync failed: {exc}", flush=True)

    pending_ids = {
        str(row.get("orderId") or row.get("order_id") or "")
        for row in pending_rows
        if str(row.get("orderId") or row.get("order_id") or "")
    }

    surviving: list[ActiveLineOrder] = []
    for order in active:
        if order.status != "placed":
            surviving.append(order)
            continue

        order_id = str(order.exchange_order_id or "")
        if pending_ids and order_id and order_id not in pending_ids:
            order.status = "stale"
            surviving.append(order)
            continue

        try:
            cancel_resp = await adapter.cancel_plan_order(
                order.symbol.upper(),
                order_id,
                mode,
                plan_type="normal_plan",
            )
            if cancel_resp.get("ok"):
                order.status = status
                cancelled += 1
                cancelled_ids.add(order_id)
                print(
                    f"[trendline_orders] HALT-CANCEL {order.symbol} {order.timeframe}: "
                    f"order_id={order_id} status={status}",
                    flush=True,
                )
            else:
                failed += 1
                print(f"[trendline_orders] halt cancel {order.symbol} failed: {cancel_resp}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[trendline_orders] halt cancel {order.symbol} err: {exc}", flush=True)
        surviving.append(order)

    for row in pending_rows:
        row_id = str(row.get("orderId") or row.get("order_id") or "")
        if not row_id or row_id in cancelled_ids:
            continue
        client_oid = str(row.get("clientOid") or "")
        if not client_oid.startswith("tl_"):
            continue
        row_symbol = str(row.get("symbol") or "").upper()
        if not row_symbol:
            continue
        try:
            cancel_resp = await adapter.cancel_plan_order(
                row_symbol,
                row_id,
                mode,
                plan_type="normal_plan",
            )
            if cancel_resp.get("ok"):
                cancelled += 1
                cancelled_ids.add(row_id)
                print(
                    f"[trendline_orders] HALT-ORPHAN-CANCEL {row_symbol}: "
                    f"order_id={row_id} status={status}",
                    flush=True,
                )
            else:
                failed += 1
                print(f"[trendline_orders] halt orphan cancel {row_symbol} failed: {cancel_resp}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[trendline_orders] halt orphan cancel {row_symbol} err: {exc}", flush=True)

    _save_active(surviving)
    return {"cancelled": cancelled, "failed": failed, "status": status}
