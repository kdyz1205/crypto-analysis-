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

import datetime
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

ACTIVE_LINES_FILE = Path("data/trendline_active_orders.json")


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
    status: str                 # placed | filled | broken


def _load_active() -> list[ActiveLineOrder]:
    if not ACTIVE_LINES_FILE.exists():
        return []
    try:
        with open(ACTIVE_LINES_FILE, "r") as f:
            data = json.load(f)
        result = []
        for d in data:
            d.setdefault("bar_count", 500)
            result.append(ActiveLineOrder(**d))
        return result
    except Exception:
        return []


def _save_active(orders: list[ActiveLineOrder]):
    ACTIVE_LINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_LINES_FILE, "w") as f:
        json.dump([asdict(o) for o in orders], f, indent=2)


def _is_bar_boundary(tf: str) -> bool:
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
    raw_buffer_pct = cfg.get("buffer_pct", 0.12) / 100
    rr = cfg.get("rr", 12.0)
    leverage = int(cfg.get("leverage", 30))
    equity = cfg.get("equity", 50.0)
    risk_pct = cfg.get("risk_pct", 0.03)

    # Index existing by (symbol, tf, kind)
    existing = {}
    for o in active:
        existing[(o.symbol, o.timeframe, o.kind)] = o

    # --- A. Add new lines not already tracked ---
    for sig in new_signals:
        key = (sig["symbol"], sig["timeframe"], sig["kind"])
        if key in existing:
            continue  # already tracking this line

        sym, tf, kind = key
        slope = sig["slope"]
        intercept = sig["intercept"]
        bar_count = sig.get("bar_count", 500)
        proj = slope * (bar_count - 1) + intercept
        if proj <= 0:
            continue

        # Filter: line too close to current price → will immediately trigger or break
        current_close = cfg.get("prices", {}).get(sym, 0)
        if current_close > 0:
            dist = abs(proj - current_close) / current_close
            if dist < 0.003:  # < 0.3% away → skip
                continue

        atr_ratio = sig.get("atr_pct", 0)
        buffer_pct = max(0.0010, min(0.0015, 0.3 * atr_ratio)) if atr_ratio > 0 else raw_buffer_pct

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

        risk_usd = equity * risk_pct
        stop_distance = abs(limit_px - stop_px)
        if stop_distance <= 0:
            continue
        qty = risk_usd / stop_distance

        # Place plan order
        try:
            try:
                await adapter._bitget_request(
                    "POST", "/api/v2/mix/account/set-leverage",
                    mode="crossed",
                    body={"symbol": sym.upper(), "productType": "USDT-FUTURES",
                          "marginCoin": "USDT", "leverage": str(leverage)},
                )
            except Exception:
                pass

            now_ts = int(time.time())
            intent = OrderIntent(
                order_intent_id=f"tl_{sym}_{now_ts}",
                signal_id=f"tl_sig_{now_ts}",
                line_id="",
                client_order_id=f"tl_{sym[:10]}_{now_ts}",
                symbol=sym.upper(), timeframe=tf, side=direction,
                order_type="market", trigger_mode="plan",
                entry_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                quantity=qty, status="approved",
                reason="trendline_plan_order",
                created_at_bar=bar_count - 1, created_at_ts=now_ts,
            )
            resp = await adapter.submit_live_plan_entry(intent, mode="crossed", trigger_price=limit_px)

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
                )
                placed += 1
                print(f"[trendline_orders] NEW {direction} {sym} {tf} @ {limit_px:.6f} "
                      f"SL={stop_px:.6f} TP={tp_px:.6f}", flush=True)
            else:
                print(f"[trendline_orders] REJECTED {sym}: {resp.get('reason')}", flush=True)
        except Exception as e:
            print(f"[trendline_orders] place {sym} err: {e}", flush=True)

    # --- B. Update existing lines at bar boundaries / remove broken ---
    surviving = []
    for key, order in existing.items():
        if order.status != "placed":
            continue

        tf = order.timeframe
        # Recalculate projection (bar_count + 1 for each elapsed bar)
        # Estimate: bars elapsed = time since last update / bar duration
        tf_seconds = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        bar_dur = tf_seconds.get(tf, 300)
        bars_elapsed = max(0, int((now - order.last_updated_ts) / bar_dur))
        new_bar_index = order.bar_count - 1 + bars_elapsed
        proj = order.slope * new_bar_index + order.intercept

        if proj <= 0:
            continue

        # Check if line is broken
        current_close = cfg.get("prices", {}).get(order.symbol, 0)
        if current_close > 0 and _is_line_broken(order.kind, proj, current_close, proj * 0.005):
            # Line broken → cancel order → remove
            try:
                await adapter._bitget_request(
                    "POST", "/api/v2/mix/order/cancel-plan-order",
                    mode="crossed",
                    body={"symbol": order.symbol.upper(),
                          "productType": "USDT-FUTURES",
                          "orderId": order.exchange_order_id},
                )
                cancelled += 1
                print(f"[trendline_orders] BROKEN {order.kind} {order.symbol} {tf} — cancelled", flush=True)
            except Exception:
                pass
            continue  # don't keep this line

        # Not broken → check if bar boundary for update
        if not _is_bar_boundary(tf):
            surviving.append(order)
            continue

        # Bar boundary → cancel old + place new at updated coordinates
        atr_ratio = 0  # TODO: pass ATR from scan
        buffer_pct = raw_buffer_pct

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

        # Cancel old
        try:
            await adapter._bitget_request(
                "POST", "/api/v2/mix/order/cancel-plan-order",
                mode="crossed",
                body={"symbol": order.symbol.upper(),
                      "productType": "USDT-FUTURES",
                      "orderId": order.exchange_order_id},
            )
            # Cancel orphaned SL/TP
            try:
                pending = await adapter._bitget_request(
                    "GET", "/api/v2/mix/order/orders-plan-pending",
                    mode="crossed", body=None,
                    params={"symbol": order.symbol.upper(),
                            "productType": "USDT-FUTURES",
                            "planType": "profit_loss"},
                )
                for orphan in ((pending.get("data") or {}).get("entrustedList") or []):
                    oid = orphan.get("orderId")
                    if oid:
                        await adapter._bitget_request(
                            "POST", "/api/v2/mix/order/cancel-plan-order",
                            mode="crossed",
                            body={"symbol": order.symbol.upper(),
                                  "productType": "USDT-FUTURES", "orderId": oid},
                        )
            except Exception:
                pass
        except Exception as e:
            print(f"[trendline_orders] cancel {order.symbol} err: {e}", flush=True)

        # Place new
        try:
            risk_usd = equity * risk_pct
            stop_distance = abs(limit_px - stop_px)
            if stop_distance <= 0:
                continue
            qty = risk_usd / stop_distance

            now_ts = int(time.time())
            intent = OrderIntent(
                order_intent_id=f"tl_{order.symbol}_{now_ts}",
                signal_id=f"tl_sig_{now_ts}", line_id="",
                client_order_id=f"tl_{order.symbol[:10]}_{now_ts}",
                symbol=order.symbol.upper(), timeframe=tf, side=direction,
                order_type="market", trigger_mode="plan",
                entry_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                quantity=qty, status="approved",
                reason="trendline_plan_order",
                created_at_bar=new_bar_index, created_at_ts=now_ts,
            )
            resp = await adapter.submit_live_plan_entry(intent, mode="crossed", trigger_price=limit_px)

            if resp.get("ok"):
                order.exchange_order_id = resp.get("exchange_order_id", "")
                order.limit_price = limit_px
                order.stop_price = stop_px
                order.tp_price = tp_px
                order.current_projected_price = proj
                order.last_updated_ts = now
                order.bar_count = new_bar_index + 1
                surviving.append(order)
                updated += 1
                print(f"[trendline_orders] MOVED {direction} {order.symbol} {tf} @ {limit_px:.6f} "
                      f"SL={stop_px:.6f}", flush=True)
            else:
                print(f"[trendline_orders] MOVE FAILED {order.symbol}: {resp.get('reason')}", flush=True)
                surviving.append(order)  # keep tracking even if place failed
        except Exception as e:
            print(f"[trendline_orders] update {order.symbol} err: {e}", flush=True)
            surviving.append(order)

    _save_active(surviving)
    return {"placed": placed, "updated": updated, "cancelled": cancelled}
