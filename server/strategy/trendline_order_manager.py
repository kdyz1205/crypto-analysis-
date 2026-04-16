"""Trendline limit-order manager — passive entry via resting limit orders.

Instead of market-ordering when price is near a line, this module:
  1. Maintains a set of active trendlines (from trendline_strategy signals)
  2. For each active line, computes the projected price at the current bar
  3. Places/updates a LIMIT ORDER at projected_price ± buffer on Bitget
  4. Every scan cycle, recalculates and moves orders as lines shift
  5. When a line breaks (price crosses from wrong side), cancels the order
  6. After fill: SL = line price (穿线即止损), TP = buffer × RR

Called from mar_bb_runner's scan loop.
"""
from __future__ import annotations

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
    current_projected_price: float
    limit_price: float
    stop_price: float
    tp_price: float
    exchange_order_id: str      # Bitget order ID (empty if not yet placed)
    created_ts: float
    last_updated_ts: float
    status: str                 # pending | placed | filled | cancelled


def _load_active() -> list[ActiveLineOrder]:
    if not ACTIVE_LINES_FILE.exists():
        return []
    with open(ACTIVE_LINES_FILE, "r") as f:
        return [ActiveLineOrder(**d) for d in json.load(f)]


def _save_active(orders: list[ActiveLineOrder]):
    ACTIVE_LINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_LINES_FILE, "w") as f:
        json.dump([asdict(o) for o in orders], f, indent=2)


async def update_trendline_orders(
    trendline_signals: list[dict],
    current_bar_index: int,
    cfg: dict,
):
    """Called each scan cycle. Manages limit orders for trendline setups.

    trendline_signals: list of dicts with keys:
      symbol, timeframe, kind, slope, intercept, anchor1_bar, anchor2_bar,
      entry_price, stop_price, tp_price, direction

    For each signal:
      - If no existing order for this line → place new limit order
      - If existing order but price moved → cancel old + place new
      - If line broken → cancel order
    """
    from server.execution.live_adapter import LiveExecutionAdapter
    from server.execution.types import OrderIntent

    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return {"placed": 0, "updated": 0, "cancelled": 0}

    active = _load_active()
    placed = updated = cancelled = 0
    now = time.time()
    # Dynamic buffer: 0.3 × ATR/price, clamped to [0.10%, 0.15%]
    # Higher-vol coins get slightly wider buffer, low-vol coins get tighter
    raw_buffer_pct = cfg.get("buffer_pct", 0.12) / 100  # fallback 0.12%
    rr = cfg.get("rr", 8.0)
    leverage = int(cfg.get("leverage", 20))

    # Build lookup of existing orders by (symbol, tf, kind, anchor key)
    existing = {}
    for o in active:
        key = (o.symbol, o.timeframe, o.kind, o.anchor1_bar, o.anchor2_bar)
        existing[key] = o

    new_active = []

    for sig in trendline_signals:
        sym = sig["symbol"]
        tf = sig["timeframe"]
        kind = sig["kind"]
        slope = sig["slope"]
        intercept = sig["intercept"]
        a1 = sig.get("anchor1_bar", 0)
        a2 = sig.get("anchor2_bar", 0)

        # Project line to current bar
        proj = slope * current_bar_index + intercept
        if proj <= 0:
            continue

        # Dynamic buffer per coin: use ATR if available, else fixed
        # ATR-based: buffer = 0.3 × ATR/price, clamped [0.10%, 0.15%]
        atr_ratio = sig.get("atr_pct", 0)
        if atr_ratio > 0:
            buffer_pct = max(0.0010, min(0.0015, 0.3 * atr_ratio))
        else:
            buffer_pct = raw_buffer_pct

        # Calculate order prices
        if kind == "support":
            limit_px = proj * (1 + buffer_pct)   # buy above line
            stop_px = proj                        # stop = line
            tp_px = limit_px * (1 + buffer_pct * rr)
            direction = "long"
        else:
            limit_px = proj * (1 - buffer_pct)   # sell below line
            stop_px = proj                        # stop = line
            tp_px = limit_px * (1 - buffer_pct * rr)
            direction = "short"

        key = (sym, tf, kind, a1, a2)
        old = existing.pop(key, None)

        if old and old.status == "placed" and old.exchange_order_id:
            # Always cancel + re-place on every new bar — line moved = order must move
            try:
                await adapter._bitget_request(
                    "POST", "/api/v2/mix/order/cancel-plan-order",
                    mode="crossed",
                    body={
                        "symbol": sym.upper(),
                        "productType": "USDT-FUTURES",
                        "orderId": old.exchange_order_id,
                    },
                )
                cancelled += 1
                # Also cancel any orphaned SL/TP orders for this symbol
                try:
                    await adapter._bitget_request(
                        "POST", "/api/v2/mix/order/cancel-all-orders",
                        mode="crossed",
                        body={
                            "symbol": sym.upper(),
                            "productType": "USDT-FUTURES",
                            "marginCoin": "USDT",
                        },
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"[trendline_orders] cancel {sym} err: {e}", flush=True)

        # Place new plan order
        try:
            # Set leverage
            try:
                await adapter._bitget_request(
                    "POST", "/api/v2/mix/account/set-leverage",
                    mode="crossed",
                    body={
                        "symbol": sym.upper(),
                        "productType": "USDT-FUTURES",
                        "marginCoin": "USDT",
                        "leverage": str(leverage),
                    },
                )
            except Exception:
                pass

            # Calculate quantity based on risk
            equity = cfg.get("equity", 50.0)
            risk_pct = cfg.get("risk_pct", 0.01)
            risk_usd = equity * risk_pct
            stop_distance = abs(limit_px - stop_px)
            if stop_distance <= 0:
                continue
            qty = risk_usd / stop_distance

            print(f"[trendline_orders] {sym} {direction}: qty={qty:.4f} trigger={limit_px:.6f} SL={stop_px:.6f} TP={tp_px:.6f} buf={buffer_pct*100:.3f}%", flush=True)

            now_ts = int(time.time())
            intent = OrderIntent(
                order_intent_id=f"tl_{sym}_{now_ts}",
                signal_id=f"tl_sig_{now_ts}",
                line_id="",
                client_order_id=f"tl_{sym[:10]}_{now_ts}",
                symbol=sym.upper(),
                timeframe=tf,
                side=direction,
                order_type="market",          # executes as market WHEN triggered
                trigger_mode="plan",
                entry_price=limit_px,
                stop_price=stop_px,
                tp_price=tp_px,
                quantity=qty,
                status="approved",
                reason="trendline_plan_order",
                created_at_bar=current_bar_index,
                created_at_ts=now_ts,
            )

            # Use PLAN ORDER — doesn't occupy margin until triggered!
            resp = await adapter.submit_live_plan_entry(
                intent, mode="crossed", trigger_price=limit_px,
            )
            order_id = resp.get("exchange_order_id", "")

            if not resp.get("ok"):
                print(f"[trendline_orders] PLAN FAILED {sym}: {resp.get('reason', resp)}", flush=True)
            if resp.get("ok"):
                new_active.append(ActiveLineOrder(
                    symbol=sym, timeframe=tf, kind=kind,
                    slope=slope, intercept=intercept,
                    anchor1_bar=a1, anchor2_bar=a2,
                    current_projected_price=proj,
                    limit_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                    exchange_order_id=order_id,
                    created_ts=now, last_updated_ts=now,
                    status="placed",
                ))
                placed += 1
                print(f"[trendline_orders] LIMIT {direction} {sym} {tf} @ {limit_px:.6f} "
                      f"SL={stop_px:.6f} TP={tp_px:.6f}", flush=True)

                # Telegram notify
                try:
                    from server.core.events import bus, Event
                    bus.emit(Event("position.opened", {
                        "symbol": sym, "side": direction,
                        "size_usd": risk_usd,
                        "entry_price": limit_px,
                        "sl": stop_px, "tp": tp_px,
                        "strategy": "trendline_limit",
                        "timeframe": tf,
                    }))
                except Exception:
                    pass

                # Log trade
                try:
                    from server.strategy.trade_log import log_trade
                    log_trade(
                        symbol=sym, timeframe=tf, strategy="trendline_limit",
                        direction=direction, entry_price=limit_px,
                        stop_price=stop_px, tp_price=tp_px,
                        size_usd=risk_usd, leverage=leverage,
                        order_id=order_id, status="limit_placed",
                    )
                except Exception:
                    pass
            else:
                print(f"[trendline_orders] REJECTED {sym}: {resp.get('reason')}", flush=True)

        except Exception as e:
            print(f"[trendline_orders] place {sym} err: {e}", flush=True)

    # Cancel remaining old orders that are no longer in signal list
    for key, old in existing.items():
        if old.status == "placed" and old.exchange_order_id:
            try:
                await adapter._bitget_request(
                    "POST", "/api/v2/mix/order/cancel-plan-order",
                    mode="crossed",
                    body={
                        "symbol": old.symbol.upper(),
                        "productType": "USDT-FUTURES",
                        "orderId": old.exchange_order_id,
                    },
                )
                # Clean orphaned SL/TP
                await adapter._bitget_request(
                    "POST", "/api/v2/mix/order/cancel-all-orders",
                    mode="crossed",
                    body={
                        "symbol": old.symbol.upper(),
                        "productType": "USDT-FUTURES",
                        "marginCoin": "USDT",
                    },
                )
                cancelled += 1
            except Exception:
                pass

    _save_active(new_active)
    return {"placed": placed, "updated": updated, "cancelled": cancelled}
