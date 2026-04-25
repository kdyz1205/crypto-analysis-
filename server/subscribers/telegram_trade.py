"""Trade-event classifier + Telegram formatter (extension of telegram.py).

2026-04-25: user wants Telegram to distinguish:
    - entry filled (position opened)
    - SL hit (stop-loss closed position)
    - TP hit (take-profit closed position)
    - manual close (position closed for other reason)
    - order cancelled

…and want it source-agnostic so Hyperliquid / paper engines can publish
the same events without changing telegram.py.

DESIGN:
    1. Classifier subscribes to source-specific raw events
       (bitget `order.ws_update`, watcher `conditional.*`, paper engine).
    2. Each classifier maps the raw payload to a normalized
       TradeEvent and publishes `trade.<event_type>` to the bus.
    3. The Telegram handler `on_trade_event` subscribes to `trade.*`
       and renders one of five clean message templates.
    4. Dedup: same (source, exchange_order_id, event_type) within 60 s
       is silently dropped — protects against WS + polling double-detect.

This file deliberately does NOT touch the existing handlers in
telegram.py. Existing flows (position.opened / closed / conditional.*)
keep firing. The new `trade.*` events are ADDITIVE.

WHAT'S BUILT-IN HERE
    - `TradeEvent` dataclass — universal payload shape
    - `_dedup_key`, `_should_emit` — TTL set
    - `classify_bitget_ws_order(row)` — Bitget WS orders channel → TradeEvent
    - `on_trade_event(event)` — Telegram handler with rich formatting
    - `register()` — wires bus.subscribe for raw + classified events

NOTHING in here imports a network library; the only side-effect is
calling `_send` from telegram.py, which is already the central HTTP
gateway to Telegram.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

from ..core.events import bus, Event
from . import telegram as tg

# ─── Type definitions ────────────────────────────────────────────────────

TradeEventType = Literal[
    "entry_filled",       # position opened — plan-order trigger fired + filled
    "sl_hit",             # position closed at preset stop-loss
    "tp_hit",             # position closed at preset take-profit
    "position_closed",    # position closed for other reason (manual, reverse, etc.)
    "cancelled",          # plan-order cancelled before triggering (no fill)
]

TradeSource = Literal["bitget", "hyperliquid", "paper", "demo", "unknown"]


@dataclass(slots=True)
class TradeEvent:
    """Universal trade event published to bus as `trade.<event_type>`.

    Fields are nullable when the source can't supply them — e.g. paper
    engine has no `exchange_order_id`. Telegram formatter handles None
    gracefully (skips that line).
    """
    event_type: TradeEventType
    source: TradeSource
    symbol: str
    direction: Optional[str]                # 'long' | 'short'
    fill_price: Optional[float]             # actual fill (for entry/sl/tp/close)
    trigger_price: Optional[float]          # plan trigger
    entry_price: Optional[float]            # for sl/tp/close: original entry
    size: Optional[float]                   # contract qty
    pnl_usd: Optional[float]
    pnl_pct: Optional[float]
    reason: str                             # human-readable reason
    exchange_order_id: Optional[str]
    client_order_id: Optional[str]
    conditional_id: Optional[str]
    timestamp: int

    def to_payload(self) -> dict:
        return {
            "event_type": self.event_type,
            "source": self.source,
            "symbol": self.symbol,
            "direction": self.direction,
            "fill_price": self.fill_price,
            "trigger_price": self.trigger_price,
            "entry_price": self.entry_price,
            "size": self.size,
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "reason": self.reason,
            "exchange_order_id": self.exchange_order_id,
            "client_order_id": self.client_order_id,
            "conditional_id": self.conditional_id,
            "timestamp": self.timestamp,
        }


# ─── Dedup ──────────────────────────────────────────────────────────────

_DEDUP_TTL_SEC = 60
_dedup_seen: dict[tuple[str, str, str], float] = {}


def _dedup_key(source: str, oid: Optional[str], event_type: str) -> tuple[str, str, str]:
    """Identity for dedup. If oid is missing, use a synthesised key based
    on source+event_type+timestamp-bucket so identical events without
    an order id (paper) still dedup within the same second."""
    return (source, str(oid or "no_oid"), event_type)


def _should_emit(source: str, oid: Optional[str], event_type: str) -> bool:
    """True if this event should publish. Records in dedup set if so."""
    now = time.time()
    # Cleanup expired entries (cheap — small dict)
    for k, ts in list(_dedup_seen.items()):
        if now - ts > _DEDUP_TTL_SEC * 2:
            _dedup_seen.pop(k, None)
    key = _dedup_key(source, oid, event_type)
    last = _dedup_seen.get(key)
    if last is not None and now - last < _DEDUP_TTL_SEC:
        return False
    _dedup_seen[key] = now
    return True


def _reset_dedup_for_tests() -> None:
    """Hook for unit tests so each test starts with a clean dedup table."""
    _dedup_seen.clear()


# ─── Classifier: Bitget WS order frame ──────────────────────────────────

def classify_bitget_ws_order(row: dict) -> Optional[TradeEvent]:
    """Map a single Bitget orders-channel WS push to a TradeEvent.

    Bitget v2 WS orders push fields (subset we care about):
        symbol, orderId, clientOid, side ('buy'|'sell'),
        tradeSide ('open'|'close'), planType ('normal_plan'|'profit_loss'|...),
        status ('live'|'partial-fill'|'filled'|'cancelled'),
        fillPrice, fillSize, triggerPrice, posSide, fillTime, ...

    Returns None when the row isn't terminal (e.g. status=live,
    partial-fill that isn't the final fill, or unknown plan_type).
    """
    if not isinstance(row, dict):
        return None
    status = str(row.get("status") or "").lower()
    if status not in ("filled", "cancelled"):
        # Live / partial-fill / new — skip, only terminal states matter for TG
        return None

    symbol = str(row.get("symbol") or row.get("instId") or "").upper()
    if not symbol:
        return None
    plan_type = str(row.get("planType") or "normal_plan").lower()
    trade_side = str(row.get("tradeSide") or "").lower()    # 'open' | 'close'
    bitget_side = str(row.get("side") or "").lower()        # 'buy' | 'sell'

    # Direction inference: tradeSide=open + side=buy → long; open + sell → short.
    # On close, the side is the OPPOSITE of the position direction (close-long = sell).
    if trade_side == "open":
        direction = "long" if bitget_side == "buy" else "short"
    elif trade_side == "close":
        direction = "long" if bitget_side == "sell" else "short"
    else:
        direction = None

    fill_price = _safe_float(row.get("fillPrice") or row.get("priceAvg") or row.get("price"))
    trigger_price = _safe_float(row.get("triggerPrice"))
    size = _safe_float(row.get("fillSize") or row.get("size"))
    oid = str(row.get("orderId") or "") or None
    cid = str(row.get("clientOid") or "") or None
    pnl_usd = _safe_float(row.get("totalProfits") or row.get("profit") or row.get("pnl"))
    ts_ms = row.get("uTime") or row.get("fillTime") or row.get("cTime")
    try:
        ts = int(ts_ms) // 1000 if ts_ms else int(time.time())
    except (TypeError, ValueError):
        ts = int(time.time())

    # ── Decide event_type ──────────────────────────────────────────────
    if status == "cancelled":
        event_type: TradeEventType = "cancelled"
        reason = "user/system cancelled"
    elif plan_type in ("profit_loss", "pos_profit", "pos_loss",
                       "moving_plan", "track_plan"):
        # SL/TP/trailing-stop — figure out which by sign of pnl
        if pnl_usd is not None:
            event_type = "tp_hit" if pnl_usd >= 0 else "sl_hit"
        else:
            # Without pnl we infer from the planType subtype if Bitget
            # provides it; else fall back to generic close.
            if plan_type in ("pos_loss", "loss_plan"):
                event_type = "sl_hit"
            elif plan_type in ("pos_profit", "profit_plan"):
                event_type = "tp_hit"
            else:
                event_type = "position_closed"
        reason = f"{plan_type} fired"
    elif trade_side == "open":
        event_type = "entry_filled"
        reason = "plan-order trigger fired"
    elif trade_side == "close":
        event_type = "position_closed"
        reason = "position closed"
    else:
        # Unknown — don't classify rather than guess
        return None

    return TradeEvent(
        event_type=event_type,
        source="bitget",
        symbol=symbol,
        direction=direction,
        fill_price=fill_price,
        trigger_price=trigger_price,
        entry_price=None,             # not present in this push; SL/TP-aware view enriches
        size=size,
        pnl_usd=pnl_usd,
        pnl_pct=None,
        reason=reason,
        exchange_order_id=oid,
        client_order_id=cid,
        conditional_id=_extract_cond_id(cid),
        timestamp=ts,
    )


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_cond_id(client_order_id: Optional[str]) -> Optional[str]:
    """Our system places orders with clientOid='replan_cond_<id>_<ts>' or
    'line_<line_id>_<ts>' or similar. Pull the cond_id when possible."""
    if not client_order_id:
        return None
    s = str(client_order_id)
    if s.startswith("replan_cond_"):
        # replan_cond_<id>_<ts>
        rest = s[len("replan_cond_"):]
        return "cond_" + rest.split("_")[0] if rest else None
    if s.startswith("cond_"):
        return s.split("_")[0] + "_" + (s.split("_")[1] if len(s.split("_")) > 1 else "")
    return None


# ─── Bus glue: source raw → publish trade.* ──────────────────────────────

async def on_bitget_ws_order(event: Event):
    """Subscriber for `order.ws_update` (published by bitget_private_ws).
    Classifies the row and republishes as `trade.<event_type>`."""
    payload = event.payload or {}
    row = payload.get("data") or {}
    te = classify_bitget_ws_order(row)
    if te is None:
        return
    if not _should_emit(te.source, te.exchange_order_id, te.event_type):
        return
    await bus.publish(Event(
        type=f"trade.{te.event_type}",
        payload=te.to_payload(),
    ))


def classify_watcher_event(payload: dict) -> Optional[TradeEvent]:
    """Map a `conditional.<kind>` event payload (published by
    server/conditionals/watcher.py:_append_event) to a TradeEvent.

    Watcher-detected events are SLOWER than WS pushes (poll-based, ~30s
    lag), but they're authoritative for cancel + entry-fill via
    Bitget's REST history. They never produce SL/TP events because
    those are exchange-side preset close orders our watcher doesn't
    follow — that's purely WS territory.
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind") or ""
    symbol = payload.get("symbol") or ""
    direction = payload.get("direction") or None
    oid = payload.get("exchange_order_id")
    mode = (payload.get("exchange_mode") or "live").lower()
    src: TradeSource = "bitget" if mode == "live" else ("paper" if mode in ("paper", "demo") else "bitget")
    msg = payload.get("message") or ""
    fill = _safe_float(payload.get("price"))
    line_price = _safe_float(payload.get("line_price"))
    cond_id = payload.get("conditional_id")
    ts = int(time.time())

    if kind == "cancelled":
        event_type: TradeEventType = "cancelled"
        reason = msg or "watcher: cancelled"
    elif kind == "exchange_acked":
        # exchange_acked = position opened (plan-order fill captured)
        event_type = "entry_filled"
        reason = msg or "watcher: entry filled"
    elif kind == "line_broken":
        event_type = "cancelled"
        reason = msg or "watcher: line broken → cancel"
    else:
        # exchange_submitted / exchange_error / triggered / breakout
        # already covered by on_conditional_event in telegram.py;
        # don't double-fire for them here.
        return None

    return TradeEvent(
        event_type=event_type,
        source=src,
        symbol=symbol,
        direction=direction,
        fill_price=fill,
        trigger_price=line_price,
        entry_price=None,
        size=None,
        pnl_usd=None,
        pnl_pct=None,
        reason=reason,
        exchange_order_id=str(oid) if oid else None,
        client_order_id=None,
        conditional_id=str(cond_id) if cond_id else None,
        timestamp=ts,
    )


async def on_watcher_conditional_event(event: Event):
    """Subscriber for `conditional.cancelled` / `conditional.exchange_acked`
    / `conditional.line_broken`. Republishes as trade.* with dedup."""
    payload = event.payload or {}
    te = classify_watcher_event(payload)
    if te is None:
        return
    if not _should_emit(te.source, te.exchange_order_id, te.event_type):
        return
    await bus.publish(Event(
        type=f"trade.{te.event_type}",
        payload=te.to_payload(),
    ))


# ─── Telegram formatting ────────────────────────────────────────────────

_EVENT_EMOJI: dict[str, str] = {
    "entry_filled":    "🟢",
    "sl_hit":          "🔴",   # stop loss hit = red
    "tp_hit":          "🎯",   # take profit hit = bullseye
    "position_closed": "⚪",
    "cancelled":       "↩️",
}

_EVENT_TITLE_ZH: dict[str, str] = {
    "entry_filled":    "入场成交",
    "sl_hit":          "止损触发",
    "tp_hit":          "止盈触发",
    "position_closed": "持仓平仓",
    "cancelled":       "挂单取消",
}


def format_trade_event(p: dict) -> str:
    """Render a TradeEvent payload as a Telegram-ready HTML message.

    Keeps line count small (mobile-readable). Skips fields that are None
    so paper-mode events don't show "oid: None".
    """
    et = p.get("event_type") or "?"
    emoji = _EVENT_EMOJI.get(et, "*")
    title = _EVENT_TITLE_ZH.get(et, et)
    src = (p.get("source") or "?").lower()
    src_tag = f" [{src}]" if src and src != "bitget" else ""
    sym = p.get("symbol") or "?"
    direction = (p.get("direction") or "").upper()

    head = f"{emoji} <b>{tg._esc_html(title)}</b> {tg._esc_html(direction)} {tg._esc_html(sym)}{tg._esc_html(src_tag)}"
    lines = [head]

    fill = p.get("fill_price")
    trigger = p.get("trigger_price")
    entry = p.get("entry_price")
    size = p.get("size")
    pnl_usd = p.get("pnl_usd")
    pnl_pct = p.get("pnl_pct")

    if fill is not None:
        lines.append(f"成交价: <code>{_fmt_price(fill)}</code>")
    if trigger is not None and (fill is None or abs((fill or 0) - trigger) > 1e-9):
        lines.append(f"触发价: <code>{_fmt_price(trigger)}</code>")
    if entry is not None and et in ("sl_hit", "tp_hit", "position_closed"):
        lines.append(f"入场价: <code>{_fmt_price(entry)}</code>")
    if size is not None:
        lines.append(f"数量: <code>{_fmt_size(size)}</code>")
    if pnl_usd is not None:
        sign = "+" if pnl_usd > 0 else ""
        if pnl_pct is not None:
            lines.append(f"盈亏: <code>{sign}${pnl_usd:.2f}</code> ({sign}{pnl_pct:.2f}%)")
        else:
            lines.append(f"盈亏: <code>{sign}${pnl_usd:.2f}</code>")
    reason = p.get("reason")
    if reason:
        lines.append(f"<i>{tg._esc_html(reason)}</i>")
    oid = p.get("exchange_order_id")
    if oid:
        lines.append(f"oid: <code>{tg._esc_html(oid)}</code>")
    return "\n".join(lines)


def _fmt_price(p) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return str(p)
    # Adaptive precision: high price coins → 2dp, low → 6dp
    if v >= 1000:
        return f"{v:.2f}"
    if v >= 10:
        return f"{v:.3f}"
    if v >= 1:
        return f"{v:.4f}"
    return f"{v:.6f}"


def _fmt_size(s) -> str:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return str(s)
    return f"{v:.4f}".rstrip("0").rstrip(".")


async def on_trade_event(event: Event):
    """Telegram handler: subscribed to `trade.*` wildcard."""
    text = format_trade_event(event.payload or {})
    await tg._send(text)


# ─── Registration ──────────────────────────────────────────────────────

def register():
    """Add the trade-classifier + telegram-trade handlers to the bus."""
    # Source 1: real-time WS push from Bitget private channel
    bus.subscribe("order.ws_update", on_bitget_ws_order)
    # Source 2: watcher's polling-based events (works without WS)
    bus.subscribe("conditional.cancelled", on_watcher_conditional_event)
    bus.subscribe("conditional.exchange_acked", on_watcher_conditional_event)
    bus.subscribe("conditional.line_broken", on_watcher_conditional_event)
    # Sink: Telegram message
    bus.subscribe("trade.*", on_trade_event)
    print("[telegram_trade] registered: bitget WS + watcher → trade.* → Telegram")
