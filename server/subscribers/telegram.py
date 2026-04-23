"""
Telegram subscriber — forwards events to configured chat.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env at registration time.
"""

import os

import httpx

from ..core.events import bus, Event

_cfg: dict | None = None


def _load_config():
    global _cfg
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        _cfg = {"token": token, "chat_id": chat_id}


async def _send(text: str):
    if not _cfg:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{_cfg['token']}/sendMessage",
                json={"chat_id": _cfg["chat_id"], "text": text, "parse_mode": "HTML"},
            )
    except Exception as e:
        print(f"[TelegramSubscriber] Send failed: {e}")


async def on_signal_blocked(event: Event):
    p = event.payload
    reasons = p.get("block_reasons", [])
    await _send(
        f"🚫 <b>Signal blocked</b> {p['symbol']} {p['side'].upper()}\n"
        f"Reasons: {', '.join(reasons[:3])}"
    )


async def on_position_opened(event: Event):
    p = event.payload
    arrow = "📈" if p["side"] == "long" else "📉"
    await _send(
        f"{arrow} <b>Opened {p['side'].upper()} {p['symbol']}</b>\n"
        f"Size: ${p['size_usd']:.0f}\n"
        f"Entry: ${p['entry_price']}\n"
        f"SL: ${p['sl']} | TP: ${p['tp']}"
    )


async def on_position_closed(event: Event):
    p = event.payload
    emoji = "🟢" if p["pnl_usd"] >= 0 else "🔴"
    await _send(
        f"{emoji} <b>Closed {p['symbol']}</b>\n"
        f"P&L: {p['pnl_pct']:+.2f}% (${p['pnl_usd']:+.2f})\n"
        f"Reason: {p['reason']}"
    )


async def on_risk_limit_hit(event: Event):
    p = event.payload
    await _send(
        f"⚠️ <b>Risk limit hit</b>\n"
        f"Limit: {p['limit']}\n"
        f"Current: {p['current']} / Max: {p['max']}"
    )


async def on_agent_started(event: Event):
    p = event.payload
    await _send(
        f"🚀 <b>Agent started</b>\n"
        f"Mode: {p['mode'].upper()}\n"
        f"Equity: ${p['equity']:.2f}\n"
        f"Generation: {p['generation']}"
    )


async def on_agent_stopped(event: Event):
    p = event.payload
    await _send(f"🛑 <b>Agent stopped</b>\nReason: {p.get('reason', 'manual')}")


async def on_agent_regime_changed(event: Event):
    p = event.payload
    await _send(
        f"🔄 <b>Market regime changed</b>\n"
        f"{p['from']} → {p['to']} (confidence: {p['confidence']:.2f})"
    )


async def on_summary_daily(event: Event):
    p = event.payload
    await _send(
        f"📊 <b>Daily summary</b>\n"
        f"Equity: ${p['equity']:.2f}\n"
        f"Daily PnL: ${p['daily_pnl']:+.2f}\n"
        f"Trades: {p['total_trades']} | Win rate: {p['win_rate']:.0f}%"
    )


# ── Manual-line conditional order events (user 2026-04-22) ────────────
# Bridged from watcher._append_event. User: "Bitget doesn't notify me
# when my plan orders fire; I need Telegram for every order event."

def _esc_html(s) -> str:
    """Escape HTML special chars so Telegram's parse_mode=HTML never
    rejects the message (user saw broken formatting when a cancel
    reason contained a bare '<' char)."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_COND_EMOJI = {
    "exchange_submitted": "🔵",
    "exchange_acked": "✅",
    "triggered": "⚡",
    "breakout": "⚡",
    "cancelled": "⚪",
    "line_broken": "❌",
    "exchange_error": "⚠️",
}

_COND_TITLE_ZH = {
    "exchange_submitted": "Bitget 下单成功",
    "exchange_acked": "订单触发 / 成交",
    "triggered": "条件达成",
    "breakout": "突破触发",
    "cancelled": "撤单",
    "line_broken": "线破 cancel",
    "exchange_error": "Bitget 错误",
}


async def on_conditional_event(event: Event):
    """Forward a manual-line conditional's lifecycle event to Telegram.
    Filters out noisy kinds upstream in _append_event (only user-
    relevant kinds reach this handler)."""
    p = event.payload or {}
    kind = p.get("kind") or "?"
    symbol = p.get("symbol") or "?"
    tf = p.get("timeframe") or ""
    direction = (p.get("direction") or "?").upper()
    emoji = _COND_EMOJI.get(kind, "*")
    title = _COND_TITLE_ZH.get(kind, kind)
    price = p.get("price")
    line_price = p.get("line_price")
    oid = p.get("exchange_order_id")
    message = p.get("message") or ""
    mode = (p.get("exchange_mode") or "").lower()
    mode_tag = "" if mode in ("", "live") else f" [{mode.upper()}]"

    lines = [
        f"{emoji} <b>{title}</b> {_esc_html(direction)} {_esc_html(symbol)} "
        f"{_esc_html(tf)}{mode_tag}"
    ]
    if price is not None:
        try:
            lines.append(f"price: <code>{float(price):.6f}</code>")
        except Exception:
            lines.append(f"price: <code>{_esc_html(price)}</code>")
    if line_price is not None:
        try:
            lines.append(f"line:  <code>{float(line_price):.6f}</code>")
        except Exception:
            lines.append(f"line:  <code>{_esc_html(line_price)}</code>")
    if oid:
        lines.append(f"oid:   <code>{_esc_html(oid)}</code>")
    if message:
        msg_short = message if len(message) < 400 else (message[:380] + "...")
        lines.append(f"<i>{_esc_html(msg_short)}</i>")
    await _send("\n".join(lines))


def register():
    _load_config()
    if not _cfg:
        print("[TelegramSubscriber] No config (TELEGRAM_BOT_TOKEN/CHAT_ID missing) -- skipping")
        return
    bus.subscribe("signal.blocked", on_signal_blocked)
    bus.subscribe("position.opened", on_position_opened)
    bus.subscribe("position.closed", on_position_closed)
    bus.subscribe("risk.limit.hit", on_risk_limit_hit)
    bus.subscribe("agent.started", on_agent_started)
    bus.subscribe("agent.stopped", on_agent_stopped)
    bus.subscribe("agent.regime.changed", on_agent_regime_changed)
    bus.subscribe("summary.daily", on_summary_daily)
    # Wildcard for all manual-line conditional events (user 2026-04-22)
    bus.subscribe("conditional.*", on_conditional_event)
    print(f"[TelegramSubscriber] Registered for chat_id={_cfg['chat_id']} (conditional.*)")
