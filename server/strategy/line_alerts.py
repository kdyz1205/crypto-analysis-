"""Trendline price alerts — TradingView-style.

When price touches a drawn trendline, fire a Telegram notification.
Supports single-fire (one-time) and repeating alerts.

Storage: JSON file at data/line_alerts.json
Poll: called from mar_bb_runner scan loop every 60s
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import urllib.request
import urllib.parse


ALERTS_FILE = Path("data/line_alerts.json")
TOUCH_THRESHOLD = 0.003  # 0.3% — price within this % of line = "touched"


@dataclass
class LineAlert:
    alert_id: str
    symbol: str
    timeframe: str
    slope: float
    intercept: float
    kind: str                          # 'support' | 'resistance'
    mode: Literal["single", "repeat"]  # single = fire once then deactivate
    active: bool = True
    last_fired_ts: float = 0.0
    cooldown_s: float = 300.0          # 5 min cooldown for repeat alerts
    label: str = ""                    # optional user note


def _load_alerts() -> list[LineAlert]:
    if not ALERTS_FILE.exists():
        return []
    with open(ALERTS_FILE, "r") as f:
        data = json.load(f)
    return [LineAlert(**d) for d in data]


def _save_alerts(alerts: list[LineAlert]):
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump([asdict(a) for a in alerts], f, indent=2)


def add_alert(
    symbol: str, timeframe: str,
    slope: float, intercept: float, kind: str,
    mode: str = "single", label: str = "",
) -> LineAlert:
    alerts = _load_alerts()
    alert_id = f"alert_{symbol}_{timeframe}_{int(time.time())}"
    a = LineAlert(
        alert_id=alert_id, symbol=symbol, timeframe=timeframe,
        slope=slope, intercept=intercept, kind=kind,
        mode=mode, label=label,
    )
    alerts.append(a)
    _save_alerts(alerts)
    return a


def remove_alert(alert_id: str) -> bool:
    alerts = _load_alerts()
    before = len(alerts)
    alerts = [a for a in alerts if a.alert_id != alert_id]
    _save_alerts(alerts)
    return len(alerts) < before


def list_alerts() -> list[dict]:
    return [asdict(a) for a in _load_alerts()]


def _send_telegram(text: str):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        req = urllib.request.Request(
            url, data=urllib.parse.urlencode(data).encode("utf-8"), method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def check_alerts(current_prices: dict[str, float], bar_index: int = 0):
    """Called every scan cycle. current_prices = {symbol: last_close}.

    For each active alert, compute line_price at current bar, compare to
    current price. If touched → fire Telegram → handle single/repeat.
    """
    alerts = _load_alerts()
    changed = False
    now = time.time()

    for a in alerts:
        if not a.active:
            continue
        price = current_prices.get(a.symbol)
        if price is None or price <= 0:
            continue

        line_price = a.slope * bar_index + a.intercept
        if line_price <= 0:
            line_price = a.intercept
        if line_price <= 0:
            continue

        distance_pct = abs(price - line_price) / line_price

        if distance_pct <= TOUCH_THRESHOLD:
            # Cooldown check for repeat alerts
            if a.mode == "repeat" and (now - a.last_fired_ts) < a.cooldown_s:
                continue

            # FIRE
            direction = "above" if price > line_price else "below"
            emoji = "🔔" if a.mode == "single" else "🔁"
            _send_telegram(
                f"{emoji} *ALERT: {a.symbol}*\n"
                f"Price `{price:.4f}` touched {a.kind} line\n"
                f"Line at `{line_price:.4f}` ({direction})\n"
                f"Distance: `{distance_pct*100:.3f}%`\n"
                f"Mode: `{a.mode}`\n"
                f"_{a.label}_" if a.label else ""
            )

            a.last_fired_ts = now
            if a.mode == "single":
                a.active = False
            changed = True

    if changed:
        _save_alerts(alerts)
