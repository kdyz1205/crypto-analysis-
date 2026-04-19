"""Trade log — records every order for daily review + ML training.

Appends one JSON line per trade to data/trade_log.jsonl.
Also provides daily summary + CSV export.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path("data/trade_log.jsonl")
QUARANTINE_FILE = Path("data/trade_log_quarantine.jsonl")


def _quarantine(record: dict, reason: str) -> None:
    """Send a malformed close/fill record to a side file so the audit trail
    stays intact AND a loud warning shows in logs. Prevents the 2026-04-18
    pattern where entry/close prices were silently 0.0 for every row.
    """
    QUARANTINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(record)
    entry["_quarantine_reason"] = reason
    entry["_quarantine_ts"] = time.time()
    try:
        with open(QUARANTINE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    print(
        f"[trade_log] QUARANTINE {reason}: {record.get('symbol','?')} "
        f"{record.get('direction','?')} close={record.get('close_price')} "
        f"entry={record.get('entry_price')} pnl={record.get('pnl')}",
        flush=True,
    )


def log_trade(
    symbol: str, timeframe: str, strategy: str, direction: str,
    entry_price: float, stop_price: float, tp_price: float,
    size_usd: float, leverage: float,
    order_id: str = "", status: str = "filled",
    **extra,
):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "dt": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "tp_price": tp_price,
        "size_usd": size_usd,
        "leverage": leverage,
        "order_id": order_id,
        "status": status,
        **extra,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def log_fill(order_id: str, symbol: str, direction: str, fill_price: float,
             quantity: float, **extra):
    """Record when a plan order triggers and position opens."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
        "event": "fill", "order_id": order_id, "symbol": symbol,
        "direction": direction, "fill_price": fill_price, "quantity": quantity,
        **extra,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def log_close(order_id: str, symbol: str, direction: str, close_price: float,
              pnl: float, pnl_pct: float, reason: str = "", **extra):
    """Record when a position is closed (SL/TP/manual/line-broken).

    Integrity guard: close_price and (when supplied) entry_price must be > 0.
    Rows that fail this check are routed to the quarantine file with a reason,
    NOT silently written into the main log. This fixes the 2026-04-18 pattern
    where every trade had `entry_price=0.0, close_price=0.0, pnl_pct=+0.00%`
    because a Bitget field-name bug returned nulls.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
        "event": "close", "order_id": order_id, "symbol": symbol,
        "direction": direction, "close_price": close_price,
        "pnl": pnl, "pnl_pct": pnl_pct, "reason": reason,
        **extra,
    }
    # Integrity check: reject or quarantine obviously-broken price fields.
    # We allow pnl=0 (could legitimately break even) and pnl_pct=0 (computed
    # from zero prices, so it's a symptom, not the cause). close_price=0 is
    # never legitimate for a real close.
    entry_px = record.get("entry_price")
    bad = []
    try:
        if float(close_price or 0) <= 0:
            bad.append("close_price<=0")
    except (TypeError, ValueError):
        bad.append("close_price_nonnumeric")
    if entry_px is not None:
        try:
            if float(entry_px or 0) <= 0:
                bad.append("entry_price<=0")
        except (TypeError, ValueError):
            bad.append("entry_price_nonnumeric")
    if bad:
        _quarantine(record, reason=",".join(bad))
        return record
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def log_sl_move(symbol: str, direction: str, old_sl: float, new_sl: float,
                tf: str = "", bars: int = 0, **extra):
    """Record SL movement at bar boundary."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
        "event": "sl_move", "symbol": symbol, "direction": direction,
        "old_sl": old_sl, "new_sl": new_sl, "tf": tf, "bars": bars,
        **extra,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def get_trades(last_n: int = 100) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text().strip().split("\n")
    trades = [json.loads(l) for l in lines if l.strip()]
    return trades[-last_n:]


def get_daily_summary(date_str: str | None = None) -> dict:
    """Summary for a given date (default today UTC)."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    trades = [t for t in get_trades(10000) if t.get("dt", "").startswith(date_str)]
    return {
        "date": date_str,
        "total_trades": len(trades),
        "strategies": list(set(t.get("strategy", "") for t in trades)),
        "symbols": list(set(t.get("symbol", "") for t in trades)),
        "trades": trades,
    }


def export_csv(output_path: str = "data/trade_log.csv"):
    """Export all trades to CSV for analysis."""
    trades = get_trades(100000)
    if not trades:
        return 0
    import csv
    keys = list(trades[0].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(trades)
    return len(trades)
