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
