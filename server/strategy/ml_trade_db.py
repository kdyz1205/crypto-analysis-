"""ML Trade Database — saves every trade lifecycle with full features for PyTorch.

Each trade is a JSON object with:
- Entry context: line features, market state at entry
- Execution: fill price, slippage, timing
- Exit: reason, price, PnL, hold duration
- SL movements: full history of SL changes

Stored in data/ml_trades.jsonl, one JSON per line per event.
Queryable by symbol, TF, date range for training.
"""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path

ML_TRADES_FILE = Path("data/ml_trades.jsonl")


def _write(record: dict):
    ML_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ML_TRADES_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_plan_placed(
    symbol: str, tf: str, direction: str, kind: str,
    slope: float, intercept: float, bar_count: int,
    entry_price: float, stop_price: float, tp_price: float,
    buffer_pct: float, rr: float, leverage: int,
    quantity: float, order_id: str,
    pivot1_bar: int = 0, pivot2_bar: int = 0,
    atr_pct: float = 0, bb_width: float = 0, ribbon_score: float = 0,
    **extra,
):
    _write({
        "event": "plan_placed",
        "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "tf": tf, "direction": direction, "kind": kind,
        "slope": slope, "intercept": intercept, "bar_count": bar_count,
        "entry_price": entry_price, "stop_price": stop_price, "tp_price": tp_price,
        "buffer_pct": buffer_pct, "rr": rr, "leverage": leverage,
        "quantity": quantity, "order_id": order_id,
        "pivot1_bar": pivot1_bar, "pivot2_bar": pivot2_bar,
        "atr_pct": atr_pct, "bb_width": bb_width, "ribbon_score": ribbon_score,
        **extra,
    })


def log_plan_triggered(
    order_id: str, symbol: str, direction: str,
    fill_price: float, trigger_price: float,
    slippage_pct: float = 0,
    **extra,
):
    _write({
        "event": "plan_triggered",
        "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id, "symbol": symbol, "direction": direction,
        "fill_price": fill_price, "trigger_price": trigger_price,
        "slippage_pct": slippage_pct,
        **extra,
    })


def log_sl_moved(
    symbol: str, direction: str, tf: str,
    old_sl: float, new_sl: float, projected_line: float,
    bars_since_entry: int, entry_price: float,
    **extra,
):
    _write({
        "event": "sl_moved",
        "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "direction": direction, "tf": tf,
        "old_sl": old_sl, "new_sl": new_sl, "projected_line": projected_line,
        "bars_since_entry": bars_since_entry, "entry_price": entry_price,
        **extra,
    })


def log_position_closed(
    symbol: str, direction: str, tf: str,
    entry_price: float, close_price: float,
    pnl_usd: float, pnl_pct: float,
    reason: str, hold_bars: int = 0, hold_seconds: float = 0,
    max_favorable: float = 0, max_adverse: float = 0,
    **extra,
):
    _write({
        "event": "position_closed",
        "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "direction": direction, "tf": tf,
        "entry_price": entry_price, "close_price": close_price,
        "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
        "reason": reason,
        "hold_bars": hold_bars, "hold_seconds": hold_seconds,
        "max_favorable": max_favorable, "max_adverse": max_adverse,
        **extra,
    })


def log_line_broken(
    symbol: str, tf: str, kind: str,
    projected_price: float, close_price: float,
    had_position: bool, order_id: str = "",
    **extra,
):
    _write({
        "event": "line_broken",
        "ts": time.time(), "dt": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "tf": tf, "kind": kind,
        "projected_price": projected_price, "close_price": close_price,
        "had_position": had_position, "order_id": order_id,
        **extra,
    })


def get_trades(last_n: int = 1000) -> list[dict]:
    if not ML_TRADES_FILE.exists():
        return []
    lines = ML_TRADES_FILE.read_text().strip().split("\n")
    trades = [json.loads(l) for l in lines[-last_n:] if l.strip()]
    return trades


def get_closed_trades(last_n: int = 500) -> list[dict]:
    return [t for t in get_trades(last_n * 3) if t.get("event") == "position_closed"][-last_n:]
