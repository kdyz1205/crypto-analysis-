"""Daily Analysis + Auto-Review (Level 3).

Responsibilities:
  1. Load today's trades from data/trade_log.jsonl
  2. Match against Bitget closed positions to determine outcome (win/loss)
  3. Compute daily metrics: WR, PF, per-strategy / per-TF / per-coin
  4. Drift detection: rolling 50-trade WR vs backtest baseline (42%)
  5. Send summary to Telegram
  6. Persist report to data/daily_reports/YYYY-MM-DD.json

The main entry point ``generate_daily_report()`` is designed to be
called from the scan loop (via ``check_daily_report()`` which gates
on UTC midnight).
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from server.execution import _bitget_fields as bgf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Resolve project root robustly."""
    try:
        from server.core.config import PROJECT_ROOT
        return Path(PROJECT_ROOT)
    except Exception:
        return Path(__file__).resolve().parents[3]

def _reports_dir() -> Path:
    d = _project_root() / "data" / "daily_reports"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _trade_log_path() -> Path:
    return _project_root() / "data" / "trade_log.jsonl"


# ---------------------------------------------------------------------------
# Trade loader
# ---------------------------------------------------------------------------

def _load_all_trades() -> list[dict]:
    """Load every trade from the JSONL log."""
    p = _trade_log_path()
    if not p.exists():
        return []
    trades = []
    for line in p.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            trades.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return trades


def _trades_for_date(date_str: str) -> list[dict]:
    """Filter trades belonging to the UTC day ``date_str`` (YYYY-MM-DD).

    Uses numeric epoch range ``[t0, t1)`` on ``t["ts"]`` so trades logged in
    the 1-second window around UTC midnight are classified correctly. Falls
    back to ``dt.startswith(date_str)`` only when ``ts`` is missing.
    """
    try:
        y, m, d = (int(x) for x in date_str.split("-", 2))
        t0 = datetime(y, m, d, tzinfo=timezone.utc).timestamp()
        t1 = t0 + 86400
    except Exception:
        # Malformed date_str — fall back to the lossy string path
        return [t for t in _load_all_trades() if str(t.get("dt", "")).startswith(date_str)]

    out: list[dict] = []
    for t in _load_all_trades():
        ts = t.get("ts")
        if ts not in (None, ""):
            try:
                ts_f = float(ts)
            except (TypeError, ValueError):
                ts_f = None
            if ts_f is not None:
                if t0 <= ts_f < t1:
                    out.append(t)
                continue
        # No usable ts — fall back to the dt-string prefix
        if str(t.get("dt", "")).startswith(date_str):
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Outcome matching via Bitget position history
# ---------------------------------------------------------------------------

async def _fetch_bitget_outcomes(days: int = 2) -> list[dict]:
    """Fetch closed-position history from Bitget for outcome matching.

    Returns raw position dicts from Bitget.  Falls back to empty list
    if the adapter isn't available or API keys are missing.
    """
    try:
        from server.strategy.mar_bb_history import fetch_position_history
        return await fetch_position_history(days=days, mode="live")
    except Exception as e:
        print(f"[daily_report] Bitget outcome fetch err: {e}", flush=True)
        return []


def _match_outcomes(trades: list[dict], bitget_rows: list[dict]) -> list[dict]:
    """Enrich each trade dict with ``outcome`` and ``pnl`` from Bitget.

    Matching logic:
      1. By order_id (clientOid prefix match)
      2. Fallback: symbol + rough timestamp match (within 5 min)

    Returns the same list with added keys:
      ``outcome``: "win" | "loss" | "breakeven" | "open" | "unknown"
      ``pnl``: float (USDT)
      ``pnl_pct``: float (%)
    """
    # Index Bitget rows by clientOid and by symbol+time
    by_oid: dict[str, dict] = {}
    by_sym_ts: dict[str, list[dict]] = defaultdict(list)
    for row in bitget_rows:
        coid = (row.get("clientOid") or row.get("clientOId") or "").lower()
        if coid:
            by_oid[coid] = row
        sym = (row.get("symbol") or "").upper()
        ts_ms = int(row.get("uTime") or row.get("cTime") or 0)
        if sym and ts_ms:
            by_sym_ts[sym].append({**row, "_ts_s": ts_ms / 1000})

    for trade in trades:
        oid = str(trade.get("order_id", "")).lower()
        symbol = str(trade.get("symbol", "")).upper()
        trade_ts = float(trade.get("ts", 0))

        matched = None

        # 1. Direct order_id match (our clientOid contains the exchange order_id)
        if oid:
            # Try exact match first
            matched = by_oid.get(oid)
            # Try prefix scan (we use "marbb_{SYMBOL}_{TS}" as clientOid)
            if not matched:
                for key, row in by_oid.items():
                    if oid in key or key in oid:
                        matched = row
                        break

        # 2. Symbol + time fallback
        if not matched and symbol and trade_ts > 0:
            candidates = by_sym_ts.get(symbol, [])
            for cand in candidates:
                if abs(cand["_ts_s"] - trade_ts) < 300:  # within 5 min
                    matched = cand
                    break

        if matched:
            pnl = bgf.realized_pnl_usd(matched)
            entry = bgf.open_price(matched)
            size = float(matched.get("closeTotalPos") or 0)
            # Prefer margin_used for pnl% denominator when available (matches
            # exchange's own pnl% view); fall back to entry*size notional,
            # then to the plan's size_usd.
            margin = bgf.margin_used(matched)
            if margin > 0:
                notional = margin
            elif entry > 0 and size > 0:
                notional = entry * size
            else:
                notional = float(trade.get("size_usd", 1))
            pnl_pct = (pnl / max(notional, 1.0)) * 100.0
            if pnl > 0:
                outcome = "win"
            elif pnl < 0:
                outcome = "loss"
            else:
                outcome = "breakeven"
            trade["outcome"] = outcome
            trade["pnl"] = round(pnl, 6)
            trade["pnl_pct"] = round(pnl_pct, 4)
        else:
            # Could still be open or just not in history yet
            trade["outcome"] = trade.get("outcome", "unknown")
            trade["pnl"] = trade.get("pnl", 0.0)
            trade["pnl_pct"] = trade.get("pnl_pct", 0.0)

    return trades


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(trades: list[dict]) -> dict[str, Any]:
    """Compute comprehensive daily metrics from enriched trades."""
    resolved = [t for t in trades if t.get("outcome") in ("win", "loss", "breakeven")]
    total = len(trades)
    wins = [t for t in resolved if t["outcome"] == "win"]
    losses = [t for t in resolved if t["outcome"] == "loss"]
    open_or_unknown = [t for t in trades if t.get("outcome") not in ("win", "loss", "breakeven")]

    win_pnls = [float(t.get("pnl", 0)) for t in wins]
    loss_pnls = [float(t.get("pnl", 0)) for t in losses]

    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    gross_profit = sum(win_pnls) if win_pnls else 0.0
    gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    net_pnl = gross_profit - gross_loss

    # Best / worst
    all_pnls = [(t.get("symbol", "?"), float(t.get("pnl", 0))) for t in resolved]
    best = max(all_pnls, key=lambda x: x[1]) if all_pnls else ("", 0.0)
    worst = min(all_pnls, key=lambda x: x[1]) if all_pnls else ("", 0.0)

    # Per-strategy breakdown
    by_strategy: dict[str, dict] = {}
    for t in resolved:
        strat = t.get("strategy", "unknown")
        if strat not in by_strategy:
            by_strategy[strat] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        by_strategy[strat]["trades"] += 1
        if t["outcome"] == "win":
            by_strategy[strat]["wins"] += 1
        elif t["outcome"] == "loss":
            by_strategy[strat]["losses"] += 1
        by_strategy[strat]["pnl"] += float(t.get("pnl", 0))
    for s in by_strategy.values():
        s["pnl"] = round(s["pnl"], 4)
        s["winrate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0.0

    # Per-TF breakdown
    by_tf: dict[str, dict] = {}
    for t in resolved:
        tf = t.get("timeframe", "?")
        if tf not in by_tf:
            by_tf[tf] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        by_tf[tf]["trades"] += 1
        if t["outcome"] == "win":
            by_tf[tf]["wins"] += 1
        elif t["outcome"] == "loss":
            by_tf[tf]["losses"] += 1
        by_tf[tf]["pnl"] += float(t.get("pnl", 0))
    for s in by_tf.values():
        s["pnl"] = round(s["pnl"], 4)
        s["winrate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0.0

    # Per-coin breakdown (top 5 winners, top 5 losers)
    by_coin: dict[str, float] = defaultdict(float)
    by_coin_count: dict[str, int] = defaultdict(int)
    for t in resolved:
        sym = t.get("symbol", "?")
        by_coin[sym] += float(t.get("pnl", 0))
        by_coin_count[sym] += 1
    sorted_coins = sorted(by_coin.items(), key=lambda x: x[1], reverse=True)
    top5_winners = sorted_coins[:5] if len(sorted_coins) >= 5 else sorted_coins
    top5_losers = sorted_coins[-5:] if len(sorted_coins) >= 5 else sorted_coins

    return {
        "total_trades": total,
        "resolved_trades": len(resolved),
        "open_or_unknown": len(open_or_unknown),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(resolved) * 100, 1) if resolved else 0.0,
        "avg_win_pct": round(sum(float(t.get("pnl_pct", 0)) for t in wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_pct": round(sum(float(t.get("pnl_pct", 0)) for t in losses) / len(losses), 2) if losses else 0.0,
        "avg_win_usd": round(avg_win, 4),
        "avg_loss_usd": round(avg_loss, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
        "net_pnl": round(net_pnl, 4),
        "best_trade": {"symbol": best[0], "pnl": round(best[1], 4)},
        "worst_trade": {"symbol": worst[0], "pnl": round(worst[1], 4)},
        "by_strategy": by_strategy,
        "by_timeframe": by_tf,
        "top5_winners": [{"symbol": s, "pnl": round(p, 4)} for s, p in top5_winners],
        "top5_losers": [{"symbol": s, "pnl": round(p, 4)} for s, p in top5_losers],
    }


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

BACKTEST_WR = 42.0   # baseline backtest win rate (%)
DRIFT_WINDOW = 50    # rolling trade count
DRIFT_WARN_THRESHOLD = 30.0  # warn if rolling WR drops below this


def _check_drift() -> dict[str, Any]:
    """Compare rolling 50-trade WR against backtest baseline.

    Returns a dict with drift info:
      ``rolling_wr``: float (%)
      ``baseline_wr``: float (%)
      ``warning``: bool
      ``message``: str
      ``sample_size``: int
    """
    all_trades = _load_all_trades()
    # Only look at trades that have an outcome
    resolved = [t for t in all_trades if t.get("outcome") in ("win", "loss")]

    if len(resolved) < DRIFT_WINDOW:
        return {
            "rolling_wr": 0.0,
            "baseline_wr": BACKTEST_WR,
            "warning": False,
            "message": f"Not enough resolved trades ({len(resolved)}/{DRIFT_WINDOW}) for drift check",
            "sample_size": len(resolved),
        }

    recent = resolved[-DRIFT_WINDOW:]
    wins = sum(1 for t in recent if t["outcome"] == "win")
    rolling_wr = wins / len(recent) * 100.0

    warning = rolling_wr < DRIFT_WARN_THRESHOLD
    if warning:
        msg = (f"DRIFT WARNING: rolling {DRIFT_WINDOW}-trade WR = {rolling_wr:.1f}% "
               f"(baseline {BACKTEST_WR:.0f}%, threshold {DRIFT_WARN_THRESHOLD:.0f}%). "
               f"Strategy may be degrading.")
    else:
        msg = f"Rolling {DRIFT_WINDOW}-trade WR = {rolling_wr:.1f}% (baseline {BACKTEST_WR:.0f}%)"

    return {
        "rolling_wr": round(rolling_wr, 1),
        "baseline_wr": BACKTEST_WR,
        "warning": warning,
        "message": msg,
        "sample_size": len(recent),
    }


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

async def _send_telegram(text: str) -> None:
    """Send a message via Telegram bot. Reads tokens from env."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("[daily_report] Telegram not configured (no BOT_TOKEN/CHAT_ID)", flush=True)
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            )
    except Exception as e:
        print(f"[daily_report] Telegram send err: {e}", flush=True)


def _format_report_telegram(date_str: str, metrics: dict, drift: dict) -> str:
    """Format the daily report as an HTML Telegram message."""
    lines = []
    lines.append(f"<b>Daily Report {date_str}</b>")
    lines.append("")

    if metrics["total_trades"] == 0:
        lines.append("Quiet day -- no trades fired.")
        return "\n".join(lines)

    wr = metrics["win_rate"]
    wr_emoji = "+" if wr >= 50 else "-" if wr < 35 else "~"
    pnl = metrics["net_pnl"]
    pnl_sign = "+" if pnl >= 0 else ""

    lines.append(f"Trades: {metrics['total_trades']} (resolved: {metrics['resolved_trades']})")
    lines.append(f"W/L: {metrics['wins']}/{metrics['losses']} ({wr_emoji}{wr:.1f}%)")
    lines.append(f"Net PnL: {pnl_sign}${pnl:.4f}")
    lines.append(f"PF: {metrics['profit_factor']:.2f}")
    lines.append(f"Avg win: ${metrics['avg_win_usd']:.4f} | Avg loss: ${metrics['avg_loss_usd']:.4f}")
    lines.append(f"Best: {metrics['best_trade']['symbol']} ${metrics['best_trade']['pnl']:.4f}")
    lines.append(f"Worst: {metrics['worst_trade']['symbol']} ${metrics['worst_trade']['pnl']:.4f}")

    # Strategy breakdown
    if metrics["by_strategy"]:
        lines.append("")
        lines.append("<b>By strategy:</b>")
        for strat, s in metrics["by_strategy"].items():
            lines.append(f"  {strat}: {s['wins']}/{s['trades']} ({s['winrate']:.0f}%) ${s['pnl']:.4f}")

    # TF breakdown
    if metrics["by_timeframe"]:
        lines.append("")
        lines.append("<b>By timeframe:</b>")
        for tf, s in sorted(metrics["by_timeframe"].items()):
            lines.append(f"  {tf}: {s['wins']}/{s['trades']} ({s['winrate']:.0f}%) ${s['pnl']:.4f}")

    # Drift
    lines.append("")
    if drift["warning"]:
        lines.append(f"<b>DRIFT WARNING</b>: {drift['message']}")
    else:
        lines.append(f"Drift: {drift['message']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

def _save_report(date_str: str, report: dict) -> Path:
    """Save the full report JSON to data/daily_reports/YYYY-MM-DD.json."""
    d = _reports_dir()
    path = d / f"{date_str}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def load_report(date_str: str) -> dict | None:
    """Load a previously saved daily report. Returns None if not found."""
    path = _reports_dir() / f"{date_str}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_reports(last_n: int = 30) -> list[str]:
    """Return the last N report dates (YYYY-MM-DD) in reverse chronological order."""
    d = _reports_dir()
    files = sorted(d.glob("*.json"), reverse=True)
    return [f.stem for f in files[:last_n]]


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

async def generate_daily_report(date_str: str | None = None) -> dict:
    """Generate the full daily report for a given date (default: today UTC).

    Steps:
      1. Load trades for the date
      2. Fetch Bitget outcomes and match
      3. Compute metrics
      4. Run drift detection
      5. Send Telegram summary
      6. Persist to disk

    Returns the full report dict.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"[daily_report] Generating report for {date_str}", flush=True)

    # 1. Load trades
    trades = _trades_for_date(date_str)
    print(f"[daily_report] Found {len(trades)} trades for {date_str}", flush=True)

    # 2. Match outcomes
    bitget_rows = await _fetch_bitget_outcomes(days=3)
    trades = _match_outcomes(trades, bitget_rows)

    # 3. Compute metrics
    metrics = _compute_metrics(trades)

    # 4. Drift detection
    drift = _check_drift()

    # 5. Build report
    report = {
        "date": date_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "drift": drift,
        "trades": trades,
    }

    # 6. Send Telegram
    try:
        tg_text = _format_report_telegram(date_str, metrics, drift)
        await _send_telegram(tg_text)
    except Exception as e:
        print(f"[daily_report] Telegram notification err: {e}", flush=True)

    # 7. Persist
    path = _save_report(date_str, report)
    print(f"[daily_report] Saved to {path}", flush=True)

    return report


# ---------------------------------------------------------------------------
# Scan-loop hook: call daily at UTC midnight
# ---------------------------------------------------------------------------

_last_report_date: str = ""


async def check_daily_report() -> dict | None:
    """Gate function for the scan loop. Generates a report once per UTC day.

    Returns the report dict if generated, or None if already done today.
    Call this after every scan -- it's cheap (string comparison) when
    there's nothing to do.
    """
    global _last_report_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if today == _last_report_date:
        return None

    # Check if we already have a report for today (e.g. server restart)
    existing = load_report(today)
    if existing:
        _last_report_date = today
        return None

    # Generate yesterday's report (today just started, yesterday's trades are final)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    _last_report_date = today

    # Only generate if we haven't already
    if load_report(yesterday) is not None:
        return None

    try:
        return await generate_daily_report(yesterday)
    except Exception as e:
        print(f"[daily_report] check_daily_report err: {e}", flush=True)
        return None


__all__ = [
    "generate_daily_report",
    "check_daily_report",
    "load_report",
    "list_reports",
]
