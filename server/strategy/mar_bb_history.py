"""Historical trade statistics for the MA Ribbon + Trendline runner.

Reads closed positions from Bitget (`/api/v2/mix/position/history-position`)
and aggregates into:

  - overall: total trades, wins, losses, winrate, total pnl, total fees,
             net pnl, avg win, avg loss
  - by strategy (mar_bb / trendline / other): same breakdown
  - daily series: [{date, pnl, fee, trades}]
  - weekly rollup: [{week_start, pnl, fee, trades}]
  - per-symbol: top 10 by |pnl|

Strategy attribution uses our clientOid prefix convention:
  - "marbb_" → tagged as mar_bb
  - "line_"  → tagged as trendline (from the line-order path)
  - "cond_"  → tagged as conditional (from /api/conditionals)
  - else     → "other" (manual trades, RAVE, anything we didn't fire)
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from server.execution.live_adapter import LiveExecutionAdapter


def _tag_strategy(client_oid: str | None, order_source: str | None = None) -> str:
    """Classify a closed position by our clientOid prefix convention.

    Prefixes we emit (see trendline_order_manager + mar_bb_runner + watcher):
      tl_        — automatic runner trendline plan (the main strategy)
      marbb_     — MA Ribbon breakout plan
      line_      — user-drawn line order, initial placement
      replan_    — conditional watcher re-placing a line after trigger/cancel
      cond_      — conditional watcher's own side-orders
      rev_       — reverse/auto-flip after SL
    """
    coid = (client_oid or "").lower()
    if coid.startswith("tl_"):
        return "trendline_auto"
    if coid.startswith("marbb_"):
        return "mar_bb"
    if coid.startswith("line_"):
        return "trendline_manual"
    if coid.startswith("replan_"):
        return "trendline_replan"
    if coid.startswith("cond_"):
        return "conditional"
    if coid.startswith("rev_"):
        return "reverse"
    return "other"


async def fetch_position_history(
    symbol: str | None = None,
    days: int = 30,
    mode: str = "live",
) -> list[dict[str, Any]]:
    """Pull closed positions from Bitget for the last `days`.

    Bitget's `history-position` endpoint returns records with at least:
      symbol, holdSide, openAvgPrice, closeAvgPrice, closeTotalPos,
      netProfit, openFee, closeFee, utime (ms), cTime, uTime, pnl, ...

    Returns the raw list of dicts, already trimmed to within `days`.
    """
    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return []

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    params: dict[str, Any] = {
        "productType": "USDT-FUTURES",
        "startTime": str(start_ms),
        "endTime": str(now_ms),
        "limit": "100",
    }
    if symbol:
        params["symbol"] = symbol.upper()

    try:
        resp = await adapter._bitget_request(
            "GET", "/api/v2/mix/position/history-position",
            mode=mode, params=params,
        )
    except Exception as e:
        print(f"[mar_bb_history] bitget fetch err: {e}", flush=True)
        return []

    if resp.get("code") != "00000":
        print(f"[mar_bb_history] bitget error: {resp.get('msg')}", flush=True)
        return []

    data = resp.get("data") or {}
    rows = data.get("list") or data.get("entrustedList") or data
    if not isinstance(rows, list):
        return []
    return rows


def aggregate_history(rows: list[dict]) -> dict:
    """Aggregate raw history rows into the stats dict the UI consumes."""
    empty = {
        "overall": {
            "trades": 0, "wins": 0, "losses": 0, "breakeven": 0,
            "winrate": 0.0, "total_pnl": 0.0, "total_fees": 0.0,
            "net_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "best": None, "worst": None,
        },
        "by_strategy": {},
        "daily": [],
        "weekly": [],
        "by_symbol": [],
    }
    if not rows:
        return empty

    # ─── Normalize each row ───
    parsed: list[dict] = []
    for r in rows:
        try:
            close_ts_ms = int(r.get("uTime") or r.get("updateTime") or r.get("utime") or r.get("cTime") or 0)
            if close_ts_ms == 0:
                continue
            pnl = float(r.get("netProfit") or r.get("net_pnl") or r.get("pnl") or 0)
            open_fee = float(r.get("openFee") or r.get("open_fee") or 0)
            close_fee = float(r.get("closeFee") or r.get("close_fee") or 0)
            total_fee = abs(open_fee) + abs(close_fee)
            side = (r.get("holdSide") or r.get("posSide") or "").lower()
            symbol = (r.get("symbol") or "").upper()
            client_oid = r.get("clientOid") or r.get("clientOId") or r.get("client_oid")
            parsed.append({
                "ts_ms": close_ts_ms,
                "ts_s": close_ts_ms // 1000,
                "date": datetime.fromtimestamp(close_ts_ms / 1000, tz=timezone.utc).date().isoformat(),
                "symbol": symbol,
                "side": side,
                "pnl": pnl,
                "fee": total_fee,
                "net": pnl,   # bitget netProfit already excludes fees
                "strategy": _tag_strategy(client_oid),
                "open_price": float(r.get("openAvgPrice") or 0),
                "close_price": float(r.get("closeAvgPrice") or 0),
                "size": float(r.get("closeTotalPos") or 0),
            })
        except Exception:
            continue

    if not parsed:
        return empty

    # ─── Overall ───
    def _overall(items: list[dict]) -> dict:
        if not items:
            return {
                "trades": 0, "wins": 0, "losses": 0, "breakeven": 0,
                "winrate": 0.0, "total_pnl": 0.0, "total_fees": 0.0,
                "net_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "best": None, "worst": None,
            }
        wins = [x for x in items if x["net"] > 0]
        losses = [x for x in items if x["net"] < 0]
        breakeven = [x for x in items if x["net"] == 0]
        total_pnl = sum(x["net"] for x in items)
        total_fees = sum(x["fee"] for x in items)
        best = max(items, key=lambda x: x["net"])
        worst = min(items, key=lambda x: x["net"])
        return {
            "trades": len(items),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "winrate": round(len(wins) / len(items) * 100, 2) if items else 0,
            "total_pnl": round(total_pnl, 4),
            "total_fees": round(total_fees, 4),
            "net_pnl": round(total_pnl - total_fees, 4),
            "avg_win": round(sum(x["net"] for x in wins) / len(wins), 4) if wins else 0,
            "avg_loss": round(sum(x["net"] for x in losses) / len(losses), 4) if losses else 0,
            "best": {"symbol": best["symbol"], "pnl": round(best["net"], 4), "date": best["date"]},
            "worst": {"symbol": worst["symbol"], "pnl": round(worst["net"], 4), "date": worst["date"]},
        }

    overall = _overall(parsed)

    # ─── By strategy ───
    by_strategy: dict[str, dict] = {}
    for strat in {x["strategy"] for x in parsed}:
        subset = [x for x in parsed if x["strategy"] == strat]
        by_strategy[strat] = _overall(subset)

    # ─── Daily series ───
    daily_acc: dict[str, dict] = {}
    for x in parsed:
        d = x["date"]
        if d not in daily_acc:
            daily_acc[d] = {"date": d, "pnl": 0.0, "fee": 0.0, "trades": 0,
                            "wins": 0, "net": 0.0}
        daily_acc[d]["pnl"] += x["net"]
        daily_acc[d]["fee"] += x["fee"]
        daily_acc[d]["trades"] += 1
        if x["net"] > 0:
            daily_acc[d]["wins"] += 1
    daily = sorted(daily_acc.values(), key=lambda x: x["date"])
    # Build running-equity curve + round
    running = 0.0
    for d in daily:
        d["pnl"] = round(d["pnl"], 4)
        d["fee"] = round(d["fee"], 4)
        d["net"] = round(d["pnl"] - d["fee"], 4)
        running += d["net"]
        d["cumulative"] = round(running, 4)

    # ─── Weekly rollup (iso week) ───
    weekly_acc: dict[str, dict] = {}
    for x in parsed:
        dt = datetime.fromtimestamp(x["ts_ms"] / 1000, tz=timezone.utc)
        year, week, _ = dt.isocalendar()
        key = f"{year}-W{week:02d}"
        if key not in weekly_acc:
            # Week start (Monday) for sorting
            week_start = dt - timedelta(days=dt.weekday())
            weekly_acc[key] = {
                "week": key,
                "week_start": week_start.date().isoformat(),
                "pnl": 0.0, "fee": 0.0, "trades": 0, "wins": 0,
            }
        weekly_acc[key]["pnl"] += x["net"]
        weekly_acc[key]["fee"] += x["fee"]
        weekly_acc[key]["trades"] += 1
        if x["net"] > 0:
            weekly_acc[key]["wins"] += 1
    weekly = sorted(weekly_acc.values(), key=lambda x: x["week_start"])
    for w in weekly:
        w["pnl"] = round(w["pnl"], 4)
        w["fee"] = round(w["fee"], 4)
        w["net"] = round(w["pnl"] - w["fee"], 4)

    # ─── Per-symbol (top 10 by |pnl|) ───
    sym_acc: dict[str, dict] = {}
    for x in parsed:
        s = x["symbol"]
        if s not in sym_acc:
            sym_acc[s] = {"symbol": s, "pnl": 0.0, "fee": 0.0, "trades": 0, "wins": 0}
        sym_acc[s]["pnl"] += x["net"]
        sym_acc[s]["fee"] += x["fee"]
        sym_acc[s]["trades"] += 1
        if x["net"] > 0:
            sym_acc[s]["wins"] += 1
    by_symbol = sorted(sym_acc.values(), key=lambda x: abs(x["pnl"]), reverse=True)[:10]
    for s in by_symbol:
        s["pnl"] = round(s["pnl"], 4)
        s["fee"] = round(s["fee"], 4)
        s["net"] = round(s["pnl"] - s["fee"], 4)
        s["winrate"] = round(s["wins"] / s["trades"] * 100, 2) if s["trades"] else 0

    return {
        "overall": overall,
        "by_strategy": by_strategy,
        "daily": daily,
        "weekly": weekly,
        "by_symbol": by_symbol,
    }


async def get_history(days: int = 30, symbol: str | None = None, mode: str = "live") -> dict:
    rows = await fetch_position_history(symbol=symbol, days=days, mode=mode)
    stats = aggregate_history(rows)
    stats["meta"] = {
        "days": days,
        "symbol": symbol,
        "fetched_at": int(time.time()),
        "raw_count": len(rows),
    }
    return stats


__all__ = ["fetch_position_history", "aggregate_history", "get_history"]
