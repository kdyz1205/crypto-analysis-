"""Retrofit: compute anchor_ts for pre-existing filled trendline orders and
write them as source=auto_triggered lines into manual_trendlines.json.

Context: `_persist_auto_line` is called inside update_trendline_orders when
a plan transitions placed → filled. Orders that went filled BEFORE the
persist code deployed (2026-04-19) never run through that branch again,
so their anchor data never reaches manual_trendlines.json. This script
computes an approximate anchor_ts from created_ts + tf + anchor bars, and
retrofits the entries so the user can see them on the chart.

Approximation used:
  - At the moment the line was detected, anchor2_bar == last bar of
    bars["c"] (line_entry_bar in _check_trendline_signal).
  - So anchor2_ts ≈ created_ts (unix seconds).
  - anchor1_bar is typically 0 (see mar_bb_runner.py _trendline_limit_signals).
  - bars spanned bar_count bars back from anchor2. Therefore:
      anchor1_ts = anchor2_ts - (bar_count - 1) * TF_SECONDS
  - Prices: slope * bar + intercept (already stored per order).

This is a one-shot, idempotent migration. Re-running it will not double
-insert (helper dedupes by manual_line_id).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
ACTIVE = ROOT / "data" / "trendline_active_orders.json"
MANUAL = ROOT / "data" / "manual_trendlines.json"

TF_SECONDS = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}


def retrofit_anchors(order: dict) -> tuple[int, int, float, float] | None:
    """Returns (anchor1_ts_ms, anchor2_ts_ms, anchor1_price, anchor2_price)
    approximated from created_ts + slope/intercept + bar_count.
    None if we cannot compute (missing data)."""
    tf = order.get("timeframe") or ""
    secs = TF_SECONDS.get(tf)
    if secs is None:
        return None
    bar_count = int(order.get("bar_count") or 0)
    if bar_count <= 1:
        return None
    slope = float(order.get("slope") or 0.0)
    intercept = float(order.get("intercept") or 0.0)
    created = float(order.get("created_ts") or 0)
    if created <= 0:
        return None
    a1_bar = int(order.get("anchor1_bar") or 0)
    a2_bar = int(order.get("anchor2_bar") or (bar_count - 1))
    # anchor2 at the scan time = the last bar → created_ts
    a2_ts_sec = int(created)
    a1_ts_sec = int(a2_ts_sec - max(1, a2_bar - a1_bar) * secs)
    a1_price = float(slope * a1_bar + intercept)
    a2_price = float(slope * a2_bar + intercept)
    if a1_price <= 0 or a2_price <= 0:
        return None
    # Return in ms (manual_trendlines.json stores ts in seconds for t_start/t_end
    # but created_at/updated_at in ms — keep consistent with the helper).
    return a1_ts_sec, a2_ts_sec, a1_price, a2_price


def _auto_line_id(symbol: str, timeframe: str, kind: str, a1_ts: int, a2_ts: int, order_id: str) -> str:
    # Same id convention as server.strategy.trendline_order_manager._auto_line_id
    # But using ms * 1000 to match the runtime code (it uses anchor1_ts in ms).
    # The runtime code stores anchor1_ts as ms-since-epoch (int(ts_ms)). Retrofit
    # must produce the same format so dedup works if the live code fires later
    # for the same order.
    return (
        f"auto-{symbol.upper()}-{timeframe}-{kind}"
        f"-{a1_ts * 1000}-{a2_ts * 1000}"
        f"-{str(order_id or '')[:16]}"
    )


def main() -> int:
    if not ACTIVE.exists():
        print("no active_orders file")
        return 0
    active = json.loads(ACTIVE.read_text(encoding="utf-8"))
    manual = json.loads(MANUAL.read_text(encoding="utf-8")) if MANUAL.exists() else []
    existing_ids = {l.get("manual_line_id") for l in manual}

    # Candidates: orders with status in {filled, closed, broken} and bar_count > 0.
    candidates = [o for o in active if o.get("status") in ("filled", "closed", "broken")
                  and (o.get("bar_count") or 0) > 1]
    print(f"candidate orders (filled/closed/broken): {len(candidates)}")

    added = 0
    skipped = 0
    for o in candidates:
        sym = (o.get("symbol") or "").upper()
        tf = o.get("timeframe") or ""
        kind = o.get("kind") or ""
        if not sym or not tf or not kind:
            skipped += 1
            continue
        approx = retrofit_anchors(o)
        if approx is None:
            skipped += 1
            continue
        a1_ts_sec, a2_ts_sec, a1_px, a2_px = approx
        oid = str(o.get("exchange_order_id") or "")
        line_id = _auto_line_id(sym, tf, kind, a1_ts_sec, a2_ts_sec, oid)
        if line_id in existing_ids:
            skipped += 1
            continue
        # Ensure t_start <= t_end ordering
        t1, p1 = a1_ts_sec, a1_px
        t2, p2 = a2_ts_sec, a2_px
        if t2 < t1:
            t1, t2 = t2, t1
            p1, p2 = p2, p1
        # created_at/updated_at use UNIX SECONDS to match the rest of
        # manual_trendlines.json. Frontend's fmtDateShort does
        # `new Date(ts * 1000)` so feeding milliseconds crashes the year.
        now_ts = int(time.time())
        status = o.get("status")
        label = f"Auto · {kind[:3]} · {status} (retrofit)"
        entry = {
            "manual_line_id": line_id,
            "symbol": sym,
            "timeframe": tf,
            "side": kind,
            "source": "auto_triggered",
            "t_start": t1,
            "t_end": t2,
            "price_start": p1,
            "price_end": p2,
            "extend_left": False,
            "extend_right": True,
            "locked": True,
            "label": label,
            "notes": (
                f"retrofit from pre-2026-04-19 active_order; "
                f"order_id={oid} entry≈{o.get('limit_price')} "
                f"stop≈{o.get('stop_price')} tp≈{o.get('tp_price')} "
                f"status={status} slope={o.get('slope'):.6g} intercept={o.get('intercept'):.6g}"
            ),
            "comparison_status": "uncompared",
            "override_mode": "display_only",
            "nearest_auto_line_id": None,
            "slope_diff": None,
            "projected_price_diff": None,
            "overlap_ratio": None,
            "created_at": int(o.get("created_ts") or time.time()),
            "updated_at": now_ts,
            "line_width": 1.8,
        }
        manual.append(entry)
        existing_ids.add(line_id)
        added += 1

    tmp = MANUAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(manual, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANUAL)

    print(f"added: {added}  skipped: {skipped}")
    from collections import Counter
    kinds = Counter((l.get("symbol"), l.get("timeframe"), l.get("side")) for l in manual if l.get("source") == "auto_triggered")
    print(f"auto_triggered by (symbol,tf,side) top 10:")
    for k, n in kinds.most_common(10):
        print(f"  {k}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
