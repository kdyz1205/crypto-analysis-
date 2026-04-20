"""Retrofit v2: re-derive TRUE pivot-to-pivot anchors for pre-existing
filled/closed/broken trendline orders and write accurate lines into
`data/manual_trendlines.json`.

Why v2? The v1 retrofit (`retrofit_auto_lines.py`) used the approximation
  anchor1_bar = 0, anchor2_bar = bar_count - 1
which stretches the line across the full historical data window instead of
connecting the two real swing pivots. The rendered lines were visually wrong.

v2 algorithm per order:
  1. Re-fetch OHLCV for (symbol, timeframe) ending at created_ts with bar_count
     bars (+buffer), via `server.data_service.get_ohlcv`.
  2. Call `ts.research.pivots.find_pivots(df, k=3, min_reaction_atr=0.5)`.
  3. Filter pivots to the matching kind (support → lows, resistance → highs).
  4. Score each pivot by deviation from the stored line
       dev_i = |p.price - (slope*p.idx + intercept)| / atr_at_pivot_i
  5. Pick the two pivots with smallest deviation (both must be ≤ 1 ATR).
  6. i1 = min(idx), i2 = max(idx). Build manual line with:
       t_start = df.iloc[i1].timestamp       (unix seconds)
       t_end   = df.iloc[i2].timestamp
       price_start = slope*i1 + intercept
       price_end   = slope*i2 + intercept
     (use the line price, not the pivot price, so rendering exactly matches
      what the plan order was triggered against).

Behaviors:
  - source="manual" lines are NEVER touched.
  - source="auto_triggered" lines whose label contains "(retrofit)" (v1 output)
    are REMOVED first, then replaced by the v2 accurate lines.
  - Idempotent: re-running never double-inserts (dedupes by manual_line_id).
  - manual_line_id uses seconds (matching live `trendline_order_manager.
    _auto_line_id`) so future live writes deduplicate against v2 rows.

Failure modes (per spec):
  - If re-fetched OHLCV has ≠ bar_count after trimming, label gets "drift=N bars"
    tag and we continue (we still search pivots in whatever we got).
  - If no 2-pivot pair fits within 1 ATR tolerance, the order is marked
    "unmatched" and skipped — we do NOT write an inaccurate line.
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# On Windows default stdout is cp1252, which can't encode e.g. the middle-dot
# label character. Force stdout/stderr to UTF-8 so prints don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure we can import from both crypto-analysis- and trading-system
ROOT = Path(__file__).resolve().parent.parent
TRADING_SYSTEM = Path(r"C:/Users/alexl/trading-system")
for p in (str(ROOT), str(TRADING_SYSTEM)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ACTIVE = ROOT / "data" / "trendline_active_orders.json"
MANUAL = ROOT / "data" / "manual_trendlines.json"

TF_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}

# Pivot-fit tolerance. A candidate pair is accepted only if BOTH pivots sit
# within this many ATRs of the stored line. v2 spec = 1.0.
PIVOT_FIT_TOL_ATR = 1.0

# Kind filter eligibility
ELIGIBLE_STATUS = {"filled", "closed", "broken"}


def _ts_to_end_time_string(ts: float) -> str:
    """UTC 'YYYY-MM-DDTHH:MM' for polars.to_datetime on the server."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _required_days(bar_count: int, tf_seconds: int, created_ts: float) -> int:
    """Days-back-from-now needed so the fetch window covers `bar_count` bars
    ending at created_ts. `get_ohlcv(days=D, history_mode="fast_window")` gives
    the last D days counted from the latest bar in the source (≈ now), then
    filters by end_time. So we need D ≥ (now - (created_ts - bc*tf)) / 86400."""
    now = time.time()
    want_first_ts = created_ts - bar_count * tf_seconds
    span = (now - want_first_ts) / 86400.0
    return max(2, int(math.ceil(span + 2.0)))  # +2-day buffer


async def _fetch_window(symbol: str, tf: str, created_ts: float, bar_count: int) -> pd.DataFrame | None:
    """Return a DataFrame with columns [timestamp, open, high, low, close] of
    (at most) the last `bar_count` bars ending at created_ts. timestamps are
    unix seconds (UTC).

    None on fetch failure.
    """
    from server.data_service import get_ohlcv

    tf_sec = TF_SECONDS.get(tf)
    if tf_sec is None:
        return None
    days = _required_days(bar_count, tf_sec, created_ts)
    end_time = _ts_to_end_time_string(created_ts)
    try:
        data = await get_ohlcv(
            symbol, tf, end_time=end_time, days=days, history_mode="fast_window"
        )
    except Exception as exc:
        print(f"[retrofit_v2] fetch err {symbol} {tf}: {exc}", flush=True)
        return None
    candles = data.get("candles") or []
    if not candles:
        return None
    # Trim to the last bar_count bars (end_time already caps the right edge)
    candles = candles[-bar_count:]
    df = pd.DataFrame(candles).rename(columns={"time": "timestamp"})
    # Defensive cast
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def _pick_best_pivot_pair(
    df: pd.DataFrame, slope: float, intercept: float, kind: str
) -> tuple[int, int, float, float] | None:
    """Find the 2 pivots of the given `kind` that best fit `slope*x + intercept`.

    kind == "support"    → we look at swing LOWS
    kind == "resistance" → we look at swing HIGHS

    Returns (i1, i2, dev1_atr, dev2_atr) with i1 < i2 OR None if nothing fits.
    """
    from ts.research.pivots import find_pivots  # lazy import so script stays cheap

    try:
        pivots = find_pivots(df, k=3, min_reaction_atr=0.5)
    except Exception as exc:
        print(f"[retrofit_v2] find_pivots err: {exc}", flush=True)
        return None
    if not pivots:
        return None

    want = "low" if kind == "support" else "high"
    cand = [p for p in pivots if p.kind == want]
    if len(cand) < 2:
        return None

    # Deviation from line in ATR units, per-pivot
    scored: list[tuple[float, Any]] = []
    for p in cand:
        proj = slope * p.idx + intercept
        atr = float(p.atr_at_pivot) if p.atr_at_pivot > 0 else 0.0
        if atr <= 0:
            continue
        dev = abs(float(p.price) - proj) / atr
        scored.append((dev, p))
    scored.sort(key=lambda x: x[0])

    # Grab the best 2 distinct-bar pivots under the tolerance.
    chosen: list[Any] = []
    seen_idx: set[int] = set()
    for dev, p in scored:
        if dev > PIVOT_FIT_TOL_ATR:
            break
        if p.idx in seen_idx:
            continue
        chosen.append((dev, p))
        seen_idx.add(p.idx)
        if len(chosen) >= 2:
            break
    if len(chosen) < 2:
        return None

    pa, pb = chosen[0][1], chosen[1][1]
    if pa.idx <= pb.idx:
        return int(pa.idx), int(pb.idx), float(chosen[0][0]), float(chosen[1][0])
    return int(pb.idx), int(pa.idx), float(chosen[1][0]), float(chosen[0][0])


def _auto_line_id(symbol: str, tf: str, kind: str, a1_ts: int, a2_ts: int, order_id: str) -> str:
    """Match live `server.strategy.trendline_order_manager._auto_line_id`.

    Live code formats as: auto-{SYM}-{tf}-{kind}-{ts1}-{ts2}-{oid[:16]} with
    ts = bars["t"][bar], which is unix SECONDS. We keep the same format so
    a future live fill for this order deduplicates against our retrofit row.
    """
    return (
        f"auto-{symbol.upper()}-{tf}-{kind}"
        f"-{int(a1_ts)}-{int(a2_ts)}"
        f"-{str(order_id or '')[:16]}"
    )


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[retrofit_v2] load {path} err: {exc}", flush=True)
        return []


def _atomic_save(path: Path, data: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _is_v1_retrofit(line: dict) -> bool:
    """True if this row was written by `retrofit_auto_lines.py` (v1)."""
    if line.get("source") != "auto_triggered":
        return False
    label = line.get("label") or ""
    return "(retrofit)" in label


async def run() -> int:
    orders = _load_json(ACTIVE)
    lines = _load_json(MANUAL)
    print(f"[retrofit_v2] loaded {len(orders)} active orders, {len(lines)} manual lines", flush=True)

    # Step 1: strip v1 retrofit entries. Keep everything else (manual lines,
    # future-live auto_triggered lines with proper labels, etc).
    removed_v1 = sum(1 for l in lines if _is_v1_retrofit(l))
    lines = [l for l in lines if not _is_v1_retrofit(l)]
    print(f"[retrofit_v2] removed {removed_v1} v1-retrofit entries", flush=True)

    # Step 2: determine candidates.
    candidates = [
        o for o in orders
        if (o.get("status") in ELIGIBLE_STATUS)
        and int(o.get("bar_count") or 0) > 1
        and (o.get("symbol") and o.get("timeframe") and o.get("kind"))
    ]
    print(f"[retrofit_v2] {len(candidates)} candidate orders (status in {sorted(ELIGIBLE_STATUS)})", flush=True)


    # Step 3: group by (symbol, timeframe, created_ts, bar_count) so we fetch
    # each data window only once even if multiple orders share it.
    by_window: dict[tuple[str, str, int, int], list[dict]] = {}
    for o in candidates:
        key = (
            (o.get("symbol") or "").upper(),
            o.get("timeframe") or "",
            int(o.get("created_ts") or 0),
            int(o.get("bar_count") or 0),
        )
        by_window.setdefault(key, []).append(o)
    print(f"[retrofit_v2] grouped into {len(by_window)} unique (sym,tf,ts,bc) windows", flush=True)

    # Counters
    processed = 0
    rewritten = 0
    unmatched = 0
    errored = 0
    drift_notes: list[str] = []
    per_symbol = Counter()
    per_status = Counter()

    existing_ids = {l.get("manual_line_id") for l in lines}
    now_ms = int(time.time() * 1000)

    for (sym, tf, created_ts_int, bar_count), orders_here in by_window.items():
        created_ts = float(created_ts_int)
        try:
            df = await _fetch_window(sym, tf, created_ts, bar_count)
        except Exception as exc:
            print(f"[retrofit_v2] fetch exception {sym} {tf}: {exc}\n{traceback.format_exc()}", flush=True)
            errored += len(orders_here)
            processed += len(orders_here)
            continue
        if df is None or len(df) < 10:
            print(f"[retrofit_v2] no data for {sym} {tf} ts={created_ts_int} — skipping {len(orders_here)} order(s)", flush=True)
            errored += len(orders_here)
            processed += len(orders_here)
            continue

        drift = int(len(df)) - int(bar_count)
        if drift != 0:
            drift_notes.append(f"{sym}/{tf}/bc={bar_count}→{len(df)}")

        for o in orders_here:
            processed += 1
            kind = (o.get("kind") or "").lower()
            slope = float(o.get("slope") or 0.0)
            intercept = float(o.get("intercept") or 0.0)
            status = o.get("status") or "filled"
            oid = str(o.get("exchange_order_id") or "")

            pick = _pick_best_pivot_pair(df, slope, intercept, kind)
            if pick is None:
                unmatched += 1
                per_status[f"{kind}/unmatched"] += 1
                continue
            i1, i2, d1, d2 = pick
            # Bound the indices into df (defensive)
            i1 = max(0, min(i1, len(df) - 1))
            i2 = max(0, min(i2, len(df) - 1))
            if i1 == i2:
                unmatched += 1
                per_status[f"{kind}/unmatched_same_bar"] += 1
                continue
            t1 = int(df.iloc[i1]["timestamp"])
            t2 = int(df.iloc[i2]["timestamp"])
            p1 = float(slope * i1 + intercept)
            p2 = float(slope * i2 + intercept)
            if p1 <= 0 or p2 <= 0 or t1 <= 0 or t2 <= 0:
                unmatched += 1
                per_status[f"{kind}/bad_values"] += 1
                continue
            line_id = _auto_line_id(sym, tf, kind, t1, t2, oid)
            if line_id in existing_ids:
                # Already present (from a future live write or an earlier v2 run)
                per_status[f"{kind}/dedup"] += 1
                continue

            drift_tag = f" drift={drift:+d}b" if drift != 0 else ""
            # Spec-mandated label format (middle-dot U+00B7 separator)
            label = f"Auto \u00b7 {kind} \u00b7 {status} \u00b7 i1={i1} i2={i2}{drift_tag}"
            notes = (
                f"order_id={oid} status={status} "
                f"entry={o.get('limit_price'):.6g} stop={o.get('stop_price'):.6g} "
                f"tp={o.get('tp_price'):.6g} "
                f"slope={slope:.6g} intercept={intercept:.6g} "
                f"bar_count_order={bar_count} bar_count_fetched={len(df)} "
                f"fit_dev_atr=({d1:.3f},{d2:.3f}) "
                f"source=retrofit_v2"
            )
            entry = {
                "manual_line_id": line_id,
                "symbol": sym,
                "timeframe": tf,
                "side": kind,  # 'support' | 'resistance'
                "source": "auto_triggered",
                "t_start": t1,
                "t_end": t2,
                "price_start": p1,
                "price_end": p2,
                "extend_left": False,
                "extend_right": True,
                "locked": True,
                "label": label,
                "notes": notes,
                "comparison_status": "uncompared",
                "override_mode": "display_only",
                "nearest_auto_line_id": None,
                "slope_diff": None,
                "projected_price_diff": None,
                "overlap_ratio": None,
                "created_at": int(created_ts * 1000),
                "updated_at": now_ms,
                "line_width": 1.8,
            }
            lines.append(entry)
            existing_ids.add(line_id)
            rewritten += 1
            per_symbol[sym] += 1

    _atomic_save(MANUAL, lines)

    # ---- Report ----
    print(f"\n[retrofit_v2] === SUMMARY ===", flush=True)
    print(f"  removed v1-retrofit entries: {removed_v1}", flush=True)
    print(f"  candidates processed:        {processed}", flush=True)
    print(f"  rows rewritten (accurate):   {rewritten}", flush=True)
    print(f"  rows unmatched (skipped):    {unmatched}", flush=True)
    print(f"  rows errored (fetch/etc):    {errored}", flush=True)
    print(f"  manual_trendlines total:     {len(lines)}", flush=True)
    if drift_notes:
        print(f"  drift encountered on {len(drift_notes)} windows (first 5): {drift_notes[:5]}", flush=True)
    print(f"\n  per-symbol rewritten (top 15):", flush=True)
    for sym, n in per_symbol.most_common(15):
        print(f"    {sym}: {n}", flush=True)
    if per_status:
        print(f"\n  skip reasons:", flush=True)
        for reason, n in per_status.most_common():
            print(f"    {reason}: {n}", flush=True)
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
