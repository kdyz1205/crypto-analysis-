"""Retrofit v3: validated pivot-anchor retrofit for `trendline_active_orders.json`.

Why v3?
  v2 (`retrofit_auto_lines_v2.py`) picked the two pivots with smallest normalized
  distance to the stored (slope, intercept). That makes the anchors fit the line
  ALGEBRAICALLY — but classical TA requires the segment between them NOT to be
  sliced by intermediate candles. ENAUSDT 1h screenshot showed a retrofit line
  that cut through multiple candle bodies between its two anchors → invalid.

v3 fixes this by adding a hard VALIDATION step:

  For a support line:
    - Candidate anchors: two pivot LOWS with small normalized distance to the line.
    - For EVERY bar b in (i1..i2) inclusive:
        violation = (line_price - df.low[b]) / atr[b]
        (how far below the line the bar's low dips, in ATR units)
      If any violation > VIOLATION_TOL_ATR (default 0.2) → the line is BROKEN
      BETWEEN its anchors → REJECT this pair, try the next-best pair.

  For a resistance line:
    - Candidate anchors: two pivot HIGHS.
    - violation = (df.high[b] - line_price) / atr[b]
    - Same rejection threshold.

  If NO pair validates → we still write the line but with:
    locked        = False           (user can delete it)
    override_mode = "display_only"
    label         = "Auto · <kind> · <status> (unmatched)"
    notes.*       = list of bar indices that violate (debug)

  Valid lines get:
    locked        = True
    override_mode = "display_only"
    label         = "Auto · <kind> · <status> · i1=… i2=… dev=(…)"

Behaviors
  - source="manual" lines are NEVER touched (P4).
  - source="auto_triggered" lines are removed en masse (any label pattern) so this
    script is idempotent and cleans up stale v1/v2 output.
  - `ts.research.pivots.compute_atr` is the ATR source (per spec).
  - Data fetch path: `server.data_service.get_ohlcv(symbol, tf, end_time, days,
    history_mode="fast_window")`.

Output fields (unchanged from v2 except label/notes/locked for unmatched branch).
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows so prints with U+00B7 (middle dot) don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
}

# --- Tuning ---
# Candidate pair inclusion: both pivots must fit the line within this ATR tolerance.
# Kept equal to v2 so valid pairs are never rejected up-front.
PIVOT_FIT_TOL_ATR = 1.0

# Line validity: within (i1, i2), no bar's low (support) / high (resistance) may
# deviate beyond the line by more than this many ATRs. Spec says 0.2.
VIOLATION_TOL_ATR = 0.2

ELIGIBLE_STATUS = {"filled", "closed", "broken"}

# Pivot search params (from ts.strategies.trendline_retest defaults — Phase 1.75)
PIVOT_K = 3
MIN_REACTION_ATR = 0.5
ATR_PERIOD = 14


def _ts_to_end_time_string(ts: float) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _required_days(bar_count: int, tf_seconds: int, created_ts: float) -> int:
    now = time.time()
    want_first_ts = created_ts - bar_count * tf_seconds
    span = (now - want_first_ts) / 86400.0
    return max(2, int(math.ceil(span + 2.0)))


async def _fetch_window(symbol: str, tf: str, created_ts: float, bar_count: int) -> pd.DataFrame | None:
    """Return the last `bar_count` bars ending at `created_ts` as a DataFrame with
    columns [timestamp (unix sec), open, high, low, close]."""
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
        print(f"[retrofit_v3] fetch err {symbol} {tf}: {exc}", flush=True)
        return None
    candles = data.get("candles") or []
    if not candles:
        return None
    candles = candles[-bar_count:]
    df = pd.DataFrame(candles).rename(columns={"time": "timestamp"})
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def _validate_line(
    df: pd.DataFrame,
    atr: np.ndarray,
    slope: float,
    intercept: float,
    i1: int,
    i2: int,
    kind: str,
) -> tuple[bool, list[tuple[int, float]]]:
    """For every bar b in [i1, i2], check whether the bar's body breaches the
    line. Returns (is_valid, violations) where violations is a list of
    (bar_idx, violation_atr) with violation_atr > VIOLATION_TOL_ATR.

    The anchors i1 and i2 themselves are excluded from the violation check
    (they're ON the line by construction — numerical noise shouldn't reject
    the whole pair).
    """
    lows = df["low"].values
    highs = df["high"].values
    n = len(df)
    lo = max(0, min(i1, i2))
    hi = min(n - 1, max(i1, i2))
    violations: list[tuple[int, float]] = []
    if hi - lo < 2:
        return True, violations  # trivially valid (no middle bars)

    # Build line prices across the segment in one vectorized op
    bars = np.arange(lo + 1, hi, dtype=np.int64)  # exclusive of anchors
    line_p = slope * bars + intercept
    atr_seg = atr[bars]
    if kind == "support":
        # Low below the line → breaks support
        viol = (line_p - lows[bars]) / atr_seg
    else:
        # High above the line → breaks resistance
        viol = (highs[bars] - line_p) / atr_seg

    # Treat invalid ATR (NaN / <=0) as non-violating: we'd rather accept than
    # false-reject when ATR isn't computable (beginning of the series).
    viol = np.where(np.isfinite(viol), viol, 0.0)

    bad = viol > VIOLATION_TOL_ATR
    if np.any(bad):
        idxs = np.where(bad)[0]
        for k in idxs:
            violations.append((int(bars[k]), float(viol[k])))

    return len(violations) == 0, violations


def _score_pivot_pairs(
    pivots: list,
    slope: float,
    intercept: float,
    want_kind: str,
) -> list[tuple[float, float, float, Any, Any]]:
    """Return sorted list of candidate pivot pairs for the stored line.

    Each entry is (combined_dev_atr, dev1, dev2, p_lower_idx, p_higher_idx).
    Only pairs where BOTH pivot devs ≤ PIVOT_FIT_TOL_ATR are returned.
    Sorted ascending by max(dev1, dev2) then sum — so the tightest-fit pair
    is tried first, breaking ties by symmetry.
    """
    want = "low" if want_kind == "support" else "high"
    cand = [p for p in pivots if p.kind == want and p.atr_at_pivot > 0]
    if len(cand) < 2:
        return []

    # Per-pivot deviation from the line
    scored = []
    for p in cand:
        proj = slope * p.idx + intercept
        dev = abs(float(p.price) - proj) / float(p.atr_at_pivot)
        if dev <= PIVOT_FIT_TOL_ATR:
            scored.append((dev, p))
    scored.sort(key=lambda x: x[0])

    # Build all pairs (i < j by index) among scored pivots
    pairs = []
    for i, (d1, p1) in enumerate(scored):
        for (d2, p2) in scored[i + 1:]:
            if p1.idx == p2.idx:
                continue
            pa, pb = (p1, p2) if p1.idx < p2.idx else (p2, p1)
            da, db = (d1, d2) if p1.idx < p2.idx else (d2, d1)
            # Prefer pairs that are well-separated (>= 5 bars apart) — otherwise
            # a pair of adjacent swing lows gives a near-flat/degenerate segment
            # with no real reaction arc between them.
            if pb.idx - pa.idx < 5:
                continue
            pairs.append((max(da, db), da, db, pa, pb))
    # Tightest fit first; within that, biggest anchor span next (more "history").
    pairs.sort(key=lambda x: (x[0], x[1] + x[2], -(x[4].idx - x[3].idx)))
    return pairs


def _pick_best_valid_pair(
    df: pd.DataFrame,
    slope: float,
    intercept: float,
    kind: str,
) -> tuple[tuple[int, int, float, float] | None, list[tuple[int, float]]]:
    """Main selection routine.

    Returns (pick, last_violations):
      pick = (i1, i2, dev1, dev2)  if a valid pair exists
      pick = None                  if none validate; caller should fallback
                                   to display_only + unmatched, using i1/i2
                                   from the TIGHTEST-FIT pair (we still return
                                   its violation list so the caller can mark
                                   debug info).
    """
    from ts.research.pivots import compute_atr, find_pivots

    try:
        pivots = find_pivots(df, k=PIVOT_K, min_reaction_atr=MIN_REACTION_ATR)
    except Exception as exc:
        print(f"[retrofit_v3] find_pivots err: {exc}", flush=True)
        return None, []
    if not pivots:
        return None, []

    atr = compute_atr(df, ATR_PERIOD).values

    candidate_pairs = _score_pivot_pairs(pivots, slope, intercept, kind)
    if not candidate_pairs:
        return None, []

    best_unmatched_violations: list[tuple[int, float]] = []
    best_unmatched_pair: tuple[int, int, float, float] | None = None

    for combined, d1, d2, pa, pb in candidate_pairs:
        ok, violations = _validate_line(df, atr, slope, intercept, pa.idx, pb.idx, kind)
        if ok:
            return (int(pa.idx), int(pb.idx), float(d1), float(d2)), []
        if best_unmatched_pair is None:
            best_unmatched_pair = (int(pa.idx), int(pb.idx), float(d1), float(d2))
            best_unmatched_violations = violations

    # None validated. We return None for pick to signal "unmatched" — caller
    # writes a display_only line but we still give it the tightest pair so
    # it has anchors to render from.
    return None, best_unmatched_violations if best_unmatched_pair else []


def _tightest_pair_or_none(
    df: pd.DataFrame,
    slope: float,
    intercept: float,
    kind: str,
) -> tuple[int, int, float, float] | None:
    """Used for the unmatched fallback to get anchors anyway."""
    from ts.research.pivots import find_pivots

    try:
        pivots = find_pivots(df, k=PIVOT_K, min_reaction_atr=MIN_REACTION_ATR)
    except Exception:
        return None
    pairs = _score_pivot_pairs(pivots, slope, intercept, kind)
    if not pairs:
        return None
    combined, d1, d2, pa, pb = pairs[0]
    return int(pa.idx), int(pb.idx), float(d1), float(d2)


def _auto_line_id(
    symbol: str, tf: str, kind: str,
    a1_ts: int, a2_ts: int, order_id: str,
) -> str:
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
        print(f"[retrofit_v3] load {path} err: {exc}", flush=True)
        return []


def _atomic_save(path: Path, data: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _is_prior_auto_retrofit(line: dict) -> bool:
    """True if this row is any kind of auto_triggered retrofit (v1, v2, v3)
    that we should evict on re-run. Manual lines (source='manual') are
    always preserved."""
    if line.get("source") != "auto_triggered":
        return False
    # Any auto_triggered line is considered a prior retrofit (we only have one
    # producer of these rows right now — the retrofit scripts). Covers:
    #   v1 label: "Auto · sup · filled (retrofit)"
    #   v2 label: "Auto · support · filled · i1=… i2=…"
    #   v3 label (valid): "Auto · support · filled · i1=… i2=…"
    #   v3 label (unmatched): "Auto · support · filled (unmatched)"
    return True


async def run() -> int:
    orders = _load_json(ACTIVE)
    lines = _load_json(MANUAL)
    print(f"[retrofit_v3] loaded {len(orders)} active orders, {len(lines)} manual lines", flush=True)

    # Step 1: evict ALL prior auto_triggered entries (v1/v2/v3), keep manual.
    n_manual_before = sum(1 for l in lines if l.get("source") == "manual")
    n_auto_before = sum(1 for l in lines if _is_prior_auto_retrofit(l))
    lines = [l for l in lines if not _is_prior_auto_retrofit(l)]
    print(
        f"[retrofit_v3] removed {n_auto_before} prior auto_triggered rows; "
        f"preserved {n_manual_before} manual rows",
        flush=True,
    )

    # Step 2: candidate orders
    candidates = [
        o for o in orders
        if (o.get("status") in ELIGIBLE_STATUS)
        and int(o.get("bar_count") or 0) > 1
        and (o.get("symbol") and o.get("timeframe") and o.get("kind"))
    ]
    print(
        f"[retrofit_v3] {len(candidates)} candidate orders "
        f"(status in {sorted(ELIGIBLE_STATUS)})",
        flush=True,
    )

    # Step 3: group by (symbol, tf, created_ts, bar_count) so we fetch each
    # window only once (multiple orders may share the same snapshot).
    by_window: dict[tuple[str, str, int, int], list[dict]] = {}
    for o in candidates:
        key = (
            (o.get("symbol") or "").upper(),
            o.get("timeframe") or "",
            int(o.get("created_ts") or 0),
            int(o.get("bar_count") or 0),
        )
        by_window.setdefault(key, []).append(o)
    print(f"[retrofit_v3] grouped into {len(by_window)} unique (sym,tf,ts,bc) windows", flush=True)

    # Counters
    processed = 0
    valid_written = 0
    unmatched_written = 0
    errored = 0
    skipped = 0
    per_symbol_valid = Counter()
    per_symbol_unmatched = Counter()
    skip_reasons = Counter()

    existing_ids = {l.get("manual_line_id") for l in lines}
    now_ms = int(time.time() * 1000)

    for (sym, tf, created_ts_int, bar_count), orders_here in by_window.items():
        created_ts = float(created_ts_int)
        try:
            df = await _fetch_window(sym, tf, created_ts, bar_count)
        except Exception as exc:
            print(
                f"[retrofit_v3] fetch exception {sym} {tf}: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
            errored += len(orders_here)
            processed += len(orders_here)
            continue
        if df is None or len(df) < 20:
            print(
                f"[retrofit_v3] no data for {sym} {tf} ts={created_ts_int} — skipping {len(orders_here)} order(s)",
                flush=True,
            )
            errored += len(orders_here)
            processed += len(orders_here)
            continue

        drift = int(len(df)) - int(bar_count)

        for o in orders_here:
            processed += 1
            kind = (o.get("kind") or "").lower()
            if kind not in ("support", "resistance"):
                skipped += 1
                skip_reasons[f"bad_kind={kind}"] += 1
                continue
            slope = float(o.get("slope") or 0.0)
            intercept = float(o.get("intercept") or 0.0)
            status = o.get("status") or "filled"
            oid = str(o.get("exchange_order_id") or "")

            pick, unmatched_violations = _pick_best_valid_pair(df, slope, intercept, kind)
            if pick is None:
                # No valid pair; still write as display_only fallback using the
                # tightest-fit pair so user can see+delete.
                fallback = _tightest_pair_or_none(df, slope, intercept, kind)
                if fallback is None:
                    skipped += 1
                    skip_reasons[f"{kind}/no_pivot_fit"] += 1
                    continue
                i1, i2, d1, d2 = fallback
                locked = False
                label_tag = "unmatched"
                per_symbol_unmatched[sym] += 1
                unmatched_written += 1
                viol_bars = [b for (b, _v) in unmatched_violations[:20]]
                viol_note = (
                    f"violations={len(unmatched_violations)} "
                    f"first_bars={viol_bars[:5]}"
                )
            else:
                i1, i2, d1, d2 = pick
                locked = True
                label_tag = f"i1={i1} i2={i2}"
                per_symbol_valid[sym] += 1
                valid_written += 1
                viol_note = "violations=0"

            # Clamp indices (defensive)
            i1 = max(0, min(i1, len(df) - 1))
            i2 = max(0, min(i2, len(df) - 1))
            if i1 == i2:
                skipped += 1
                skip_reasons[f"{kind}/same_bar"] += 1
                # Roll back the counters we bumped above
                if locked:
                    valid_written -= 1
                    per_symbol_valid[sym] -= 1
                else:
                    unmatched_written -= 1
                    per_symbol_unmatched[sym] -= 1
                continue

            t1 = int(df.iloc[i1]["timestamp"])
            t2 = int(df.iloc[i2]["timestamp"])
            # Line prices (NOT pivot prices) — so rendering matches the stored
            # slope/intercept the live plan was armed with.
            p1 = float(slope * i1 + intercept)
            p2 = float(slope * i2 + intercept)
            if p1 <= 0 or p2 <= 0 or t1 <= 0 or t2 <= 0:
                skipped += 1
                skip_reasons[f"{kind}/bad_values"] += 1
                if locked:
                    valid_written -= 1
                    per_symbol_valid[sym] -= 1
                else:
                    unmatched_written -= 1
                    per_symbol_unmatched[sym] -= 1
                continue

            line_id = _auto_line_id(sym, tf, kind, t1, t2, oid)
            if line_id in existing_ids:
                skipped += 1
                skip_reasons[f"{kind}/dedup"] += 1
                if locked:
                    valid_written -= 1
                    per_symbol_valid[sym] -= 1
                else:
                    unmatched_written -= 1
                    per_symbol_unmatched[sym] -= 1
                continue

            drift_tag = f" drift={drift:+d}b" if drift != 0 else ""
            label = (
                f"Auto \u00b7 {kind} \u00b7 {status} \u00b7 {label_tag}{drift_tag}"
            )
            notes = (
                f"order_id={oid} status={status} "
                f"entry={(o.get('limit_price') or 0):.6g} "
                f"stop={(o.get('stop_price') or 0):.6g} "
                f"tp={(o.get('tp_price') or 0):.6g} "
                f"slope={slope:.6g} intercept={intercept:.6g} "
                f"bar_count_order={bar_count} bar_count_fetched={len(df)} "
                f"fit_dev_atr=({d1:.3f},{d2:.3f}) "
                f"{viol_note} "
                f"source=retrofit_v3"
            )

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
                "locked": locked,
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

    _atomic_save(MANUAL, lines)

    # ---- Report ----
    print(f"\n[retrofit_v3] === SUMMARY ===", flush=True)
    print(f"  prior auto_triggered removed: {n_auto_before}", flush=True)
    print(f"  candidates processed:         {processed}", flush=True)
    print(f"  valid lines written:          {valid_written}", flush=True)
    print(f"  unmatched (display_only):     {unmatched_written}", flush=True)
    print(f"  skipped:                      {skipped}", flush=True)
    print(f"  errored (fetch/etc):          {errored}", flush=True)
    print(f"  manual_trendlines total:      {len(lines)}", flush=True)

    if per_symbol_valid:
        print(f"\n  per-symbol VALID (top 15):", flush=True)
        for sym, n in per_symbol_valid.most_common(15):
            if n > 0:
                print(f"    {sym}: {n}", flush=True)
    if per_symbol_unmatched:
        print(f"\n  per-symbol UNMATCHED (top 15):", flush=True)
        for sym, n in per_symbol_unmatched.most_common(15):
            if n > 0:
                print(f"    {sym}: {n}", flush=True)
    if skip_reasons:
        print(f"\n  skip reasons:", flush=True)
        for reason, n in skip_reasons.most_common():
            print(f"    {reason}: {n}", flush=True)
    return 0


async def verify() -> int:
    """Re-read `manual_trendlines.json` and assert every auto_triggered line
    we wrote has `violation_count_in_middle == 0` (for locked=True) OR is
    labelled "(unmatched)" / locked=False (for the display-only fallback).

    Prints a per-line table for eyeball verification.
    """
    import re
    from ts.research.pivots import compute_atr  # noqa: F401

    lines = _load_json(MANUAL)
    orders = _load_json(ACTIVE)
    by_oid = {str(o.get("exchange_order_id")): o for o in orders}

    v3 = [l for l in lines if l.get("source") == "auto_triggered"]
    print(f"[verify] checking {len(v3)} auto_triggered lines", flush=True)

    violations_table: list[tuple[str, str, str, int, int, float, int, bool]] = []
    bad = 0

    for line in v3:
        sym = line["symbol"]
        tf = line["timeframe"]
        kind = line["side"]
        locked = bool(line.get("locked", False))
        label = line.get("label", "")
        notes = line.get("notes", "")
        m_i = re.search(r"i1=(\d+)\s+i2=(\d+)", label)
        if not m_i:
            # unmatched fallback: label doesn't include i1/i2 — parsed from notes instead
            m_i = re.search(r"i1=(\d+)\s+i2=(\d+)", notes)
        if not m_i:
            continue
        i1, i2 = int(m_i.group(1)), int(m_i.group(2))

        m_oid = re.search(r"order_id=(\d+)", notes)
        if not m_oid:
            continue
        oid = m_oid.group(1)
        order = by_oid.get(oid)
        if order is None:
            continue
        bc = int(order["bar_count"])
        created_ts = float(order["created_ts"])
        slope = float(order["slope"])
        intercept = float(order["intercept"])

        df = await _fetch_window(sym, tf, created_ts, bc)
        if df is None or len(df) < 20:
            continue
        atr = compute_atr(df, ATR_PERIOD).values
        ok, viols = _validate_line(df, atr, slope, intercept, i1, i2, kind)
        max_v = max((v for _b, v in viols), default=0.0)
        n_v = len(viols)
        violations_table.append((sym, tf, kind, i1, i2, max_v, n_v, locked))
        if locked and n_v > 0:
            bad += 1
            print(
                f"  FAIL: {sym} {tf} {kind} locked=True but n_violations={n_v} max_v={max_v:.3f}",
                flush=True,
            )

    # Print top-10 by max_v
    violations_table.sort(key=lambda x: -x[5])
    print("\n[verify] top-10 by max_violation_atr:", flush=True)
    for row in violations_table[:10]:
        sym, tf, kind, i1, i2, max_v, n_v, locked = row
        print(
            f"  {sym:12s} {tf:4s} {kind:11s} i1={i1:4d} i2={i2:4d} "
            f"max_v={max_v:6.3f} n_v={n_v:3d} locked={locked}",
            flush=True,
        )

    print(f"\n[verify] total checked: {len(violations_table)}  failing locked lines: {bad}", flush=True)
    if bad > 0:
        print("[verify] FAIL — validator bug: locked lines with body violations above threshold", flush=True)
        return 1
    print("[verify] PASS — all locked lines have violation_count_in_middle == 0", flush=True)
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Re-read written lines and assert validity")
    args = parser.parse_args()
    if args.verify:
        return asyncio.run(verify())
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
