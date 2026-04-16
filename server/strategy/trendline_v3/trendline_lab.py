"""Trendline Laboratory — enumerate 2-point trendlines and measure forward behavior.

Goal (from user spec): understand what makes a trendline "work" BEFORE writing
strategy code. Output: per-trendline feature/outcome table that can be sliced
and queried.

Trendline = linear extension of two same-type pivots (low-low or high-high).
No 3-point fits in this pass; we want the purest definition possible.

For each line we record:
  Anchor features (known at t = anchor2.idx):
    - anchor_distance_bars, anchor_distance_pct (tight vs loose pivot pair)
    - slope, slope_atr_per_bar (steepness in regime-normalized units)
    - pivot1_strength, pivot2_strength (ATR reaction at each anchor)

  Forward behavior tracked from t = anchor2.idx + 1 until break or end:
    - For each tolerance in TOL_GRID:
        - n_touches
        - n_respects (touch + reaction ≥ 1 ATR away in expected direction)
        - first_break_bar (close breaches line by 0.3 × ATR)
        - survival_bars (= first_break_bar - anchor2.idx, or end)
    - max_bounce_atr (best single respect, measured in ATR)
    - max_bounce_pct (same, in pct)

Tolerance grid is what we sweep to answer user Q1 ("容差该多宽").
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from ts.research.pivots import Pivot, compute_atr


# Tolerance grid: mix of absolute pct and ATR-relative
# Each entry: (name, mode, value)
#   mode = 'pct' → tolerance = price * value
#   mode = 'atr' → tolerance = atr * value
TOL_GRID = [
    ("pct_0.05", "pct", 0.0005),   # 0.05% — very tight (matches user's SL range)
    ("pct_0.10", "pct", 0.0010),
    ("pct_0.20", "pct", 0.0020),
    ("atr_0.3",  "atr", 0.3),
    ("atr_0.5",  "atr", 0.5),
    ("atr_1.0",  "atr", 1.0),
]

# A "break" is when close crosses the line by this much, in ATR. Conservative.
BREAK_ATR = 0.5
# A "respect" requires a bounce of at least this many ATR after touch.
RESPECT_ATR = 1.0
# Max bars we track forward after a line is born.
MAX_TRACK_BARS = 300


@dataclass
class TrendlineRecord:
    kind: str                # 'support' | 'resistance'
    anchor1_idx: int
    anchor2_idx: int
    slope: float
    intercept: float

    # Anchor features
    anchor_distance_bars: int
    anchor_distance_pct: float
    slope_atr_per_bar: float
    pivot1_strength: float
    pivot2_strength: float

    # Forward outcomes (per tolerance)
    touches_by_tol: dict[str, int] = field(default_factory=dict)
    respects_by_tol: dict[str, int] = field(default_factory=dict)

    # Universal outcomes
    first_break_bar: int | None = None
    survival_bars: int = 0
    max_bounce_atr: float = 0.0
    max_bounce_pct: float = 0.0
    n_bars_tracked: int = 0


def _line_price_at(record_slope: float, intercept: float, i: int) -> float:
    return record_slope * i + intercept


def enumerate_trendlines(
    df: pd.DataFrame,
    pivots: list[Pivot],
    atr_period: int = 14,
    max_anchor_gap: int | None = None,
    measure_touches: bool = True,
) -> list[TrendlineRecord]:
    """Enumerate every 2-point trendline (same pivot type) and walk it forward.

    max_anchor_gap: if set, skip pivot pairs whose anchor-distance in bars
    exceeds this. Useful to prevent O(N²) blowup on long series.

    measure_touches: if False, skip the per-trendline forward walk that
    computes touches_by_tol / respects_by_tol / max_bounce. Use this when
    downstream consumers (e.g. passive_events) do their own forward walk and
    don't need the tolerance-sweep metadata. Dramatically faster for big runs.
    """
    atr = compute_atr(df, atr_period).values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    lows_p = [p for p in pivots if p.kind == "low"]
    highs_p = [p for p in pivots if p.kind == "high"]

    records: list[TrendlineRecord] = []

    for bucket, kind in ((lows_p, "support"), (highs_p, "resistance")):
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a1, a2 = bucket[i], bucket[j]
                gap = a2.idx - a1.idx
                if gap < 3:
                    continue
                if max_anchor_gap is not None and gap > max_anchor_gap:
                    continue

                slope = (a2.price - a1.price) / gap
                intercept = a1.price - slope * a1.idx

                # Anchor ATR (normalize slope)
                anchor_atr = atr[a2.idx]
                if not np.isfinite(anchor_atr) or anchor_atr == 0:
                    continue
                slope_atr_per_bar = slope / anchor_atr
                anchor_distance_pct = (a2.price - a1.price) / a1.price

                rec = TrendlineRecord(
                    kind=kind,
                    anchor1_idx=a1.idx,
                    anchor2_idx=a2.idx,
                    slope=slope,
                    intercept=intercept,
                    anchor_distance_bars=gap,
                    anchor_distance_pct=float(anchor_distance_pct),
                    slope_atr_per_bar=float(slope_atr_per_bar),
                    pivot1_strength=a1.strength,
                    pivot2_strength=a2.strength,
                    touches_by_tol={name: 0 for name, _, _ in TOL_GRID},
                    respects_by_tol={name: 0 for name, _, _ in TOL_GRID},
                )

                if not measure_touches:
                    records.append(rec)
                    continue

                # Walk forward from anchor2 + 1
                start = a2.idx + 1
                end = min(n, start + MAX_TRACK_BARS)
                last_touch_tol = {name: -1 for name, _, _ in TOL_GRID}

                for b in range(start, end):
                    cur_atr = atr[b]
                    if not np.isfinite(cur_atr) or cur_atr == 0:
                        continue
                    line_p = slope * b + intercept
                    bar_high = highs[b]
                    bar_low = lows[b]
                    bar_close = closes[b]

                    # BREAK: close beyond line by BREAK_ATR
                    if kind == "support":
                        if line_p - bar_close >= BREAK_ATR * cur_atr:
                            rec.first_break_bar = b
                            break
                    else:  # resistance
                        if bar_close - line_p >= BREAK_ATR * cur_atr:
                            rec.first_break_bar = b
                            break

                    # TOUCH and RESPECT per tolerance
                    for name, mode, val in TOL_GRID:
                        tol = line_p * val if mode == "pct" else cur_atr * val
                        touched = False
                        if kind == "support":
                            touched = bar_low <= line_p + tol and bar_low >= line_p - tol
                            # Touched if bar's low wick is inside the band
                            touched = (bar_low <= line_p + tol) and (bar_high >= line_p - tol)
                        else:
                            touched = (bar_high >= line_p - tol) and (bar_low <= line_p + tol)

                        if touched and (b - last_touch_tol[name]) >= 2:
                            rec.touches_by_tol[name] += 1
                            last_touch_tol[name] = b
                            # Check for respect: within next 10 bars, does price
                            # move ≥ RESPECT_ATR in the expected direction?
                            forward_end = min(n, b + 11)
                            if kind == "support":
                                fwd_high = highs[b:forward_end].max()
                                bounce = (fwd_high - line_p) / cur_atr
                            else:
                                fwd_low = lows[b:forward_end].min()
                                bounce = (line_p - fwd_low) / cur_atr
                            if bounce >= RESPECT_ATR:
                                rec.respects_by_tol[name] += 1

                            # Track biggest bounce (use widest tolerance for this)
                            if name == "atr_1.0":
                                bounce_pct = bounce * cur_atr / line_p
                                if bounce > rec.max_bounce_atr:
                                    rec.max_bounce_atr = float(bounce)
                                    rec.max_bounce_pct = float(bounce_pct)

                rec.n_bars_tracked = min(end, rec.first_break_bar or end) - start
                rec.survival_bars = (rec.first_break_bar - a2.idx) if rec.first_break_bar else rec.n_bars_tracked
                records.append(rec)

    return records


def records_to_dataframe(records: list[TrendlineRecord]) -> pd.DataFrame:
    """Flatten records to DataFrame for analysis."""
    rows = []
    for r in records:
        row = {
            "kind": r.kind,
            "anchor1_idx": r.anchor1_idx,
            "anchor2_idx": r.anchor2_idx,
            "anchor_distance_bars": r.anchor_distance_bars,
            "anchor_distance_pct": r.anchor_distance_pct,
            "slope_atr_per_bar": r.slope_atr_per_bar,
            "pivot1_strength": r.pivot1_strength,
            "pivot2_strength": r.pivot2_strength,
            "first_break_bar": r.first_break_bar if r.first_break_bar else -1,
            "survival_bars": r.survival_bars,
            "n_bars_tracked": r.n_bars_tracked,
            "max_bounce_atr": r.max_bounce_atr,
            "max_bounce_pct": r.max_bounce_pct,
        }
        for name, _, _ in TOL_GRID:
            row[f"touches_{name}"] = r.touches_by_tol[name]
            row[f"respects_{name}"] = r.respects_by_tol[name]
        rows.append(row)
    return pd.DataFrame(rows)
