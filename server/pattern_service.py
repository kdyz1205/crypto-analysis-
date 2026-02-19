"""
Pattern detection service — wraps sr_patterns.detect_patterns() and returns
JSON-serialisable data for the Lightweight Charts frontend.
"""

import polars as pl
from pathlib import Path
import sys

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sr_patterns import detect_patterns, SRParams, PatternResult, TrianglePattern

from .data_service import _find_csv, _load_csv, resample_ohlcv, RESAMPLE_MAP

# Reuse the tuned params from sr_patterns (matches PARAMS global)
DEFAULT_PARAMS = SRParams(
    window_left=1,
    window_right=5,
    window_short=2,
    prominence_multiplier=0.0,  # Disable by default to avoid filtering out all extrema
    ema_span=50,
    trend_threshold=0.5,
    use_fast_trend=True,
    atr_period=14,
    atr_multiplier=0.5,
    line_break_factor=0.3,
    tolerance_percent=0.005,  # 0.5% default, more lenient to avoid rejecting all lines
    k_atr=0.5,
    fixed_pct=0.005,
    high_volatility_threshold=0.02,
    high_vol_tolerance_pct=0.007,  # 0.7% for high volatility assets
    ceiling_tolerance=0.001,  # Strict non-crossing: resistance High <= line*(1+tol)
    sr_eps_pct=0.01,
    lookback=None,
    min_touches=2,
    zone_merge_factor=1.5,
    slope_tolerance=0.0001,
    max_lines=15,
)


def _ts(dt) -> int:
    """Convert datetime to unix timestamp (seconds). Handles polars and Python datetime."""
    if hasattr(dt, "timestamp"):
        return int(dt.timestamp())
    # polars scalar or other
    import datetime
    if hasattr(dt, "to_pydatetime"):
        return int(dt.to_pydatetime().timestamp())
    return int(dt)


def _line_to_json(line, times, n: int, extend_bars: int = 0) -> dict:
    """Convert a TrendLine to JSON. extend_bars: allow line to extend past last bar (e.g. 4 for Assist)."""
    x1 = int(line.x1)
    x2 = int(line.x2)
    y1 = float(line.y1)
    y2 = float(line.y2)
    max_x = (n - 1) + extend_bars if extend_bars else n - 1

    if x2 > max_x:
        x2 = max_x
        y2 = y1 + line.slope * (x2 - x1)
    elif x2 >= n and extend_bars > 0:
        # Keep extension; y2 already from slope
        y2 = y1 + line.slope * (x2 - x1)

    t1 = _ts(times[min(x1, n - 1)])
    # Time for x2: if x2 >= n, use last bar time + rough delta for display
    if x2 < n:
        t2 = _ts(times[x2])
    else:
        from datetime import datetime, timedelta
        last_ts = _ts(times[n - 1])
        # Approximate: one bar = same interval as last two bars
        if n >= 2:
            delta = _ts(times[n - 1]) - _ts(times[n - 2])
            t2 = last_ts + delta * (x2 - (n - 1))
        else:
            t2 = last_ts
    return {
        "x1": t1,
        "y1": y1,
        "x2": t2,
        "y2": float(y2),
        "slope": float(line.slope),
        "touches": int(line.touches),
        "strength": float(line.strength),
        "type": line.line_type,
        "tolerance": float(line.tolerance) if hasattr(line, 'tolerance') else 0.0,
    }


def _date_str(dt) -> str:
    """Format datetime as YYYY-MM-DD."""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d")
    if hasattr(dt, "to_pydatetime"):
        return dt.to_pydatetime().strftime("%Y-%m-%d")
    import datetime
    if hasattr(dt, "timestamp"):
        return datetime.datetime.fromtimestamp(dt.timestamp(), tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    return str(dt)[:10]


def _triangle_to_structured(tri: TrianglePattern, times: list) -> dict:
    """Structured pattern for Recognizing mode: type, start_date, end_date, high_points, low_points."""
    sup = tri.support_line
    res = tri.resistance_line
    start_idx = min(int(sup.x1), int(res.x1))
    end_idx = max(int(sup.x2), int(res.x2), tri.apex_x)
    n = len(times)
    start_idx = min(start_idx, n - 1)
    end_idx = min(end_idx, n - 1)
    start_date = _date_str(times[start_idx])
    end_date = _date_str(times[end_idx])
    # High points: resistance line (time, price) at start/end
    high_points = [
        {"time": _ts(times[int(res.x1)]), "value": float(res.y1)},
        {"time": _ts(times[min(int(res.x2), n - 1)]), "value": float(res.y2)},
    ]
    low_points = [
        {"time": _ts(times[int(sup.x1)]), "value": float(sup.y1)},
        {"time": _ts(times[min(int(sup.x2), n - 1)]), "value": float(sup.y2)},
    ]
    return {
        "type": "triangle",
        "pattern_type": tri.pattern_type,
        "start_date": start_date,
        "end_date": end_date,
        "high_points": high_points,
        "low_points": low_points,
        "breakout_bias": tri.breakout_bias,
        "completion_pct": round(tri.completion_pct, 1),
    }


def get_patterns_from_df(
    df: pl.DataFrame,
    symbol: str,
    interval: str,
    end_time: str | None = None,
    mode: str = "full",
) -> dict:
    """Run pattern detection on an existing dataframe. mode: full | recognizing | assist."""
    replay_idx = None
    if end_time:
        end_dt = pl.Series([end_time]).str.to_datetime("%Y-%m-%dT%H:%M")[0]
        mask = df["open_time"] <= end_dt
        if mask.sum() == 0:
            return _empty_pattern_response(mode)
        replay_idx = int(mask.sum()) - 1

    params = DEFAULT_PARAMS
    n_bars = len(df)
    if n_bars < params.ema_span:
        from dataclasses import replace
        params = replace(params, ema_span=max(10, n_bars // 2))

    try:
        result: PatternResult = detect_patterns(df, params=params, replay_idx=replay_idx)
    except Exception:
        return _empty_pattern_response(mode)
    return _pattern_result_to_json(result, mode=mode)


def _empty_pattern_response(mode: str = "full") -> dict:
    base = {
        "supportLines": [],
        "resistanceLines": [],
        "consolidationZones": [],
        "trendSegments": [],
        "patterns": [],
        "currentTrend": 0,
        "trendLabel": "SIDEWAYS",
        "trendSlope": 0.0,
    }
    if mode == "recognizing":
        return {k: base[k] for k in ("patterns", "currentTrend", "trendLabel", "trendSlope")}
    if mode == "assist":
        return {k: base[k] for k in ("supportLines", "resistanceLines", "consolidationZones", "currentTrend", "trendLabel", "trendSlope")}
    return base


def _pattern_result_to_json(result: PatternResult, mode: str = "full") -> dict:
    """Build JSON by mode: recognizing = patterns only; assist = trendlines only (extended); full = all."""
    times = result.df["open_time"].to_list()
    n = len(times)
    extend_bars = 4 if mode == "assist" else 0

    # Structured patterns: triangles + channels (consolidation zones)
    patterns = []
    for tri in result.triangles:
        if int(tri.support_line.x1) < n and int(tri.resistance_line.x1) < n:
            patterns.append(_triangle_to_structured(tri, times))
    for cz in result.consolidation_zones:
        if int(cz.start_idx) < n and int(cz.end_idx) < n:
            t_start = _ts(times[int(cz.start_idx)])
            t_end = _ts(times[int(cz.end_idx)])
            patterns.append({
                "type": "channel",
                "pattern_type": "consolidation",
                "start_date": _date_str(times[int(cz.start_idx)]),
                "end_date": _date_str(times[int(cz.end_idx)]),
                "high_points": [{"time": t_start, "value": float(cz.price_high)}, {"time": t_end, "value": float(cz.price_high)}],
                "low_points": [{"time": t_start, "value": float(cz.price_low)}, {"time": t_end, "value": float(cz.price_low)}],
                "breakout_bias": "neutral",
                "completion_pct": 0,
            })

    trend_label = {1: "UPTREND", -1: "DOWNTREND", 0: "SIDEWAYS"}.get(
        result.current_trend, "SIDEWAYS"
    )
    meta = {
        "currentTrend": result.current_trend,
        "trendLabel": trend_label,
        "trendSlope": round(result.trend_slope, 3),
    }

    if mode == "recognizing":
        return {"patterns": patterns, **meta}

    support_lines = []
    for line in result.support_lines:
        if int(line.x1) < n:
            support_lines.append(_line_to_json(line, times, n, extend_bars))

    resistance_lines = []
    for line in result.resistance_lines:
        if int(line.x1) < n:
            resistance_lines.append(_line_to_json(line, times, n, extend_bars))

    if mode == "assist":
        consol_zones = []
        for cz in result.consolidation_zones:
            if int(cz.start_idx) < n and int(cz.end_idx) < n:
                consol_zones.append({
                    "startTime": _ts(times[int(cz.start_idx)]),
                    "endTime": _ts(times[int(cz.end_idx)]),
                    "priceLow": float(cz.price_low),
                    "priceHigh": float(cz.price_high),
                })
        return {
            "supportLines": support_lines,
            "resistanceLines": resistance_lines,
            "consolidationZones": consol_zones,
            **meta,
        }

    consol_zones = []
    for cz in result.consolidation_zones:
        if int(cz.start_idx) < n and int(cz.end_idx) < n:
            consol_zones.append({
                "startTime": _ts(times[int(cz.start_idx)]),
                "endTime": _ts(times[int(cz.end_idx)]),
                "priceLow": float(cz.price_low),
                "priceHigh": float(cz.price_high),
            })

    trend_segments = []
    for seg in result.trend_segments:
        if int(seg.start_idx) < n and int(seg.end_idx) < n:
            trend_segments.append({
                "startTime": _ts(times[int(seg.start_idx)]),
                "endTime": _ts(times[int(seg.end_idx)]),
                "trend": int(seg.trend),
            })

    return {
        "supportLines": support_lines,
        "resistanceLines": resistance_lines,
        "consolidationZones": consol_zones,
        "trendSegments": trend_segments,
        "patterns": patterns,
        **meta,
    }


def get_patterns(
    symbol: str,
    interval: str,
    end_time: str | None = None,
    days: int = 30,
    mode: str = "full",
) -> dict:
    """Run pattern detection and return JSON-ready results.

    Parameters mirror the /api/ohlcv endpoint for consistency.
    """
    base_interval, resample_to = RESAMPLE_MAP.get(interval, (interval, None))

    csv_path = _find_csv(symbol, base_interval)

    if csv_path is None:
        for fallback in ["5m", "1h"]:
            alt = _find_csv(symbol, fallback)
            if alt is not None:
                csv_path = alt
                resample_to = interval if interval != fallback else None
                break

    if csv_path is None:
        raise ValueError(f"No data found for {symbol} {interval}. Please load the chart first (this will download/create the data file).")

    df = _load_csv(csv_path)

    if resample_to:
        df = resample_ohlcv(df, resample_to)

    replay_idx = None
    if end_time:
        end_dt = pl.Series([end_time]).str.to_datetime("%Y-%m-%dT%H:%M")[0]
        mask = df["open_time"] <= end_dt
        if mask.sum() == 0:
            raise ValueError("end_time is before all data")
        replay_idx = int(mask.sum()) - 1

    params = DEFAULT_PARAMS
    n_bars = len(df)
    if n_bars < params.ema_span:
        from dataclasses import replace
        params = replace(params, ema_span=max(10, n_bars // 2))

    result: PatternResult = detect_patterns(df, params=params, replay_idx=replay_idx)
    return _pattern_result_to_json(result, mode=mode)
