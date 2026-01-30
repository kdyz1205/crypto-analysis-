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

from sr_patterns import detect_patterns, SRParams, PatternResult

from .data_service import _find_csv, _load_csv, resample_ohlcv, RESAMPLE_MAP

# Reuse the tuned params from sr_patterns (matches PARAMS global)
DEFAULT_PARAMS = SRParams(
    window_left=1,
    window_right=5,
    window_short=2,
    ema_span=50,
    trend_threshold=0.5,
    use_fast_trend=True,
    atr_period=14,
    atr_multiplier=0.5,
    line_break_factor=0.3,
    sr_eps_pct=0.01,
    lookback=None,
    min_touches=2,
    zone_merge_factor=1.5,
    slope_tolerance=0.0001,
    max_lines=15,
)


def _ts(dt) -> int:
    """Convert polars datetime to unix timestamp."""
    return int(dt.timestamp())


def _line_to_json(line, times, n: int) -> dict:
    """Convert a TrendLine dataclass to a JSON-serialisable dict.

    The frontend needs {time, value} pairs (unix seconds) for LineSeries.
    x2 may be projected beyond the data, so we clamp to the last bar and
    recalculate y2 from the slope.
    """
    x1 = int(line.x1)
    x2 = int(line.x2)
    y1 = float(line.y1)
    y2 = float(line.y2)

    # Clamp x2 to last bar if projected beyond data
    if x2 >= n:
        x2 = n - 1
        y2 = y1 + line.slope * (x2 - x1)

    t1 = _ts(times[x1])
    t2 = _ts(times[x2])
    return {
        "x1": t1,
        "y1": y1,
        "x2": t2,
        "y2": float(y2),
        "slope": float(line.slope),
        "touches": int(line.touches),
        "strength": float(line.strength),
        "type": line.line_type,
    }


def get_patterns(
    symbol: str,
    interval: str,
    end_time: str | None = None,
    days: int = 30,
) -> dict:
    """Run pattern detection and return JSON-ready results.

    Parameters mirror the /api/ohlcv endpoint for consistency.
    """
    # Mirror data_service.get_ohlcv data loading exactly so patterns match the chart.
    base_interval, resample_to = RESAMPLE_MAP.get(interval, (interval, None))

    csv_path = _find_csv(symbol, base_interval)

    if csv_path is None:
        # Same fallback order as data_service
        for fallback in ["5m", "1h"]:
            alt = _find_csv(symbol, fallback)
            if alt is not None:
                csv_path = alt
                resample_to = interval if interval != fallback else None
                break

    if csv_path is None:
        raise ValueError(f"No data found for {symbol} {interval}. Load the chart first.")

    df = _load_csv(csv_path)

    if resample_to:
        df = resample_ohlcv(df, resample_to)

    # Determine replay_idx if end_time supplied
    replay_idx = None
    if end_time:
        end_dt = pl.Series([end_time]).str.to_datetime("%Y-%m-%dT%H:%M")[0]
        mask = df["open_time"] <= end_dt
        if mask.sum() == 0:
            raise ValueError("end_time is before all data")
        replay_idx = int(mask.sum()) - 1

    # Adapt EMA span if data is too short for the default
    params = DEFAULT_PARAMS
    n_bars = len(df)
    if n_bars < params.ema_span:
        from dataclasses import replace
        params = replace(params, ema_span=max(10, n_bars // 2))

    # Run detection
    result: PatternResult = detect_patterns(df, params=params, replay_idx=replay_idx)

    # Gather timestamps from the analysis df for index→time mapping
    times = result.df["open_time"].to_list()
    n = len(times)

    # --- Build JSON payload ---

    # Support / resistance trendlines
    support_lines = []
    for line in result.support_lines:
        if int(line.x1) < n:
            support_lines.append(_line_to_json(line, times, n))

    resistance_lines = []
    for line in result.resistance_lines:
        if int(line.x1) < n:
            resistance_lines.append(_line_to_json(line, times, n))

    # Consolidation zones
    consol_zones = []
    for cz in result.consolidation_zones:
        if int(cz.start_idx) < n and int(cz.end_idx) < n:
            consol_zones.append({
                "startTime": _ts(times[int(cz.start_idx)]),
                "endTime": _ts(times[int(cz.end_idx)]),
                "priceLow": float(cz.price_low),
                "priceHigh": float(cz.price_high),
            })

    # Trend segments
    trend_segments = []
    for seg in result.trend_segments:
        if int(seg.start_idx) < n and int(seg.end_idx) < n:
            trend_segments.append({
                "startTime": _ts(times[int(seg.start_idx)]),
                "endTime": _ts(times[int(seg.end_idx)]),
                "trend": int(seg.trend),
            })

    # Current trend
    trend_label = {1: "UPTREND", -1: "DOWNTREND", 0: "SIDEWAYS"}.get(
        result.current_trend, "SIDEWAYS"
    )

    return {
        "supportLines": support_lines,
        "resistanceLines": resistance_lines,
        "consolidationZones": consol_zones,
        "trendSegments": trend_segments,
        "currentTrend": result.current_trend,
        "trendLabel": trend_label,
        "trendSlope": round(result.trend_slope, 3),
    }
