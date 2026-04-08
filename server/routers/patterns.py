"""
Pattern routes: S/R detection, trendlines, pattern features, similarity.
Serves both Market Workbench (live detection) and Research Lab (historical analysis).
"""

import asyncio
import logging
import sys

from fastapi import APIRouter, Query, HTTPException

from ..core.config import PROJECT_ROOT
from ..data_service import get_ohlcv_with_df, API_ONLY
from ..pattern_service import get_patterns, get_patterns_from_df, DEFAULT_PARAMS
from ..backtest_service import load_df_from_csv
from ..pattern_features import (
    run_trendline_backtest, extract_features, similarity,
    is_same_class, current_vs_history,
    user_line_to_features, _signal_to_features,
)

router = APIRouter(prefix="/api", tags=["patterns"])


# ── Shared helper ────────────────────────────────────────────────────────

async def _load_df_for_analysis(symbol: str, interval: str, end_time=None, days: int = 365):
    """Load OHLCV as polars DataFrame. CSV first, API fallback."""
    if API_ONLY:
        return await get_ohlcv_with_df(symbol, interval, end_time, days)
    df = load_df_from_csv(symbol, interval)
    if df is not None and not df.is_empty():
        return df, None
    return await get_ohlcv_with_df(symbol, interval, end_time, days)


def _to_unix_ts(dt) -> int:
    """Convert datetime to unix seconds for frontend."""
    if hasattr(dt, "timestamp"):
        return int(dt.timestamp())
    if hasattr(dt, "to_pydatetime"):
        return int(dt.to_pydatetime().timestamp())
    return int(dt)


def _ensure_sr_patterns():
    """Ensure sr_patterns module is importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ── Routes ───────────────────────────────────────────────────────────────

@router.get("/patterns")
async def api_patterns(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(30, description="Days of data to fetch"),
    mode: str = Query("full", description="full | recognizing (patterns only) | assist (trendlines only, extended)"),
):
    """Run SR pattern detection."""
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")
    if mode not in ("full", "recognizing", "assist"):
        mode = "full"

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    from ..pattern_service import _empty_pattern_response
    empty_response = _empty_pattern_response(mode)

    try:
        if API_ONLY:
            use_days = max(days, 365)
            df, _ = await get_ohlcv_with_df(symbol, interval, end_time, use_days)
            return get_patterns_from_df(df, symbol, interval, end_time, mode=mode)
        return get_patterns(symbol, interval, end_time, days, mode=mode)
    except ValueError as e:
        print(f"Pattern API (no data): {e}")
        return empty_response
    except Exception as e:
        import traceback
        print(f"Pattern detection error: {e}\n{traceback.format_exc()}")
        return empty_response


@router.get("/pattern-stats/backtest")
async def api_pattern_stats_backtest(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="5m, 15m, 1h, 4h, 1d"),
    days: int = Query(365, description="Days of data"),
):
    """
    Backtest trendlines with frozen-at-t: at each bar t, structure is built only from [0, t],
    then we look forward N bars. Success = price moves in line direction by >= k*ATR.
    """
    _ensure_sr_patterns()
    from sr_patterns import detect_patterns

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    valid = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid:
        raise HTTPException(400, f"Interval must be one of {valid}")

    try:
        df, _ = await _load_df_for_analysis(symbol, interval, None, days)
    except ValueError as e:
        raise HTTPException(400, f"No data for {symbol} {interval}: {e}")
    if df is None or df.is_empty():
        raise HTTPException(400, f"No data for {symbol} {interval}")

    params = DEFAULT_PARAMS
    n_bars = len(df)
    if n_bars < getattr(params, "ema_span", 50):
        from dataclasses import replace
        params = replace(params, ema_span=max(10, n_bars // 2))

    def get_patterns_at_t(dff, replay_idx):
        try:
            return detect_patterns(dff, params=params, replay_idx=replay_idx)
        except Exception:
            return None

    try:
        result = await asyncio.to_thread(run_trendline_backtest, df, interval, get_patterns_at_t)
        result["symbol"] = symbol
        return result
    except Exception as e:
        import traceback
        print(f"Pattern stats backtest error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@router.get("/pattern-stats/features")
async def api_pattern_stats_features(
    symbol: str = Query(...),
    interval: str = Query("1h"),
    end_time: str | None = Query(None, description="Replay end time; if omitted, use latest"),
):
    """Get current trendlines as feature vectors for similarity comparison."""
    _ensure_sr_patterns()
    import polars as pl
    from sr_patterns import detect_patterns, calculate_atr

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    try:
        df, _ = await _load_df_for_analysis(symbol, interval, end_time, days=90)
    except Exception as e:
        logging.warning(f"Feature load failed for {symbol} {interval}: {e}")
        return {"symbol": symbol, "interval": interval, "features": []}
    if df is None or df.is_empty():
        return {"symbol": symbol, "interval": interval, "features": []}
    df = calculate_atr(df, 14)
    bar_times = df["open_time"].to_list()
    atr = df["atr"].to_numpy()
    replay_idx = None
    if end_time:
        end_dt = pl.Series([end_time]).str.to_datetime("%Y-%m-%dT%H:%M")[0]
        mask = df["open_time"] <= end_dt
        if mask.sum() > 0:
            replay_idx = int(mask.sum()) - 1
    result = detect_patterns(df, params=DEFAULT_PARAMS, replay_idx=replay_idx)
    features = []
    for line in (result.support_lines or []) + (result.resistance_lines or []):
        feats = extract_features(line, interval, bar_times, atr)
        if feats:
            features.append(feats.to_dict())
    return {"symbol": symbol, "interval": interval, "features": features}


@router.get("/pattern-stats/current-vs-history")
async def api_pattern_stats_current_vs_history(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h"),
    days: int = Query(365),
    epsilon: float = Query(0.35, description="Similarity threshold for same-class"),
):
    """Current pattern vs historical success rate."""
    _ensure_sr_patterns()
    from sr_patterns import detect_patterns, calculate_atr

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    valid = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid:
        raise HTTPException(400, f"Interval must be one of {valid}")

    try:
        df, _ = await _load_df_for_analysis(symbol, interval, None, days)
    except ValueError as e:
        raise HTTPException(400, f"No data for {symbol} {interval}: {e}")
    if df is None or df.is_empty():
        raise HTTPException(400, f"No data for {symbol} {interval}")

    params = DEFAULT_PARAMS
    n_bars = len(df)
    if n_bars < getattr(params, "ema_span", 50):
        from dataclasses import replace
        params = replace(params, ema_span=max(10, n_bars // 2))

    df = calculate_atr(df, 14)
    bar_times = df["open_time"].to_list()
    atr = df["atr"].to_numpy()

    result_end = detect_patterns(df, params=params, replay_idx=None)
    current_features = []
    for line in (result_end.support_lines or []) + (result_end.resistance_lines or []):
        feats = extract_features(line, interval, bar_times, atr)
        if feats:
            current_features.append(feats)

    def get_patterns_at_t(dff, replay_idx):
        try:
            return detect_patterns(dff, params=params, replay_idx=replay_idx)
        except Exception:
            return None

    backtest_result = await asyncio.to_thread(run_trendline_backtest, df, interval, get_patterns_at_t)
    historical_signals = backtest_result.get("signals") or []

    vs = current_vs_history(current_features, historical_signals, interval, epsilon=epsilon)
    vs["symbol"] = symbol
    vs["interval"] = interval
    vs["epsilon"] = epsilon
    vs["backtest_total_signals"] = backtest_result.get("total_signals", 0)
    vs["backtest_overall_success_rate_pct"] = backtest_result.get("success_rate_pct")
    return vs


@router.get("/pattern-stats/line-similar")
async def api_pattern_stats_line_similar(
    symbol: str = Query(...),
    interval: str = Query("1h"),
    days: int = Query(365),
    x1: int = Query(..., description="Bar index of first point"),
    y1: float = Query(..., description="Price at first point"),
    x2: int = Query(..., description="Bar index of second point"),
    y2: float = Query(..., description="Price at second point"),
    epsilon: float = Query(0.4, description="Similarity threshold; lower = stricter"),
    max_lines: int = Query(15, description="Max similar lines to return"),
):
    """Compare a user-drawn line against historical trendline backtest."""
    _ensure_sr_patterns()
    from sr_patterns import detect_patterns, calculate_atr

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    valid = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid:
        raise HTTPException(400, f"Interval must be one of {valid}")

    try:
        df, _ = await _load_df_for_analysis(symbol, interval, None, days)
    except Exception as e:
        logging.warning(f"Similar lines data load failed for {symbol} {interval}: {e}")
        return {"count": 0, "lines": []}
    if df is None or df.is_empty():
        return {"count": 0, "lines": []}

    params = DEFAULT_PARAMS
    n_bars = len(df)
    if n_bars < getattr(params, "ema_span", 50):
        from dataclasses import replace
        params = replace(params, ema_span=max(10, n_bars // 2))

    df = calculate_atr(df, 14)
    bar_times = df["open_time"].to_list()
    atr = df["atr"].to_numpy()

    user_feat = user_line_to_features(int(x1), float(y1), int(x2), float(y2), interval, bar_times, atr)
    if user_feat is None:
        return {"count": 0, "lines": []}

    def get_patterns_at_t(dff, replay_idx):
        try:
            return detect_patterns(dff, params=params, replay_idx=replay_idx)
        except Exception:
            return None

    backtest_result = await asyncio.to_thread(run_trendline_backtest, df, interval, get_patterns_at_t)
    historical_signals = backtest_result.get("signals") or []

    similar_with_score = []
    for s in historical_signals:
        if s.get("x1") is None:
            continue
        try:
            s_feat = _signal_to_features(s, interval)
            if not is_same_class(user_feat, s_feat, epsilon):
                continue
            sim = similarity(user_feat, s_feat)
            similar_with_score.append((sim, s))
        except Exception:
            continue

    similar_with_score.sort(key=lambda x: x[0])
    out_lines = []
    for sim, s in similar_with_score[:max_lines]:
        xx1, xx2 = int(s["x1"]), int(s["x2"])
        if xx1 < 0 or xx2 < 0 or xx1 >= len(bar_times) or xx2 >= len(bar_times):
            continue
        t1 = _to_unix_ts(bar_times[xx1])
        t2 = _to_unix_ts(bar_times[xx2])
        out_lines.append({
            "x1": xx1, "y1": float(s["y1"]),
            "x2": xx2, "y2": float(s["y2"]),
            "t1": t1, "v1": float(s["y1"]),
            "t2": t2, "v2": float(s["y2"]),
            "success": s.get("success", False),
            "similarity": round(sim, 4),
            "return_pct": s.get("return_pct"),
        })
    return {"count": len(out_lines), "lines": out_lines}
