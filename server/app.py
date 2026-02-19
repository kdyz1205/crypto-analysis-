from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .data_service import load_symbols, load_okx_swap_symbols, get_ohlcv, get_ohlcv_with_df, API_ONLY
from .pattern_service import get_patterns, get_patterns_from_df, DEFAULT_PARAMS
from .backtest_service import run_backtest, load_df_from_csv, BacktestParams, optimize_backtest
from .ma_ribbon_service import get_current_ribbon, run_ribbon_backtest, RibbonBacktestConfig
from .pattern_features import (
    run_trendline_backtest,
    extract_features,
    similarity,
    is_same_class,
    current_vs_history,
    TrendLineFeatures,
    PatternBacktestConfig,
    user_line_to_features,
)
from .pattern_features import _signal_to_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def _load_df_for_analysis(symbol: str, interval: str, end_time=None, days: int = 365):
    """Load OHLCV as polars DataFrame. When API_ONLY always uses API; otherwise CSV then API fallback. Returns (df, ohlcv)."""
    if API_ONLY:
        return await get_ohlcv_with_df(symbol, interval, end_time, days)
    df = load_df_from_csv(symbol, interval)
    if df is not None and not df.is_empty():
        return df, None
    return await get_ohlcv_with_df(symbol, interval, end_time, days)


app = FastAPI(title="Crypto TA")

# Add CORS middleware to allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify server is running."""
    return {"status": "ok", "message": "Server is running"}


@app.get("/")
async def index():
    return FileResponse(str(PROJECT_ROOT / "frontend" / "index.html"))


def _symbols_from_data_folder() -> list[str]:
    """Collect unique symbols from data/*.csv (standard and OKX-style names)."""
    from .data_service import DATA_DIR, _scan_okx_csv_files
    symbols = set()
    if not DATA_DIR.exists():
        return []
    for p in DATA_DIR.iterdir():
        if p.suffix.lower() != ".csv":
            continue
        name = p.stem.lower()
        if name.startswith("okx_"):
            okx = _scan_okx_csv_files()
            for (sym, _), _ in okx.items():
                symbols.add(sym.upper())
        else:
            # standard: btcusdt_1h -> btcusdt
            parts = name.split("_")
            if len(parts) >= 2 and parts[1] in ("1m", "5m", "15m", "1h", "2h", "4h", "1d"):
                symbols.add(parts[0].upper())
    return sorted(symbols)


@app.get("/api/symbols")
async def get_symbols():
    """Return all available ticker symbols: API + symbols from data/*.csv."""
    from .data_service import EXCHANGE

    from_data = _symbols_from_data_folder()

    if EXCHANGE.lower() == "okx":
        try:
            swap_symbols = await load_okx_swap_symbols()
            if swap_symbols and len(swap_symbols) > 0:
                api_symbols = sorted(swap_symbols.keys())
            else:
                api_symbols = []
        except Exception as e:
            print(f"Warning: Failed to load OKX symbols: {e}")
            api_symbols = []
        # Merge API + local CSV symbols so OKX_*.csv coins appear without API
        combined = sorted(set(api_symbols) | set(s.upper() for s in from_data))
        return combined if combined else (from_data or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT'])
    else:
        try:
            file_symbols = load_symbols()
            combined = sorted(set(file_symbols or []) | set(s.upper() for s in from_data))
            return combined if combined else (from_data or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT'])
        except Exception as e:
            print(f"Warning: Failed to load symbols: {e}")
            return from_data or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT']


@app.get("/api/symbol-info")
async def get_symbol_info(symbol: str):
    """Return metadata for a specific symbol (e.g., price precision)."""
    from .data_service import EXCHANGE, load_okx_swap_symbols
    
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    
    if EXCHANGE.lower() == "okx":
        swap_symbols = await load_okx_swap_symbols()
        if symbol in swap_symbols:
            info = swap_symbols[symbol]
            # OKX-style: decimal places from tickSz ("0.1"->1, "0.01"->2, "0.0001"->4)
            tick_sz = info["tickSz"]
            if "." in str(tick_sz):
                precision = len(str(tick_sz).split(".")[1])
            else:
                precision = 0
            return {
                "symbol": symbol,
                "instId": info["instId"],
                "pricePrecision": precision,
                "tickSz": tick_sz,
            }
    
    return {"symbol": symbol, "pricePrecision": None}


@app.get("/api/chart")
async def api_chart(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(365, description="Days of data; ignored when data is from local CSV (full history returned)"),
):
    """Return OHLCV + support/resistance lines in one response (same data = lines align with candles)."""
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    empty_patterns = {
        "supportLines": [],
        "resistanceLines": [],
        "consolidationZones": [],
        "trendSegments": [],
        "currentTrend": 0,
        "trendLabel": "SIDEWAYS",
        "trendSlope": 0.0,
    }

    try:
        df, ohlcv = await get_ohlcv_with_df(symbol, interval, end_time, days)
        patterns = get_patterns_from_df(df, symbol, interval, end_time)
        return {**ohlcv, **patterns}
    except ValueError as e:
        print(f"ValueError in /api/chart for {symbol} {interval}: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        print(f"Exception in /api/chart: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.get("/api/ohlcv")
async def api_ohlcv(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(30, description="Days of data to fetch"),
):
    """Return OHLCV data as JSON. Auto-downloads from Binance if missing."""
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        result = await get_ohlcv(symbol, interval, end_time, days)
        return result
    except ValueError as e:
        print(f"ValueError in /api/ohlcv for {symbol} {interval}: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        error_detail = f"Error fetching data for {symbol} {interval}: {e}\n{traceback.format_exc()}"
        print(f"Exception in /api/ohlcv: {error_detail}")
        raise HTTPException(500, f"Error fetching data: {str(e)}")


@app.get("/api/backtest")
async def api_backtest(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("4h", description="4h recommended for MFI/MA strategy"),
    days: int = Query(365, description="Days of data"),
    mfi_period: int | None = Query(None),
    ma_fast: int | None = Query(None),
    ema_span: int | None = Query(None),
    ma_slow: int | None = Query(None),
    atr_period: int | None = Query(None),
    atr_sl_mult: float | None = Query(None),
    bb_period: int | None = Query(None),
    bb_std: float | None = Query(None),
):
    """Run MFI/MA backtest. Optional query params override strategy params (see STRATEGY_RULES.md)."""
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")

    try:
        df, _ = await _load_df_for_analysis(symbol, interval, None, days)
    except ValueError as e:
        raise HTTPException(400, f"No data for {symbol} {interval}. Run: python tools/download_full_okx.py {symbol} {interval}\n{e}")
    if df is None or df.is_empty():
        raise HTTPException(400, f"No data for {symbol} {interval}. Run: python tools/download_full_okx.py {symbol} {interval}")

    params = None
    if any(x is not None for x in (mfi_period, ma_fast, ema_span, ma_slow, atr_period, atr_sl_mult, bb_period, bb_std)):
        params = BacktestParams.from_dict({k: v for k, v in {
            "mfi_period": mfi_period, "ma_fast": ma_fast, "ema_span": ema_span, "ma_slow": ma_slow,
            "atr_period": atr_period, "atr_sl_mult": atr_sl_mult, "bb_period": bb_period, "bb_std": bb_std,
        }.items() if v is not None})

    try:
        result = run_backtest(df, params)
        result["symbol"] = symbol
        result["interval"] = interval
        return result
    except Exception as e:
        import traceback
        print(f"Backtest error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.post("/api/backtest/optimize")
async def api_backtest_optimize(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    objective: str = Query("total_pnl", description="total_pnl | sharpe | win_rate"),
    maxiter: int = Query(80, description="Max optimization steps"),
    method: str = Query("L-BFGS-B", description="L-BFGS-B or Nelder-Mead"),
):
    """Optimize strategy parameters (gradient-free) to maximize total_pnl, sharpe, or win_rate."""
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")

    try:
        df, _ = await _load_df_for_analysis(symbol, interval, None, days)
    except ValueError as e:
        raise HTTPException(400, f"No data for {symbol} {interval}\n{e}")
    if df is None or df.is_empty():
        raise HTTPException(400, f"No data for {symbol} {interval}")

    try:
        out = optimize_backtest(df, objective=objective, maxiter=maxiter, method=method)
        out["symbol"] = symbol
        out["interval"] = interval
        return out
    except Exception as e:
        import traceback
        print(f"Optimize error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.get("/api/patterns")
async def api_patterns(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(30, description="Days of data to fetch"),
    mode: str = Query("full", description="full | recognizing (patterns only) | assist (trendlines only, extended)"),
):
    """Run SR pattern detection. mode=recognizing returns only patterns; mode=assist returns only trendlines (extended 3-4 bars)."""
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")
    if mode not in ("full", "recognizing", "assist"):
        mode = "full"

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    from .pattern_service import _empty_pattern_response
    empty_response = _empty_pattern_response(mode)

    try:
        if API_ONLY:
            use_days = max(days, 365)
            df, _ = await get_ohlcv_with_df(symbol, interval, end_time, use_days)
            return get_patterns_from_df(df, symbol, interval, end_time, mode=mode)
        return get_patterns(symbol, interval, end_time, days, mode=mode)
    except ValueError as e:
        # No data / no CSV: return 200 with empty arrays so frontend still renders chart
        print(f"Pattern API (no data): {e}")
        return empty_response
    except Exception as e:
        import traceback
        error_detail = f"Pattern detection error: {e}\n{traceback.format_exc()}"
        print(error_detail)  # Log to server console for debugging
        # Return 200 with empty arrays so frontend doesn't break; lines just won't show
        return empty_response


# ── Pattern → Feature → Backtest (可计算、可回测、无 look-ahead) ──

@app.get("/api/pattern-stats/backtest")
async def api_pattern_stats_backtest(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="5m, 15m, 1h, 4h, 1d"),
    days: int = Query(365, description="Days of data"),
):
    """
    Backtest trendlines with frozen-at-t: at each bar t, structure is built only from [0, t],
    then we look forward N bars (timeframe-bound). Success = price moves in line direction by ≥ k×ATR.
    Returns success_rate_pct and breakdown by line_type / direction.
    """
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
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
        result = run_trendline_backtest(df, interval, get_patterns_at_t)
        result["symbol"] = symbol
        return result
    except Exception as e:
        import traceback
        print(f"Pattern stats backtest error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.get("/api/pattern-stats/features")
async def api_pattern_stats_features(
    symbol: str = Query(...),
    interval: str = Query("1h"),
    end_time: str | None = Query(None, description="Replay end time; if omitted, use latest"),
):
    """
    Get current trendlines as feature vectors (slope, n_points, volatility_norm, direction, etc.)
    so they can be compared with history via similarity().
    """
    import sys
    import polars as pl
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sr_patterns import detect_patterns, calculate_atr

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    try:
        df, _ = await _load_df_for_analysis(symbol, interval, end_time, days=90)
    except Exception:
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


@app.get("/api/pattern-stats/current-vs-history")
async def api_pattern_stats_current_vs_history(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h"),
    days: int = Query(365),
    epsilon: float = Query(0.35, description="Similarity threshold for same-class"),
):
    """
    当前形态 vs 历史同类成功率：取当前图表上的趋势线特征，在历史回测信号中找「同类」（相似度 < ε），
    汇总同类样本的成功率。供前端展示「当前形态 vs 历史同类成功率」。
    """
    import sys
    import polars as pl
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
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

    # Current structure: trendlines at end of data
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

    backtest_result = run_trendline_backtest(df, interval, get_patterns_at_t)
    historical_signals = backtest_result.get("signals") or []

    vs = current_vs_history(current_features, historical_signals, interval, epsilon=epsilon)
    vs["symbol"] = symbol
    vs["interval"] = interval
    vs["epsilon"] = epsilon
    vs["backtest_total_signals"] = backtest_result.get("total_signals", 0)
    vs["backtest_overall_success_rate_pct"] = backtest_result.get("success_rate_pct")
    return vs


def _to_unix_ts(dt) -> int:
    """Convert datetime to unix seconds for frontend."""
    if hasattr(dt, "timestamp"):
        return int(dt.timestamp())
    if hasattr(dt, "to_pydatetime"):
        return int(dt.to_pydatetime().timestamp())
    return int(dt)


@app.get("/api/pattern-stats/line-similar")
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
    """
    Compare a user-drawn line (P1=(x1,y1), P2=(x2,y2)) against historical trendline backtest.
    Returns similar historical lines sorted by similarity (Draw mode: show/hide list).
    """
    import sys
    import polars as pl
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from sr_patterns import detect_patterns, calculate_atr

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    valid = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid:
        raise HTTPException(400, f"Interval must be one of {valid}")

    try:
        df, _ = await _load_df_for_analysis(symbol, interval, None, days)
    except Exception:
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

    backtest_result = run_trendline_backtest(df, interval, get_patterns_at_t)
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
        if xx1 < 0 or xx2 >= len(bar_times):
            continue
        t1 = _to_unix_ts(bar_times[xx1])
        t2 = _to_unix_ts(bar_times[xx2])
        out_lines.append({
            "x1": xx1,
            "y1": float(s["y1"]),
            "x2": xx2,
            "y2": float(s["y2"]),
            "t1": t1,
            "v1": float(s["y1"]),
            "t2": t2,
            "v2": float(s["y2"]),
            "success": s.get("success", False),
            "similarity": round(sim, 4),
            "return_pct": s.get("return_pct"),
        })
    return {"count": len(out_lines), "lines": out_lines}


# ── Multi-timeframe MA Ribbon (for Open Claw / screener) ──

@app.get("/api/ma-ribbon")
async def api_ma_ribbon(symbol: str = Query(..., description="e.g. BTCUSDT or BTC")):
    """
    Current multi-timeframe MA ribbon state: 15m, 1h, 4h, 1d.
    Ribbon = Price > MA9 > MA21 > MA55 (golden cross). Weights: 1d=4, 4h=3, 1h=2, 15m=1.
    Score 0–10; tier = none | low | medium | high | very_high | all_aligned.
    Use this from Open Claw or screeners to get alignment without opening the chart.
    """
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    try:
        out = await get_current_ribbon(symbol, get_ohlcv_with_df)
        return out
    except Exception as e:
        import traceback
        print(f"MA ribbon error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.get("/api/ma-ribbon/backtest")
async def api_ma_ribbon_backtest(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    anchor_tf: str = Query("1d", description="Anchor timeframe for bars"),
    forward_bars: int = Query(5, description="Bars to look forward for success/fail"),
    success_pct: float = Query(2.0, description="Return >= this % counts as success"),
    fail_pct: float = Query(-2.0, description="Return <= this % counts as fail"),
    min_score: int = Query(3, description="Only count signal when score >= this"),
):
    """
    Backtest multi-timeframe ribbon: success rate by score band.
    Confirms that higher alignment (daily+4h+1h+15m) has higher success rate.
    """
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    config = RibbonBacktestConfig(
        anchor_tf=anchor_tf,
        forward_bars=forward_bars,
        success_pct=success_pct,
        fail_pct=fail_pct,
        min_score_to_signal=min_score,
    )
    # Load all TFs from API when API_ONLY, else CSV then API fallback
    async def load_tf(sym: str, tf: str):
        df, _ = await _load_df_for_analysis(sym, tf, None, days=365 * 2)
        return df

    try:
        dfs = {}
        for tf in ["1d", "4h", "1h", "15m"]:
            dfs[tf] = await load_tf(symbol, tf)
        load_fn = lambda sym, tf: dfs.get(tf)
        result = run_ribbon_backtest(symbol, load_fn, config)
        return result
    except Exception as e:
        import traceback
        print(f"MA ribbon backtest error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))
