import asyncio
import time
import json
import logging
from collections import deque

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import csv


# ── Centralized Log Buffer ────────────────────────────────────────────────────
# Captures all print-style logs from agent/trader/healer into a ring buffer
# so the frontend can display them via /api/agent/logs
_LOG_BUFFER: deque[dict] = deque(maxlen=200)


class _AgentLogHandler(logging.Handler):
    """Captures log records into the shared ring buffer."""
    def emit(self, record):
        try:
            _LOG_BUFFER.append({
                "ts": record.created,
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "msg": self.format(record),
            })
        except Exception:
            pass


# Install handler on root logger so all print→logging and direct logging calls are captured
_agent_handler = _AgentLogHandler()
_agent_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_agent_handler)
logging.getLogger().setLevel(logging.INFO)

# Also monkey-patch builtins.print to capture print() calls from agent/trader modules
import builtins
_original_print = builtins.print

def _capturing_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _LOG_BUFFER.append({
        "ts": time.time(),
        "time": time.strftime("%H:%M:%S"),
        "level": "INFO",
        "msg": msg,
    })
    _original_print(*args, **kwargs)

builtins.print = _capturing_print

from .data_service import load_symbols, load_okx_swap_symbols, get_ohlcv, get_ohlcv_with_df, API_ONLY
from .pattern_service import get_patterns, get_patterns_from_df, DEFAULT_PARAMS
from .backtest_service import run_backtest, load_df_from_csv, BacktestParams, optimize_backtest
from .ma_ribbon_service import get_current_ribbon, run_ribbon_backtest, RibbonBacktestConfig
from .agent_brain import AgentBrain
from .ai_chat import AIChatEngine
from .self_healer import SelfHealer
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

# ── Agent singleton ──
_agent: AgentBrain | None = None
_chat: AIChatEngine | None = None
_healer: SelfHealer | None = None


def get_agent() -> AgentBrain:
    global _agent
    if _agent is None:
        _agent = AgentBrain()
    return _agent


def get_chat() -> AIChatEngine:
    global _chat
    if _chat is None:
        _chat = AIChatEngine()
    return _chat


def get_healer() -> SelfHealer:
    global _healer
    if _healer is None:
        _healer = SelfHealer()
    return _healer


@app.on_event("startup")
async def _startup():
    agent = get_agent()
    chat = get_chat()
    healer = get_healer()
    healer.start()  # auto-start self-healer
    print(f"[Agent] Initialized. Mode={agent.trader.state.mode} Gen={agent.trader.state.generation}")
    print(f"[AI Chat] Ready. API key={'set' if chat._anthropic_client else 'NOT set'}")
    print(f"[Healer] Self-healing active. AI={'enabled' if healer._client else 'disabled'}")


@app.on_event("shutdown")
async def _shutdown():
    if _agent is not None:
        _agent.stop()
    if _healer is not None:
        _healer.stop()

# Add CORS middleware to allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (both /static/ for legacy and root-level for Vercel-compatible paths)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify server is running."""
    return {"status": "ok", "message": "Server is running"}


@app.get("/")
async def index():
    return FileResponse(str(PROJECT_ROOT / "frontend" / "index.html"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


def _symbols_from_data_folder() -> list[str]:
    """Collect unique symbols from data/*.csv and data/*.csv.gz (standard and OKX-style names)."""
    from .data_service import DATA_DIR, _scan_okx_csv_files
    symbols = set()
    if not DATA_DIR.exists():
        return []
    for p in DATA_DIR.iterdir():
        is_csv = p.suffix.lower() == ".csv" or p.name.lower().endswith(".csv.gz")
        if not is_csv:
            continue
        name = p.stem.lower()
        if name.endswith('.csv'):
            name = name[:-4]  # strip .csv from .csv.gz (stem gives "foo.csv")
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


def _symbols_from_ticker_info_csv() -> list[str]:
    """Fallback: parse symbols from binance_futures_ticker_info.csv when exchange API is unavailable."""
    path = PROJECT_ROOT / "binance_futures_ticker_info.csv"
    if not path.exists():
        return []
    out = set()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = (row.get("symbol") or row.get("Symbol") or row.get("ticker") or row.get("Ticker") or "").strip().upper()
                if sym and sym.endswith("USDT"):
                    out.add(sym)
    except Exception:
        return []
    return sorted(out)


@app.get("/api/symbols")
async def get_symbols(include_extended: bool = Query(False, description="Include extended fallback universe from ticker info CSV")):
    """Return all available ticker symbols: API + symbols from data/*.csv."""
    from .data_service import EXCHANGE

    from_data = _symbols_from_data_folder()
    from_info_csv = _symbols_from_ticker_info_csv() if include_extended else []

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
        combined = sorted(set(api_symbols) | set(s.upper() for s in from_data) | set(from_info_csv))
        # If regular sources are empty, fallback to extended universe automatically.
        if not combined and not include_extended:
            fallback = _symbols_from_ticker_info_csv()
            if fallback:
                return fallback
        return combined if combined else (from_data or from_info_csv or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT'])
    else:
        try:
            file_symbols = load_symbols()
            combined = sorted(set(file_symbols or []) | set(s.upper() for s in from_data) | set(from_info_csv))
            if not combined and not include_extended:
                fallback = _symbols_from_ticker_info_csv()
                if fallback:
                    return fallback
            return combined if combined else (from_data or from_info_csv or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT'])
        except Exception as e:
            print(f"Warning: Failed to load symbols: {e}")
            return from_data or from_info_csv or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT']


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
        out = await asyncio.to_thread(optimize_backtest, df, objective, maxiter, method)
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
        result = await asyncio.to_thread(run_trendline_backtest, df, interval, get_patterns_at_t)
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

    backtest_result = await asyncio.to_thread(run_trendline_backtest, df, interval, get_patterns_at_t)
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


# ── Agent API endpoints ────────────────────────────────────────────────────────

@app.get("/api/agent/status")
async def api_agent_status():
    """Get current agent status: equity, positions, trades, generation, etc."""
    return get_agent().get_status()


@app.post("/api/agent/start")
async def api_agent_start():
    """Start the agent background loop."""
    agent = get_agent()
    if agent._running:
        return {"ok": True, "message": "Agent already running"}
    agent.start()
    return {"ok": True, "message": "Agent started"}


@app.post("/api/agent/stop")
async def api_agent_stop():
    """Stop the agent background loop."""
    agent = get_agent()
    agent.stop()
    return {"ok": True, "message": "Agent stopped"}


@app.post("/api/agent/revive")
async def api_agent_revive():
    """Revive the agent after emergency shutdown."""
    agent = get_agent()
    agent.trader.revive()
    agent._save_state()
    return {"ok": True, "message": "Agent revived", "equity": agent.trader.state.equity}


@app.post("/api/agent/config")
async def api_agent_config(
    mode: str | None = Query(None, description="paper or live"),
    equity: float | None = Query(None, description="Set paper equity"),
):
    """Update agent config (mode, equity)."""
    agent = get_agent()
    if mode and mode in ("paper", "live"):
        if mode == "live" and not agent.trader.has_api_keys():
            return {"ok": False, "reason": "Cannot switch to live: OKX API keys not configured. Set keys first via /api/agent/okx-keys"}
        agent.trader.state.mode = mode
    if equity is not None and equity > 0:
        agent.trader.state.equity = equity
        agent.trader.state.peak_equity = max(agent.trader.state.peak_equity, equity)
        agent.trader.state.cash = equity
    agent._save_state()
    return {"ok": True, "state": agent.get_status()}


class OKXKeysRequest(BaseModel):
    api_key: str
    secret: str
    passphrase: str


@app.post("/api/agent/okx-keys")
async def api_agent_okx_keys(req: OKXKeysRequest):
    """Set OKX API keys for live trading. Keys are stored in memory only (not persisted to disk for security)."""
    agent = get_agent()
    agent.trader.set_api_keys(req.api_key, req.secret, req.passphrase)
    # Verify keys by checking account balance
    balance = await agent.trader.get_account_balance()
    if balance.get("ok"):
        return {
            "ok": True,
            "message": "OKX API keys verified successfully",
            "balance": balance,
            "has_keys": True,
        }
    return {
        "ok": False,
        "reason": f"Keys set but verification failed: {balance.get('reason', 'unknown')}",
        "has_keys": True,
    }


@app.get("/api/agent/okx-status")
async def api_agent_okx_status():
    """Check OKX connection status and account balance."""
    agent = get_agent()
    has_keys = agent.trader.has_api_keys()
    if not has_keys:
        return {"ok": True, "has_keys": False, "mode": agent.trader.state.mode}
    balance = await agent.trader.get_account_balance()
    return {
        "ok": True,
        "has_keys": True,
        "mode": agent.trader.state.mode,
        "balance": balance if balance.get("ok") else None,
        "error": balance.get("reason") if not balance.get("ok") else None,
    }


class StrategyConfigRequest(BaseModel):
    timeframe: str | None = None
    symbols: list[str] | None = None
    top_volume: int | None = None  # auto-select top N by 24h volume
    tick_interval: int | None = None
    max_position_pct: float | None = None
    max_positions: int | None = None


@app.post("/api/agent/strategy-config")
async def api_agent_strategy_config(req: StrategyConfigRequest):
    """Update strategy runtime config (timeframe, symbols, tick interval, risk)."""
    from . import agent_brain
    agent = get_agent()
    changes = []

    if req.timeframe and req.timeframe in ("5m", "15m", "1h", "4h", "1d"):
        agent_brain.SIGNAL_INTERVAL = req.timeframe
        changes.append(f"timeframe={req.timeframe}")

    if req.top_volume and 1 <= req.top_volume <= 50:
        from .data_service import get_top_volume_symbols
        top_syms = await get_top_volume_symbols(req.top_volume)
        if top_syms:
            agent_brain.WATCH_SYMBOLS = top_syms
            changes.append(f"top_{req.top_volume}_vol={top_syms[:5]}...")
    elif req.symbols and len(req.symbols) > 0:
        clean = [s.upper().replace("/", "").strip() for s in req.symbols if s.strip()]
        if clean:
            agent_brain.WATCH_SYMBOLS = clean
            changes.append(f"symbols={clean}")

    if req.tick_interval and 10 <= req.tick_interval <= 600:
        agent_brain.TICK_INTERVAL_SEC = req.tick_interval
        changes.append(f"tick={req.tick_interval}s")

    if req.max_position_pct and 0.5 <= req.max_position_pct <= 25:
        agent.trader.risk.max_position_pct = req.max_position_pct / 100.0
        changes.append(f"pos_pct={req.max_position_pct}%")

    if req.max_positions and 1 <= req.max_positions <= 10:
        agent.trader.risk.max_positions = req.max_positions
        changes.append(f"max_pos={req.max_positions}")

    if changes:
        print(f"[Agent] Config updated: {', '.join(changes)}")
        return {"ok": True, "changes": changes, "watch_symbols": agent_brain.WATCH_SYMBOLS}
    return {"ok": True, "changes": [], "message": "No changes"}


@app.post("/api/agent/strategy-params")
async def api_agent_strategy_params(req: dict = {}):
    """Update V6 strategy parameters. Accepts partial dict of param key:value."""
    agent = get_agent()
    params = agent.trader.state.strategy_params
    changes = []
    valid_keys = set(params.keys())

    for key, value in req.items():
        if key not in valid_keys:
            continue
        try:
            if isinstance(params[key], int):
                params[key] = int(value)
            else:
                params[key] = float(value)
            changes.append(f"{key}={params[key]}")
        except (ValueError, TypeError):
            pass

    if changes:
        agent._save_state()
        print(f"[Agent] Strategy params updated: {', '.join(changes)}")
    return {"ok": True, "changes": changes, "params": params}


class RiskLimitsRequest(BaseModel):
    max_position_pct: float | None = None
    max_total_exposure_pct: float | None = None
    max_daily_loss_pct: float | None = None
    max_drawdown_pct: float | None = None
    max_positions: int | None = None
    cooldown_seconds: int | None = None


@app.post("/api/agent/risk-limits")
async def api_agent_risk_limits(req: RiskLimitsRequest):
    """Update risk limits. Values are in percentage (e.g. 5 means 5%)."""
    agent = get_agent()
    changes = []

    if req.max_position_pct is not None and 0.5 <= req.max_position_pct <= 25:
        agent.trader.risk.max_position_pct = req.max_position_pct / 100.0
        changes.append(f"max_position_pct={req.max_position_pct}%")
    if req.max_total_exposure_pct is not None and 1 <= req.max_total_exposure_pct <= 100:
        agent.trader.risk.max_total_exposure_pct = req.max_total_exposure_pct / 100.0
        changes.append(f"max_total_exposure_pct={req.max_total_exposure_pct}%")
    if req.max_daily_loss_pct is not None and 0.1 <= req.max_daily_loss_pct <= 20:
        agent.trader.risk.max_daily_loss_pct = req.max_daily_loss_pct / 100.0
        changes.append(f"max_daily_loss_pct={req.max_daily_loss_pct}%")
    if req.max_drawdown_pct is not None and 1 <= req.max_drawdown_pct <= 50:
        agent.trader.risk.max_drawdown_pct = req.max_drawdown_pct / 100.0
        changes.append(f"max_drawdown_pct={req.max_drawdown_pct}%")
    if req.max_positions is not None and 1 <= req.max_positions <= 20:
        agent.trader.risk.max_positions = req.max_positions
        changes.append(f"max_positions={req.max_positions}")
    if req.cooldown_seconds is not None and 0 <= req.cooldown_seconds <= 86400:
        agent.trader.risk.cooldown_seconds = req.cooldown_seconds
        changes.append(f"cooldown={req.cooldown_seconds}s")

    if changes:
        print(f"[Agent] Risk limits updated: {', '.join(changes)}")
    return {"ok": True, "changes": changes, "risk_limits": {
        "max_position_pct": agent.trader.risk.max_position_pct * 100,
        "max_total_exposure_pct": agent.trader.risk.max_total_exposure_pct * 100,
        "max_daily_loss_pct": agent.trader.risk.max_daily_loss_pct * 100,
        "max_drawdown_pct": agent.trader.risk.max_drawdown_pct * 100,
        "max_positions": agent.trader.risk.max_positions,
        "cooldown_seconds": agent.trader.risk.cooldown_seconds,
    }}


@app.get("/api/agent/audit-log")
async def api_agent_audit_log(limit: int = Query(50, ge=1, le=500)):
    """Get recent entries from the trade audit log."""
    from .agent_brain import TRADE_AUDIT_LOG
    if not TRADE_AUDIT_LOG.exists():
        return {"entries": []}
    try:
        lines = TRADE_AUDIT_LOG.read_text(encoding="utf-8").strip().split("\n")
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return {"entries": entries}
    except Exception:
        return {"entries": []}


@app.get("/api/agent/lessons")
async def api_agent_lessons():
    """Get the agent's lessons ledger (举一反三 insights)."""
    agent = get_agent()
    return agent.lessons.get_summary()


@app.get("/api/top-volume")
async def api_top_volume(n: int = Query(20, ge=1, le=50)):
    """Get top N symbols by 24h trading volume."""
    from .data_service import get_top_volume_symbols
    symbols = await get_top_volume_symbols(n)
    return {"symbols": symbols, "count": len(symbols)}


@app.get("/api/data-info")
async def api_data_info(
    symbol: str = Query(...),
    interval: str = Query("4h"),
):
    """Return data completeness metadata for a symbol/interval."""
    from .data_service import _find_csv, _load_csv, INTERVAL_MS, DOWNLOAD_DAYS
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    info = {
        "symbol": symbol,
        "interval": interval,
        "has_local_csv": False,
        "total_bars": 0,
        "data_start": None,
        "data_end": None,
        "coverage_days": 0,
        "missing_bars_estimate": 0,
        "max_available_days": DOWNLOAD_DAYS.get(interval, 30),
    }

    csv_path = _find_csv(symbol, interval)
    if csv_path:
        info["has_local_csv"] = True
        try:
            df = _load_csv(csv_path)
            if not df.is_empty():
                info["total_bars"] = len(df)
                info["data_start"] = str(df["open_time"].min())
                info["data_end"] = str(df["open_time"].max())
                start_ts = df["open_time"].min().timestamp()
                end_ts = df["open_time"].max().timestamp()
                info["coverage_days"] = round((end_ts - start_ts) / 86400, 1)
                interval_sec = INTERVAL_MS.get(interval, 3600000) / 1000
                expected_bars = int((end_ts - start_ts) / interval_sec) + 1
                info["missing_bars_estimate"] = max(0, expected_bars - len(df))
        except Exception as e:
            info["error"] = str(e)

    return info


@app.get("/api/agent/signals")
async def api_agent_signals():
    """Run signal check on all watched symbols and return current signals."""
    agent = get_agent()
    signals = {}
    for symbol in agent._last_signals:
        signals[symbol] = agent._last_signals[symbol]
    # Also generate fresh signals in parallel
    from .agent_brain import WATCH_SYMBOLS

    async def _gen(sym: str):
        try:
            return sym, await agent.generate_signal(sym)
        except Exception:
            return sym, None

    results = await asyncio.gather(*[_gen(sym) for sym in WATCH_SYMBOLS])
    for sym, sig in results:
        if sig:
            signals[sym] = sig
    return {"signals": signals}


@app.get("/api/agent/logs")
async def api_agent_logs(limit: int = Query(50, ge=1, le=200), filter: str = Query("agent", description="Filter: 'agent' for agent-only, 'all' for everything")):
    """Get recent agent logs from the ring buffer."""
    all_logs = list(_LOG_BUFFER)
    if filter == "agent":
        # Only show agent/trader/strategy relevant logs, not HTTP access logs
        keywords = ["[Agent]", "[OKX]", "[Healer]", "[Data]", "V6", "Layer", "signal", "position", "trade", "Evolution", "SL:", "TP:"]
        all_logs = [l for l in all_logs if any(kw in l.get("msg", "") for kw in keywords)]
    logs = all_logs[-limit:]
    return {"logs": logs}


# ── AI Chat API endpoints ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str | None = None


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Send a message to the AI and get a response."""
    chat = get_chat()
    result = await chat.chat(req.message, req.session_id, req.model)
    return result


@app.get("/api/chat/models")
async def api_chat_models():
    """List available AI models."""
    return get_chat().list_models()


@app.get("/api/chat/history")
async def api_chat_history(session_id: str = Query("default")):
    """Get chat history for a session."""
    session = get_chat().get_session(session_id)
    return {"messages": session.messages, "model": session.model}


@app.post("/api/chat/clear")
async def api_chat_clear(session_id: str = Query("default")):
    """Clear chat history."""
    get_chat().clear_session(session_id)
    return {"ok": True}


# ── Self-Healer API ────────────────────────────────────────────────────────────

@app.get("/api/healer/status")
async def api_healer_status():
    """Get self-healer status: fix count, recent errors, last fix time."""
    return get_healer().get_status()


@app.post("/api/healer/trigger")
async def api_healer_trigger():
    """Manually trigger a heal attempt right now."""
    asyncio.create_task(get_healer().try_heal())
    return {"ok": True, "message": "Heal attempt triggered"}


@app.post("/api/healer/stop")
async def api_healer_stop():
    get_healer().stop()
    return {"ok": True, "message": "Healer stopped"}


@app.post("/api/healer/start")
async def api_healer_start():
    get_healer().start()
    return {"ok": True, "message": "Healer started"}


# ── Serve frontend files at root level (Vercel-compatible relative paths) ──
# This must come AFTER all /api/ routes so it doesn't shadow them
@app.get("/style.css")
async def serve_css():
    return FileResponse(str(PROJECT_ROOT / "frontend" / "style.css"), media_type="text/css",
                        headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/app.js")
async def serve_js():
    return FileResponse(str(PROJECT_ROOT / "frontend" / "app.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache, must-revalidate"})
