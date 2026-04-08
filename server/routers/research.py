"""
Research routes: backtest, optimize, MA ribbon analysis/backtest.
"""

import asyncio

from fastapi import APIRouter, Query, HTTPException

from ..data_service import get_ohlcv_with_df, API_ONLY
from ..backtest_service import run_backtest, load_df_from_csv, BacktestParams, optimize_backtest
from ..ma_ribbon_service import get_current_ribbon, run_ribbon_backtest, RibbonBacktestConfig

router = APIRouter(prefix="/api", tags=["research"])


async def _load_df_for_analysis(symbol: str, interval: str, end_time=None, days: int = 365):
    """Load OHLCV as polars DataFrame. CSV first, API fallback."""
    if API_ONLY:
        return await get_ohlcv_with_df(symbol, interval, end_time, days)
    df = load_df_from_csv(symbol, interval)
    if df is not None and not df.is_empty():
        return df, None
    return await get_ohlcv_with_df(symbol, interval, end_time, days)


@router.get("/backtest")
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
    """Run MFI/MA backtest. Optional query params override strategy params."""
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


@router.post("/backtest/optimize")
async def api_backtest_optimize(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("4h"),
    days: int = Query(365),
    objective: str = Query("total_pnl", description="total_pnl | sharpe | win_rate"),
    maxiter: int = Query(80, description="Max optimization steps"),
    method: str = Query("L-BFGS-B", description="L-BFGS-B or Nelder-Mead"),
):
    """Optimize strategy parameters to maximize total_pnl, sharpe, or win_rate."""
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


@router.get("/ma-ribbon")
async def api_ma_ribbon(symbol: str = Query(..., description="e.g. BTCUSDT or BTC")):
    """
    Current multi-timeframe MA ribbon state: 15m, 1h, 4h, 1d.
    Score 0-10; tier = none | low | medium | high | very_high | all_aligned.
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


@router.get("/ma-ribbon/backtest")
async def api_ma_ribbon_backtest(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    anchor_tf: str = Query("1d", description="Anchor timeframe for bars"),
    forward_bars: int = Query(5, description="Bars to look forward for success/fail"),
    success_pct: float = Query(2.0, description="Return >= this % counts as success"),
    fail_pct: float = Query(-2.0, description="Return <= this % counts as fail"),
    min_score: int = Query(3, description="Only count signal when score >= this"),
):
    """Backtest multi-timeframe ribbon: success rate by score band."""
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
