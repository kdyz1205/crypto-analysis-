"""
Multi-timeframe MA Ribbon / Golden Cross: state, weighted score, and success-rate backtest.

- Ribbon = bullish stack: Price > MA9 > MA21 > MA55 (golden cross when this becomes true).
- Timeframes: 15m (weight 1), 1h (2), 4h (3), 1d (4). Score 0–10.
- Higher timeframe alignment = higher success rate; backtest measures success rate by score band.
- Designed for integration with Open Claw / screeners: API returns current score and tier.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import polars as pl

# Same MA periods across all timeframes for comparable "ribbon"
MA_FAST = 9
MA_MID = 21
MA_SLOW = 55

# Weights by timeframe (daily = highest priority)
TF_WEIGHTS = {"15m": 1, "1h": 2, "4h": 3, "1d": 4}
TF_ORDER = ["1d", "4h", "1h", "15m"]  # evaluation order for display

# Score bands → tier labels (for Open Claw / UI)
SCORE_TIERS = [
    (0, 0, "none"),
    (1, 2, "low"),
    (3, 4, "medium"),
    (5, 6, "high"),
    (7, 8, "very_high"),
    (9, 10, "all_aligned"),
]


def _sma(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(period - 1, len(x)):
        out[i] = np.mean(x[i - period + 1 : i + 1])
    return out


def ribbon_state(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MA9, MA21, MA55 and in_ribbon (golden cross) boolean array.
    in_ribbon[i] = (close > MA9 > MA21 > MA55) at bar i.
    """
    ma9 = _sma(close, MA_FAST)
    ma21 = _sma(close, MA_MID)
    ma55 = _sma(close, MA_SLOW)
    start = MA_SLOW - 1
    in_ribbon = np.zeros(len(close), dtype=bool)
    for i in range(start, len(close)):
        if np.isnan(ma9[i]) or np.isnan(ma21[i]) or np.isnan(ma55[i]):
            continue
        if close[i] > ma9[i] and ma9[i] > ma21[i] and ma21[i] > ma55[i]:
            in_ribbon[i] = True
    return ma9, ma21, ma55, in_ribbon


def ribbon_state_last_row(df: pl.DataFrame) -> dict[str, Any]:
    """Get ribbon state at the last bar of a dataframe. Returns MAs and in_ribbon for that bar."""
    close = df["close"].to_numpy().astype(float)
    ma9, ma21, ma55, in_ribbon = ribbon_state(close)
    i = len(close) - 1
    return {
        "ma9": float(ma9[i]) if not np.isnan(ma9[i]) else None,
        "ma21": float(ma21[i]) if not np.isnan(ma21[i]) else None,
        "ma55": float(ma55[i]) if not np.isnan(ma55[i]) else None,
        "close": float(close[i]),
        "in_ribbon": bool(in_ribbon[i]),
    }


def score_from_states(tf_states: dict[str, bool]) -> int:
    """Compute weighted score from { "15m": bool, "1h": bool, "4h": bool, "1d": bool }."""
    return sum(TF_WEIGHTS.get(tf, 0) for tf, in_ribbon in tf_states.items() if in_ribbon)


def tier_from_score(score: int) -> str:
    for low, high, name in SCORE_TIERS:
        if low <= score <= high:
            return name
    return "none"


async def get_current_ribbon(symbol: str, get_ohlcv_with_df) -> dict[str, Any]:
    """
    Get current multi-timeframe ribbon state for a symbol (for API / Open Claw).
    get_ohlcv_with_df(symbol, interval, end_time=None, days) -> (df, result); must be async.
    """
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    tf_states: dict[str, dict[str, Any]] = {}
    score = 0

    for interval in TF_ORDER:
        try:
            df, _ = await get_ohlcv_with_df(symbol, interval, end_time=None, days=90)
            if df is None or df.is_empty():
                tf_states[interval] = {"in_ribbon": False, "ma9": None, "ma21": None, "ma55": None, "close": None}
                continue
            state = ribbon_state_last_row(df)
            tf_states[interval] = state
            if state["in_ribbon"]:
                score += TF_WEIGHTS[interval]
        except Exception:
            tf_states[interval] = {"in_ribbon": False, "ma9": None, "ma21": None, "ma55": None, "close": None}

    return {
        "symbol": symbol,
        "timeframes": tf_states,
        "score": score,
        "tier": tier_from_score(score),
        "weights": TF_WEIGHTS,
        "message": _tier_message(score),
    }


def _tier_message(score: int) -> str:
    if score >= 9:
        return "All timeframes aligned (1d+4h+1h+15m). Historically highest success rate."
    if score >= 7:
        return "Daily + 4h + at least one lower TF. Very high confidence."
    if score >= 5:
        return "Daily and 4h in ribbon. High confidence."
    if score >= 3:
        return "At least daily or 4h in ribbon. Medium confidence."
    if score >= 1:
        return "Only lower timeframes in ribbon. Lower success rate, more noise."
    return "No multi-timeframe ribbon alignment."


# --- Backtest: success rate by score band ---

@dataclass
class RibbonBacktestConfig:
    """Config for multi-TF ribbon success-rate backtest."""
    forward_bars: int = 5          # look forward this many bars (of anchor TF)
    success_pct: float = 2.0       # consider success if return >= this %
    fail_pct: float = -2.0         # consider fail if return <= this %
    min_score_to_signal: int = 3   # only count as signal when score >= this
    anchor_tf: str = "1d"          # bar timestamps from this TF


def _align_states_at_times(
    df_anchor: pl.DataFrame,
    dfs_by_tf: dict[str, pl.DataFrame],
) -> list[dict[str, Any]]:
    """
    For each bar index of anchor (e.g. daily), get the last closed bar of each TF at or before
    that bar's open_time. Return list of { "anchor_idx", "anchor_time", "15m": in_ribbon, ... }.
    """
    anchor_times = df_anchor["open_time"].to_list()
    anchor_close = df_anchor["close"].to_numpy().astype(float)

    # Precompute in_ribbon for each TF
    ribbon_by_tf: dict[str, np.ndarray] = {}
    times_by_tf: dict[str, pl.Series] = {}
    for tf, df in dfs_by_tf.items():
        if df is None or df.is_empty():
            ribbon_by_tf[tf] = np.array([])
            times_by_tf[tf] = pl.Series([])
            continue
        close = df["close"].to_numpy().astype(float)
        _, _, _, in_ribbon = ribbon_state(close)
        ribbon_by_tf[tf] = in_ribbon
        times_by_tf[tf] = df["open_time"]

    out = []
    for idx, t in enumerate(anchor_times):
        row = {"anchor_idx": idx, "anchor_time": t, "anchor_close": float(anchor_close[idx])}
        for tf in TF_ORDER:
            df = dfs_by_tf.get(tf)
            if df is None or df.is_empty() or ribbon_by_tf[tf].size == 0:
                row[tf] = False
                continue
            times = times_by_tf[tf]
            mask = times <= t
            if not mask.any():
                row[tf] = False
                continue
            mask_np = (times <= t).to_numpy() if hasattr(times, "to_numpy") else np.array(times <= t)
            last_idx = int(np.where(mask_np)[0][-1])
            in_ribbon = ribbon_by_tf[tf]
            row[tf] = bool(in_ribbon[last_idx]) if last_idx < len(in_ribbon) else False
        out.append(row)
    return out


def run_ribbon_backtest(
    symbol: str,
    load_df_fn: Callable[[str, str], pl.DataFrame | None] | None = None,
    config: RibbonBacktestConfig | None = None,
) -> dict[str, Any]:
    """
    Backtest: at each anchor bar where multi-TF score >= min_score_to_signal,
    look forward forward_bars bars; count success (return >= success_pct) vs fail (return <= fail_pct).
    Returns success_rate by score band and overall stats.
    load_df_fn(symbol, interval) -> pl.DataFrame | None.
    """
    cfg = config or RibbonBacktestConfig()
    if load_df_fn is None:
        from .backtest_service import load_df_from_csv
        load_df_fn = lambda sym, tf: load_df_from_csv(sym, tf)

    dfs: dict[str, pl.DataFrame | None] = {}
    for tf in TF_ORDER:
        df = load_df_fn(symbol, tf)
        dfs[tf] = df

    anchor_df = dfs.get(cfg.anchor_tf)
    if anchor_df is None or anchor_df.is_empty():
        return {"error": f"No data for {symbol} {cfg.anchor_tf}", "by_score": {}, "signals": 0}

    # Build aligned states at each anchor bar
    aligned = _align_states_at_times(anchor_df, dfs)
    close_arr = anchor_df["close"].to_numpy().astype(float)
    n = len(close_arr)

    # For each bar, compute score and forward return
    signals: list[dict[str, Any]] = []
    by_score: dict[int, list[float]] = {}  # score -> list of forward returns (pct)

    for i, row in enumerate(aligned):
        score = sum(TF_WEIGHTS[tf] for tf in TF_ORDER if row.get(tf))
        if score < cfg.min_score_to_signal:
            continue
        j = i + cfg.forward_bars
        if j >= n:
            continue
        entry = close_arr[i]
        exit_ = close_arr[j]
        ret_pct = (exit_ - entry) / entry * 100
        signals.append({
            "anchor_idx": i,
            "anchor_time": str(row["anchor_time"]),
            "score": score,
            "entry": entry,
            "exit": exit_,
            "return_pct": round(ret_pct, 2),
            "success": ret_pct >= cfg.success_pct,
            "fail": ret_pct <= cfg.fail_pct,
        })
        by_score.setdefault(score, []).append(ret_pct)

    # Aggregate by score band
    band_stats = []
    for low, high, tier_name in SCORE_TIERS:
        returns: list[float] = []
        for s in range(low, high + 1):
            returns.extend(by_score.get(s, []))
        if not returns:
            band_stats.append({"score_low": low, "score_high": high, "tier": tier_name, "count": 0, "success_rate_pct": None, "avg_return_pct": None})
            continue
        success_count = sum(1 for r in returns if r >= cfg.success_pct)
        band_stats.append({
            "score_low": low,
            "score_high": high,
            "tier": tier_name,
            "count": len(returns),
            "success_rate_pct": round(success_count / len(returns) * 100, 1),
            "avg_return_pct": round(float(np.mean(returns)), 2),
        })

    return {
        "symbol": symbol.upper(),
        "anchor_tf": cfg.anchor_tf,
        "forward_bars": cfg.forward_bars,
        "success_pct_threshold": cfg.success_pct,
        "fail_pct_threshold": cfg.fail_pct,
        "min_score_to_signal": cfg.min_score_to_signal,
        "total_signals": len(signals),
        "by_score_band": band_stats,
        "sample_signals": signals[:20],
    }
