"""Backtest service: MFI/MA strategy, returns result + trade markers for chart overlay.

Strategy rules: 入场 Price > MFI > MA8 > EMA21 > MA55 | SL: EMA21 - atr_sl_mult*ATR | TP: BB upper.
See STRATEGY_RULES.md for full description.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class BacktestParams:
    """Strategy parameters for MFI/MA backtest. Used for tuning and gradient descent."""
    mfi_period: int = 14
    ma_fast: int = 8
    ema_span: int = 21
    ma_slow: int = 55
    atr_period: int = 14
    atr_sl_mult: float = 2.5
    bb_period: int = 21
    bb_std: float = 2.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "mfi_period": self.mfi_period,
            "ma_fast": self.ma_fast,
            "ema_span": self.ema_span,
            "ma_slow": self.ma_slow,
            "atr_period": self.atr_period,
            "atr_sl_mult": self.atr_sl_mult,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BacktestParams":
        defaults = cls()
        return cls(
            mfi_period=int(d.get("mfi_period", defaults.mfi_period)),
            ma_fast=int(d.get("ma_fast", defaults.ma_fast)),
            ema_span=int(d.get("ema_span", defaults.ema_span)),
            ma_slow=int(d.get("ma_slow", defaults.ma_slow)),
            atr_period=int(d.get("atr_period", defaults.atr_period)),
            atr_sl_mult=float(d.get("atr_sl_mult", defaults.atr_sl_mult)),
            bb_period=int(d.get("bb_period", defaults.bb_period)),
            bb_std=float(d.get("bb_std", defaults.bb_std)),
        )


def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    pos = np.where(tp > np.roll(tp, 1), raw_mf, 0.0)
    neg = np.where(tp < np.roll(tp, 1), raw_mf, 0.0)
    pos[0], neg[0] = 0, 0
    pos_sum = np.convolve(pos, np.ones(period), mode="valid")
    neg_sum = np.convolve(neg, np.ones(period), mode="valid")
    mfi_arr = np.full(len(close), np.nan)
    valid = neg_sum > 1e-12
    ratio = np.where(valid, pos_sum / np.maximum(neg_sum, 1e-12), 0.0)
    mfi_arr[period - 1 : period - 1 + len(pos_sum)] = np.where(valid, 100 - 100 / (1 + ratio), 100.0)
    return mfi_arr


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    a = 2.0 / (span + 1)
    out = np.empty_like(x, dtype=float)
    out[:] = np.nan
    # Find first non-NaN to seed the EMA (avoids propagating leading NaN)
    first_valid = next((i for i, v in enumerate(x) if not np.isnan(v)), None)
    if first_valid is None:
        return out
    out[first_valid] = x[first_valid]
    for i in range(first_valid + 1, len(x)):
        if np.isnan(x[i]):
            out[i] = out[i - 1]  # hold last valid value
        else:
            out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _sma(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(period - 1, len(x)):
        out[i] = np.mean(x[i - period + 1 : i + 1])
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR using simple moving average of true range. Returns full-length array (NaN-padded)."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr_conv = np.convolve(tr, np.ones(period) / period, mode="valid")
    out = np.full(len(close), np.nan)
    out[period - 1 : period - 1 + len(atr_conv)] = atr_conv
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (0-100) using Wilder's smoothing. Full-length array, NaN-padded."""
    out = np.full(len(close), np.nan)
    if len(close) < period + 1:
        return out
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss < 1e-12:
        out[period] = 100.0
    else:
        out[period] = 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-12:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def _bb_upper(close: np.ndarray, period: int = 21, num_std: float = 2.2) -> np.ndarray:
    out = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        w = close[i - period + 1 : i + 1]
        out[i] = np.mean(w) + num_std * np.std(w)
    return out


def run_backtest(df: pl.DataFrame, params: BacktestParams | None = None) -> dict[str, Any]:
    """
    Run MFI/MA strategy backtest. Returns summary + trades with entry/exit timestamps.
    Uses BacktestParams for all strategy parameters (defaults match STRATEGY_RULES.md).
    """
    p = params or BacktestParams()
    high = df["high"].to_numpy().astype(float)
    low = df["low"].to_numpy().astype(float)
    close = df["close"].to_numpy().astype(float)
    volume = df["volume"].to_numpy().astype(float)
    times_sec = df["open_time"].dt.epoch("s").to_list()

    mfi_arr = _mfi(high, low, close, volume, p.mfi_period)
    ma_fast = _sma(close, p.ma_fast)
    ema_arr = _ema(close, p.ema_span)
    ma_slow = _sma(close, p.ma_slow)
    atr_arr = _atr(high, low, close, p.atr_period)
    bb_upper = _bb_upper(close, p.bb_period, p.bb_std)
    rsi_arr = _rsi(close, 14)

    trades = []
    position = None
    entry_price = 0.0
    entry_idx = 0
    sl_price = 0.0
    tp_price = 0.0
    peak_price = 0.0  # for trailing stop
    entry_atr = 0.0
    start_idx = max(p.ma_slow, p.bb_period, p.atr_period)

    for i in range(start_idx, len(close)):
        if position is not None:
            # Trailing stop: update peak and trail SL
            if high[i] > peak_price:
                peak_price = high[i]
                # Once price moves 1×ATR in our favor, trail at peak - sl_mult×ATR
                if peak_price > entry_price + entry_atr:
                    new_sl = peak_price - p.atr_sl_mult * entry_atr
                    sl_price = max(sl_price, new_sl)  # only ratchet up

            # Check SL hit
            if low[i] <= sl_price:
                exit_price = sl_price
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i, "exit_reason": "SL", "pnl_pct": pnl_pct,
                    "entry_time": times_sec[entry_idx], "exit_time": times_sec[i],
                    "entry_price": entry_price, "exit_price": exit_price,
                })
                position = None
                continue
            # Check TP hit
            if high[i] >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price * 100
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i, "exit_reason": "TP", "pnl_pct": pnl_pct,
                    "entry_time": times_sec[entry_idx], "exit_time": times_sec[i],
                    "entry_price": entry_price, "exit_price": tp_price,
                })
                position = None
                continue
            # MA breakdown exit: if price drops below EMA, close
            if close[i] < ema_arr[i] and not np.isnan(ema_arr[i]):
                pnl_pct = (close[i] - entry_price) / entry_price * 100
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i, "exit_reason": "EMA_BREAK", "pnl_pct": pnl_pct,
                    "entry_time": times_sec[entry_idx], "exit_time": times_sec[i],
                    "entry_price": entry_price, "exit_price": close[i],
                })
                position = None
            continue

        if (np.isnan(mfi_arr[i]) or np.isnan(ma_fast[i]) or np.isnan(ema_arr[i]) or np.isnan(ma_slow[i]) or
                np.isnan(atr_arr[i]) or np.isnan(bb_upper[i]) or np.isnan(rsi_arr[i])):
            continue

        # Entry: MA stacking + MFI > 50 + RSI between 40-70 (momentum but not overbought)
        rsi_ok = 40 < rsi_arr[i] < 70
        if (close[i] > ma_fast[i] and ma_fast[i] > ema_arr[i] and ema_arr[i] > ma_slow[i]
                and mfi_arr[i] > 50 and rsi_ok and close[i] > 0):
            position = "long"
            entry_price = close[i]
            entry_idx = i
            entry_atr = atr_arr[i]
            peak_price = close[i]
            sl_price = ema_arr[i] - p.atr_sl_mult * atr_arr[i]
            tp_price = bb_upper[i]

    if position is not None:
        pnl_pct = (close[-1] - entry_price) / entry_price * 100
        trades.append({
            "entry_idx": entry_idx, "exit_idx": len(close) - 1, "exit_reason": "EOD", "pnl_pct": pnl_pct,
            "entry_time": times_sec[entry_idx], "exit_time": times_sec[-1],
            "entry_price": entry_price, "exit_price": float(close[-1]),
        })

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    total_pnl = sum(t["pnl_pct"] for t in trades)

    # Sharpe ratio (mean / std of trade PnL%)
    sharpe = 0.0
    if len(trades) >= 2:
        pnls = np.array([t["pnl_pct"] for t in trades])
        std_pnl = np.std(pnls)
        if std_pnl > 1e-8:
            sharpe = round(float(np.mean(pnls) / std_pnl), 4)

    # Max drawdown (from cumulative PnL curve)
    max_drawdown_pct = 0.0
    if trades:
        cum = np.cumsum([t["pnl_pct"] for t in trades])
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_drawdown_pct = round(float(np.max(dd)), 2) if len(dd) > 0 else 0.0

    # Profit factor
    gross_win = sum(t["pnl_pct"] for t in wins) if wins else 0.0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0.0
    profit_factor = round(gross_win / max(gross_loss, 1e-8), 2)

    # Composite score (autoresearch metric: higher is better)
    # score = sharpe * sqrt(trade_factor) - drawdown_penalty
    import math
    trade_factor = min(len(trades) / 20.0, 1.0)
    dd_penalty = max(0, max_drawdown_pct - 10.0) * 0.05
    score = round(sharpe * math.sqrt(trade_factor) - dd_penalty, 4) if len(trades) >= 3 else -999.0

    strategy_desc = (
        f"MFI/MA: Price>MFI>MA{p.ma_fast}>EMA{p.ema_span}>MA{p.ma_slow} | "
        f"SL: EMA{p.ema_span}-{p.atr_sl_mult}*ATR | TP: BB({p.bb_period},{p.bb_std})"
    )
    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl_pct": round(total_pnl, 2),
        "avg_win": round(float(np.mean([t["pnl_pct"] for t in wins])), 2) if wins else 0,
        "avg_loss": round(float(np.mean([t["pnl_pct"] for t in losses])), 2) if losses else 0,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": profit_factor,
        "score": score,
        "trades": trades,
        "strategy": strategy_desc,
        "params": p.to_dict(),
    }


def load_df_from_csv(symbol: str, interval: str) -> pl.DataFrame | None:
    from .data_service import _find_csv, _load_csv
    path = _find_csv(symbol, interval)
    if path is None:
        return None
    try:
        return _load_csv(path)
    except Exception:
        return None


# --- Parameter optimization (gradient-free: minimize negative objective) ---

# Bounds for [mfi_period, ma_fast, ema_span, ma_slow, atr_period, atr_sl_mult, bb_period, bb_std]
OPTIMIZE_BOUNDS = [
    (8, 22),   # mfi_period
    (5, 14),   # ma_fast
    (14, 35),  # ema_span
    (40, 75),  # ma_slow
    (8, 22),   # atr_period
    (0.5, 2.5),  # atr_sl_mult
    (14, 30),  # bb_period
    (1.5, 3.2),  # bb_std
]


def _x_to_params(x: np.ndarray) -> BacktestParams:
    """Map optimizer vector to BacktestParams (round ints, clip to valid)."""
    mfi = int(np.clip(round(x[0]), 5, 30))
    ma_fast = int(np.clip(round(x[1]), 3, 20))
    ema_span = int(np.clip(round(x[2]), 10, 40))
    ma_slow = int(np.clip(round(x[3]), 30, 90))
    atr_period = int(np.clip(round(x[4]), 5, 25))
    atr_sl_mult = float(np.clip(x[5], 0.3, 3.0))
    bb_period = int(np.clip(round(x[6]), 10, 35))
    bb_std = float(np.clip(x[7], 1.2, 4.0))
    return BacktestParams(
        mfi_period=mfi,
        ma_fast=ma_fast,
        ema_span=ema_span,
        ma_slow=ma_slow,
        atr_period=atr_period,
        atr_sl_mult=atr_sl_mult,
        bb_period=bb_period,
        bb_std=bb_std,
    )


def optimize_backtest(
    df: pl.DataFrame,
    objective: str = "total_pnl",
    maxiter: int = 80,
    method: str = "L-BFGS-B",
) -> dict[str, Any]:
    """
    Optimize strategy parameters by minimizing negative objective (e.g. -total_pnl_pct).
    Uses scipy.optimize.minimize; objective is non-smooth so we use gradient-free or L-BFGS-B
    with numerical gradient.
    """
    from scipy.optimize import minimize

    # Initial point: default params as float vector
    x0 = np.array([
        14, 8, 21, 55, 14, 1.0, 21, 2.2,
    ], dtype=float)

    def loss(x: np.ndarray) -> float:
        params = _x_to_params(x)
        # Enforce ma_fast < ema_span < ma_slow for valid stacking
        if params.ma_fast >= params.ema_span or params.ema_span >= params.ma_slow:
            return 1e6  # penalty
        try:
            res = run_backtest(df, params)
        except Exception:
            return 1e6
        if objective == "total_pnl":
            return -res["total_pnl_pct"]
        if objective == "sharpe":
            trades = res.get("trades", [])
            if len(trades) < 2:
                return 1e6
            pnls = np.array([t["pnl_pct"] for t in trades])
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            if std_pnl < 1e-8:
                return -mean_pnl
            return -mean_pnl / std_pnl  # minimize negative Sharpe
        if objective == "win_rate":
            return -res["win_rate"]
        return -res["total_pnl_pct"]

    # Prefer derivative-free (backtest is non-smooth). Nelder-Mead doesn't take bounds; we clip in _x_to_params.
    if method == "Nelder-Mead":
        result = minimize(loss, x0, method="Nelder-Mead", options={"maxfev": maxiter})
    else:
        result = minimize(
            loss,
            x0,
            method=method,
            bounds=OPTIMIZE_BOUNDS,
            options={"maxiter": maxiter} if method == "L-BFGS-B" else {"maxfev": maxiter},
        )

    best_params = _x_to_params(result.x)
    best_result = run_backtest(df, best_params)
    return {
        "best_params": best_params.to_dict(),
        "best_result": best_result,
        "optimization": {
            "objective": objective,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(getattr(result, "nfev", 0) or getattr(result, "njev", 0) * 2),
        },
    }
