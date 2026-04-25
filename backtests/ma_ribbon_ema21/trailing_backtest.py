"""Phase 2: real-trade simulation with EMA21-buffer trailing stop.

For each MA-ribbon formation event, open a long at next bar's open,
trail SL = EMA21 * (1 - buffer_pct) (non-decreasing), exit when bar.low
breaks the trail. Records full trade list with entry/exit/holding/PnL.

This is the FIRST module that actually simulates filled trades on
historical data — Phase 1 only measured forward returns at fixed
horizons. From here on we can compute Sharpe, max drawdown, profit
factor, expectancy — all the strategy-tester metrics.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class Trade:
    symbol: str
    tf: str
    entry_bar_idx: int
    entry_timestamp: int
    entry_price: float
    exit_bar_idx: int
    exit_timestamp: int
    exit_price: float
    exit_reason: str            # "trailing_sl" | "data_end"
    holding_bars: int
    raw_return: float           # (exit - entry) / entry
    post_fee_return: float      # raw - 2 * (fee + slip)
    max_favorable_pct: float    # MFE: best unrealized gain during trade
    max_adverse_pct: float      # MAE: worst unrealized loss during trade


def backtest_one(
    df: pd.DataFrame,
    formation_idxs: Iterable[int],
    buffer_pct: float,
    symbol: str = "",
    tf: str = "",
    fee_per_side: float = 0.0005,
    slippage_per_fill: float = 0.0001,
) -> list[Trade]:
    """Simulate one (symbol, TF) backtest given a precomputed alignment.

    df MUST have columns: timestamp, open, high, low, close, ema21.
    formation_idxs is the integer-position list of bars where bullish
    alignment first formed (i.e. False -> True transition).

    Trade lifecycle:
      - On formation at bar i: enter at bar (i+1).open.
      - Initial SL = ema21[i] * (1 - buffer_pct).
      - At each subsequent bar j > i+1:
          - If low[j] <= sl  -> exit. Fill at bar.open if gap below sl,
            else fill at sl. exit_reason = "trailing_sl".
          - Else, at close of bar j, sl = max(sl, ema21[j] * (1 - buffer_pct)).
      - If data ends with trade open: exit at last bar's close,
        exit_reason = "data_end".

    Returns: list of Trade.
    """
    if buffer_pct <= 0:
        raise ValueError(f"buffer_pct must be > 0, got {buffer_pct}")
    required = ["timestamp", "open", "high", "low", "close", "ema21"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"backtest_one: df missing columns {missing}")
    n = len(df)
    if n == 0:
        return []

    o  = df["open"].to_numpy(dtype=float)
    h  = df["high"].to_numpy(dtype=float)
    l  = df["low"].to_numpy(dtype=float)
    c  = df["close"].to_numpy(dtype=float)
    e21 = df["ema21"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=np.int64)

    cost_round_trip = 2.0 * (fee_per_side + slippage_per_fill)
    formations = sorted({int(i) for i in formation_idxs})
    trades: list[Trade] = []

    # Avoid overlapping trades: if a formation fires while we're still in a
    # previous trade, skip it. (Phase 2 = single-layer; pyramiding is Phase 3.)
    next_formation = 0

    while next_formation < len(formations):
        signal_bar = formations[next_formation]
        next_formation += 1
        entry_bar = signal_bar + 1
        if entry_bar >= n:
            break  # no room to enter

        entry_price = float(o[entry_bar])
        if entry_price <= 0 or math.isnan(entry_price):
            continue

        # Initial SL — based on ema21 of the last CLOSED bar (the signal bar).
        if math.isnan(e21[signal_bar]):
            continue
        sl = float(e21[signal_bar]) * (1.0 - buffer_pct)

        mfe = 0.0
        mae = 0.0
        exit_idx = -1
        exit_price = 0.0
        exit_reason = "data_end"

        # Walk forward
        j = entry_bar
        # Optional: capture intra-bar move on entry bar BEFORE first trail.
        # Skip exit check on the entry bar (instant entry; no time elapsed).
        bar_max = float(h[j])
        bar_min = float(l[j])
        mfe = max(mfe, (bar_max - entry_price) / entry_price)
        mae = min(mae, (bar_min - entry_price) / entry_price)
        # Trail SL at close of entry bar.
        if not math.isnan(e21[j]):
            sl = max(sl, float(e21[j]) * (1.0 - buffer_pct))

        j += 1
        while j < n:
            bar_o = float(o[j])
            bar_h = float(h[j])
            bar_l = float(l[j])

            # MFE / MAE update (intra-bar extremes vs entry)
            mfe = max(mfe, (bar_h - entry_price) / entry_price)
            mae = min(mae, (bar_l - entry_price) / entry_price)

            # Exit check first (uses SL set at prior bar's close).
            if bar_l <= sl:
                # Gap-down -> fill at open; intra-bar -> fill at sl.
                exit_price = bar_o if bar_o <= sl else sl
                exit_idx = j
                exit_reason = "trailing_sl"
                break

            # Trail SL at this bar's close.
            if not math.isnan(e21[j]):
                candidate = float(e21[j]) * (1.0 - buffer_pct)
                if candidate > sl:
                    sl = candidate

            j += 1

        if exit_idx < 0:
            # Data ran out while in trade -> exit at last close.
            exit_idx = n - 1
            exit_price = float(c[exit_idx])
            exit_reason = "data_end"

        raw = (exit_price - entry_price) / entry_price
        post = raw - cost_round_trip
        trades.append(Trade(
            symbol=symbol, tf=tf,
            entry_bar_idx=entry_bar, entry_timestamp=int(ts[entry_bar]),
            entry_price=entry_price,
            exit_bar_idx=exit_idx, exit_timestamp=int(ts[exit_idx]),
            exit_price=exit_price, exit_reason=exit_reason,
            holding_bars=exit_idx - entry_bar,
            raw_return=raw, post_fee_return=post,
            max_favorable_pct=mfe, max_adverse_pct=mae,
        ))

        # Skip any formation indices that fall inside this trade's holding window.
        while next_formation < len(formations) and formations[next_formation] <= exit_idx:
            next_formation += 1

    return trades


# ──────────────────────────── aggregate metrics ────────────────────────────


@dataclass
class TrailingMetrics:
    symbol: str
    tf: str
    buffer_pct: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_return_post_fee: float   # compounded, e.g. 0.12 = +12% over the period
    mean_return_post_fee: float
    median_return_post_fee: float
    profit_factor: float           # sum(gains) / abs(sum(losses))
    max_drawdown: float            # peak-to-trough on equity curve
    sharpe_per_trade: float        # mean / std of post-fee returns
    avg_holding_bars: float
    avg_mae: float                 # avg max adverse excursion %
    avg_mfe: float                 # avg max favorable excursion %
    train_count: int               # trades whose entry was in train split
    test_count: int
    train_mean_return_post_fee: float
    test_mean_return_post_fee: float
    sample_trades: list[dict] = field(default_factory=list)


def _equity_curve(post_fee_returns: list[float]) -> list[float]:
    eq = 1.0
    out: list[float] = []
    for r in post_fee_returns:
        eq *= (1.0 + r)
        out.append(eq)
    return out


def _max_drawdown(equity: list[float]) -> float:
    peak = -1e18
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            dd = (v - peak) / peak
            if dd < mdd:
                mdd = dd
    return mdd  # negative number, e.g. -0.15 = 15% peak-to-trough


def aggregate_trades(
    trades: list[Trade],
    train_pct: float = 0.70,
    n_bars_total: int | None = None,
    sample_size: int = 8,
    buffer_pct: float = 0.0,
) -> TrailingMetrics:
    """Aggregate one (sym, TF, buffer)'s Trade list into metrics."""
    sym = trades[0].symbol if trades else ""
    tf  = trades[0].tf if trades else ""
    if not trades:
        return TrailingMetrics(
            symbol=sym, tf=tf, buffer_pct=buffer_pct,
            trades=0, wins=0, losses=0, win_rate=0.0,
            total_return_post_fee=0.0,
            mean_return_post_fee=0.0, median_return_post_fee=0.0,
            profit_factor=0.0, max_drawdown=0.0,
            sharpe_per_trade=0.0, avg_holding_bars=0.0,
            avg_mae=0.0, avg_mfe=0.0,
            train_count=0, test_count=0,
            train_mean_return_post_fee=0.0,
            test_mean_return_post_fee=0.0,
        )

    post_fees = [t.post_fee_return for t in trades]
    wins = sum(1 for r in post_fees if r > 0)
    losses = sum(1 for r in post_fees if r <= 0)
    pos_sum = sum(r for r in post_fees if r > 0)
    neg_sum = sum(r for r in post_fees if r < 0)
    pf = pos_sum / abs(neg_sum) if neg_sum < 0 else (math.inf if pos_sum > 0 else 0.0)
    eq = _equity_curve(post_fees)
    total_ret = eq[-1] - 1.0 if eq else 0.0
    mdd = _max_drawdown(eq)
    arr = np.asarray(post_fees, dtype=float)
    mean_r = float(arr.mean())
    median_r = float(np.median(arr))
    std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = mean_r / std_r if std_r > 1e-12 else 0.0

    # split by entry_bar_idx vs total bars
    train_count = test_count = 0
    train_rets: list[float] = []
    test_rets:  list[float] = []
    cutoff = int((n_bars_total or 0) * train_pct) if n_bars_total else 0
    for t in trades:
        if cutoff > 0 and t.entry_bar_idx < cutoff:
            train_count += 1
            train_rets.append(t.post_fee_return)
        else:
            test_count += 1
            test_rets.append(t.post_fee_return)

    sample = trades[:sample_size] + trades[-sample_size:] if len(trades) > 2 * sample_size else trades

    return TrailingMetrics(
        symbol=sym, tf=tf, buffer_pct=buffer_pct,
        trades=len(trades),
        wins=wins, losses=losses,
        win_rate=wins / len(trades) if trades else 0.0,
        total_return_post_fee=total_ret,
        mean_return_post_fee=mean_r,
        median_return_post_fee=median_r,
        profit_factor=pf,
        max_drawdown=mdd,
        sharpe_per_trade=sharpe,
        avg_holding_bars=float(np.mean([t.holding_bars for t in trades])),
        avg_mae=float(np.mean([t.max_adverse_pct for t in trades])),
        avg_mfe=float(np.mean([t.max_favorable_pct for t in trades])),
        train_count=train_count, test_count=test_count,
        train_mean_return_post_fee=(float(np.mean(train_rets)) if train_rets else 0.0),
        test_mean_return_post_fee=(float(np.mean(test_rets)) if test_rets else 0.0),
        sample_trades=[{
            "entry_ts":     t.entry_timestamp,
            "entry_price":  round(t.entry_price, 6),
            "exit_ts":      t.exit_timestamp,
            "exit_price":   round(t.exit_price, 6),
            "holding_bars": t.holding_bars,
            "post_fee_return": round(t.post_fee_return, 5),
            "exit_reason":  t.exit_reason,
        } for t in sample],
    )
