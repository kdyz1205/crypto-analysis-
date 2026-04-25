"""Phase 2 engine: trailing-SL backtest across the live universe.

Re-uses Phase 1's alignment / formation detection but, instead of
measuring forward returns at fixed horizons, simulates the actual
trade with an EMA21 trailing stop. Aggregates per-(symbol, TF) metrics
and a portfolio-wide summary.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from backtests.ma_ribbon_ema21.indicators import sma, ema
from backtests.ma_ribbon_ema21.ma_alignment import (
    AlignmentConfig, bullish_aligned, formation_events,
)
from backtests.ma_ribbon_ema21.trailing_backtest import (
    Trade, TrailingMetrics, backtest_one, aggregate_trades,
)


_LOG = logging.getLogger(__name__)


@dataclass
class PortfolioSummary:
    total_trades: int
    total_wins: int
    win_rate: float
    portfolio_total_return_post_fee: float   # equal-weighted compound
    portfolio_max_drawdown: float
    portfolio_sharpe_per_trade: float
    portfolio_profit_factor: float
    avg_holding_bars: float
    symbols_with_positive_test_return: int
    symbols_total: int
    pairs_evaluated: int


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma5"]   = sma(out["close"], 5)
    out["ma8"]   = sma(out["close"], 8)
    out["ema21"] = ema(out["close"], 21)
    out["ma55"]  = sma(out["close"], 55)
    return out


def scan_pair_phase2(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    buffer_pct: float,
    alignment_cfg: AlignmentConfig | None = None,
    fee_per_side: float = 0.0005,
    slippage_per_fill: float = 0.0001,
    train_pct: float = 0.70,
) -> tuple[TrailingMetrics, list[Trade]]:
    """One (symbol, TF, buffer) backtest. Returns (metrics, trade list)."""
    if df.empty:
        return aggregate_trades([], train_pct=train_pct,
                                buffer_pct=buffer_pct), []
    enriched = _enrich(df)
    aligned = bullish_aligned(enriched, alignment_cfg or AlignmentConfig.default())
    events = list(formation_events(aligned))
    trades = backtest_one(
        df=enriched,
        formation_idxs=events,
        buffer_pct=buffer_pct,
        symbol=symbol, tf=tf,
        fee_per_side=fee_per_side,
        slippage_per_fill=slippage_per_fill,
    )
    metrics = aggregate_trades(
        trades, train_pct=train_pct,
        n_bars_total=len(enriched), buffer_pct=buffer_pct,
    )
    return metrics, trades


def scan_universe_phase2(
    data: dict[tuple[str, str], pd.DataFrame],
    buffer_pct: float,
    alignment_cfg: AlignmentConfig | None = None,
    fee_per_side: float = 0.0005,
    slippage_per_fill: float = 0.0001,
    train_pct: float = 0.70,
) -> tuple[list[TrailingMetrics], PortfolioSummary]:
    """Run Phase 2 on every (sym, TF) DataFrame in `data`.

    Returns: (per-pair metrics list, portfolio summary).
    """
    per_pair: list[TrailingMetrics] = []
    all_trades: list[Trade] = []
    pairs_evaluated = 0
    for (sym, tf), df in data.items():
        if df.empty:
            continue
        m, trades = scan_pair_phase2(
            df=df, symbol=sym, tf=tf, buffer_pct=buffer_pct,
            alignment_cfg=alignment_cfg,
            fee_per_side=fee_per_side, slippage_per_fill=slippage_per_fill,
            train_pct=train_pct,
        )
        per_pair.append(m)
        all_trades.extend(trades)
        pairs_evaluated += 1

    # ── portfolio: equal-weight ALL trades, regardless of (sym, TF) ──
    if all_trades:
        post = [t.post_fee_return for t in all_trades]
        eq = 1.0
        peak = 1.0
        mdd = 0.0
        eq_series: list[float] = []
        for r in post:
            eq *= (1.0 + r)
            eq_series.append(eq)
            peak = max(peak, eq)
            mdd = min(mdd, (eq - peak) / peak)
        wins = sum(1 for r in post if r > 0)
        pos = sum(r for r in post if r > 0)
        neg = sum(r for r in post if r < 0)
        pf = (pos / abs(neg)) if neg < 0 else (float("inf") if pos > 0 else 0.0)
        arr = np.asarray(post, dtype=float)
        mean_r = float(arr.mean())
        std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sharpe = mean_r / std_r if std_r > 1e-12 else 0.0
        avg_hold = float(np.mean([t.holding_bars for t in all_trades]))
    else:
        eq = 1.0
        mdd = 0.0
        wins = 0
        pf = 0.0
        sharpe = 0.0
        avg_hold = 0.0

    # symbol-level positive-test count
    sym_pos = 0
    sym_set: set[str] = set()
    sym_test_returns: dict[str, list[float]] = {}
    for m in per_pair:
        sym_set.add(m.symbol)
        if m.test_count > 0:
            sym_test_returns.setdefault(m.symbol, []).append(m.test_mean_return_post_fee)
    for sym, rets in sym_test_returns.items():
        if any(r > 0 for r in rets):
            sym_pos += 1

    return per_pair, PortfolioSummary(
        total_trades=len(all_trades),
        total_wins=wins,
        win_rate=(wins / len(all_trades)) if all_trades else 0.0,
        portfolio_total_return_post_fee=eq - 1.0,
        portfolio_max_drawdown=mdd,
        portfolio_sharpe_per_trade=sharpe,
        portfolio_profit_factor=pf,
        avg_holding_bars=avg_hold,
        symbols_with_positive_test_return=sym_pos,
        symbols_total=len(sym_set),
        pairs_evaluated=pairs_evaluated,
    )
