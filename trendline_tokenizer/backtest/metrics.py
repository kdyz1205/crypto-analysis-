"""Trade metrics: hit rate, expectancy, max drawdown, Sharpe."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .strategy_simulator import Trade


@dataclass
class TradeMetrics:
    n_trades: int
    hit_rate: float
    avg_return_pct: float
    median_return_pct: float
    expectancy_pct: float
    cumulative_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rr: float          # mean(win) / mean(|loss|)
    n_long: int
    n_short: int
    n_stop: int
    n_expiry: int


def _max_drawdown(cum_returns: np.ndarray) -> float:
    """Max peak-to-trough drawdown, expressed as a positive %."""
    if len(cum_returns) == 0:
        return 0.0
    peak = np.maximum.accumulate(cum_returns)
    dd = (peak - cum_returns)
    return float(dd.max())


def compute_metrics(trades: Iterable[Trade]) -> TradeMetrics:
    trades = list(trades)
    n = len(trades)
    if n == 0:
        return TradeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    rets = np.array([t.return_pct for t in trades], dtype=np.float64)
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    cum = np.cumsum(rets)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(len(rets))) if rets.std() > 0 else 0.0
    win_rr = float(wins.mean() / abs(losses.mean())) if len(losses) > 0 and losses.mean() < 0 else 0.0

    return TradeMetrics(
        n_trades=n,
        hit_rate=float((rets > 0).mean()),
        avg_return_pct=float(rets.mean()),
        median_return_pct=float(np.median(rets)),
        expectancy_pct=float(rets.mean()),
        cumulative_return_pct=float(cum[-1]),
        sharpe=sharpe,
        max_drawdown_pct=float(_max_drawdown(cum)),
        win_rr=win_rr,
        n_long=sum(1 for t in trades if t.direction == "long"),
        n_short=sum(1 for t in trades if t.direction == "short"),
        n_stop=sum(1 for t in trades if t.reason == "stop"),
        n_expiry=sum(1 for t in trades if t.reason == "expiry"),
    )


def summarize_metrics(m: TradeMetrics) -> str:
    return (
        f"trades={m.n_trades}  hit_rate={m.hit_rate:.1%}  "
        f"avg={m.avg_return_pct:+.3%}  median={m.median_return_pct:+.3%}  "
        f"cum={m.cumulative_return_pct:+.2%}  sharpe={m.sharpe:.2f}  "
        f"max_dd={m.max_drawdown_pct:.2%}  win_rr={m.win_rr:.2f}  "
        f"L={m.n_long}/S={m.n_short}  stop={m.n_stop}/exp={m.n_expiry}"
    )
