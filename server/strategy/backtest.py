from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Mapping, Sequence

import pandas as pd

from ..drawings import augment_snapshot_with_manual_signals
from ..execution import PaperExecutionConfig, PaperExecutionEngine
from ..execution.types import PaperPosition, dataclass_to_dict
from .config import StrategyConfig
from .replay import ReplayResult, ReplaySnapshot, build_latest_snapshot
from .types import StrategySignal, ensure_candles_df


@dataclass(frozen=True, slots=True)
class BacktestBarStat:
    bar_index: int
    timestamp: Any
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_position_count: int
    open_order_count: int


@dataclass(frozen=True, slots=True)
class BacktestSignalStat:
    seen: int = 0
    approved: int = 0
    blocked: int = 0
    filled: int = 0
    closed: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0


@dataclass(frozen=True, slots=True)
class StrategyBacktestResult:
    symbol: str
    timeframe: str
    bars_processed: int
    starting_equity: float
    final_equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    return_pct: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate_pct: float
    avg_trade_pnl: float
    median_trade_pnl: float
    profit_factor: float | None
    max_drawdown_pct: float
    bars: tuple[BacktestBarStat, ...]
    closed_positions: tuple[PaperPosition, ...]
    signal_stats: Mapping[str, BacktestSignalStat]

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(asdict(self))


def run_strategy_backtest(
    candles,
    strategy_config: StrategyConfig | None = None,
    execution_config: PaperExecutionConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
    enabled_trigger_modes: Sequence[str] | None = None,
    active_directions: Mapping[str, str] | None = None,
    max_fill_history: int | None = None,
    max_closed_history: int | None = None,
    window_bars: int | None = None,
    close_open_positions_on_finish: bool = True,
) -> StrategyBacktestResult:
    cfg = strategy_config or StrategyConfig()
    exec_cfg = execution_config or PaperExecutionConfig()
    candles_df = ensure_candles_df(candles)
    if candles_df.empty:
        raise ValueError("No candles available for backtest")

    engine = PaperExecutionEngine(
        config=exec_cfg,
        max_fill_history=max_fill_history,
        max_closed_history=max_closed_history,
    )

    signal_meta_by_id: dict[str, StrategySignal] = {}
    per_signal_stat = defaultdict(_new_signal_stat)
    bar_stats: list[BacktestBarStat] = []

    for current_bar in range(len(candles_df)):
        window_start = max(0, current_bar - window_bars + 1) if window_bars else 0
        prefix = candles_df.iloc[window_start : current_bar + 1]
        snapshot = build_latest_snapshot(
            prefix,
            cfg,
            symbol=symbol,
            timeframe=timeframe,
            enabled_trigger_modes=enabled_trigger_modes,
            active_directions=active_directions,
        )
        snapshot = augment_snapshot_with_manual_signals(
            snapshot,
            prefix,
            cfg,
            symbol=symbol,
            timeframe=timeframe,
            active_directions=active_directions,
        )

        for signal in snapshot.signals:
            signal_meta_by_id.setdefault(signal.signal_id, signal)
            per_signal_stat[_signal_bucket_key(signal)]["seen"] += 1

        result = engine.step(
            symbol,
            timeframe,
            candles_df,
            ReplayResult(symbol=symbol, timeframe=timeframe, snapshots=(snapshot,)),
            bar_index=current_bar,
            snapshot_offset=current_bar,
        )

        account = result["state"].account
        bar_stats.append(
            BacktestBarStat(
                bar_index=current_bar,
                timestamp=snapshot.timestamp,
                equity=float(account.equity),
                realized_pnl=float(account.realized_pnl),
                unrealized_pnl=float(account.unrealized_pnl),
                open_position_count=int(account.open_position_count),
                open_order_count=int(account.open_order_count),
            )
        )

    if close_open_positions_on_finish:
        final_bar = len(candles_df) - 1
        final_timestamp = candles_df.iloc[final_bar]["timestamp"]
        final_close = float(candles_df.iloc[final_bar]["close"])
        engine.position_manager.force_close_positions(
            final_bar,
            final_close,
            final_timestamp,
            reason="final_mark",
        )

    closed_positions = tuple(engine.position_manager.get_recent_closed_positions(limit=None))
    for intent in engine.order_manager.get_intents():
        signal = signal_meta_by_id.get(intent.signal_id)
        if signal is None:
            continue
        bucket = _signal_bucket_key(signal)
        if intent.status == "blocked":
            per_signal_stat[bucket]["blocked"] += 1
        else:
            per_signal_stat[bucket]["approved"] += 1

    for fill in engine.order_manager.get_recent_fills(limit=None):
        signal = signal_meta_by_id.get(fill.signal_id)
        if signal is None:
            continue
        per_signal_stat[_signal_bucket_key(signal)]["filled"] += 1

    for position in closed_positions:
        signal = signal_meta_by_id.get(position.signal_id)
        if signal is None:
            continue
        bucket = _signal_bucket_key(signal)
        per_signal_stat[bucket]["closed"] += 1
        per_signal_stat[bucket]["total_pnl"] += float(position.realized_pnl)
        if position.realized_pnl > 0:
            per_signal_stat[bucket]["wins"] += 1
        elif position.realized_pnl < 0:
            per_signal_stat[bucket]["losses"] += 1

    trade_pnls = [float(position.realized_pnl) for position in closed_positions]
    win_count = sum(1 for pnl in trade_pnls if pnl > 0)
    loss_count = sum(1 for pnl in trade_pnls if pnl < 0)
    total_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    total_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
    final_state = engine.get_state()
    realized_pnl = float(final_state.account.realized_pnl)
    unrealized_pnl = float(final_state.account.unrealized_pnl)
    final_equity = float(final_state.account.equity)
    total_pnl = realized_pnl + unrealized_pnl
    max_drawdown_pct = _calculate_max_drawdown_pct(bar_stats, exec_cfg.starting_equity, final_equity)

    signal_stats = {
        key: BacktestSignalStat(**values)
        for key, values in sorted(per_signal_stat.items(), key=lambda item: item[0])
    }

    return StrategyBacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        bars_processed=len(candles_df),
        starting_equity=float(exec_cfg.starting_equity),
        final_equity=final_equity,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_pnl=float(total_pnl),
        return_pct=(float(total_pnl) / float(exec_cfg.starting_equity) * 100.0) if exec_cfg.starting_equity else 0.0,
        trade_count=len(trade_pnls),
        win_count=win_count,
        loss_count=loss_count,
        win_rate_pct=(win_count / len(trade_pnls) * 100.0) if trade_pnls else 0.0,
        avg_trade_pnl=(sum(trade_pnls) / len(trade_pnls)) if trade_pnls else 0.0,
        median_trade_pnl=(median(trade_pnls) if trade_pnls else 0.0),
        profit_factor=(total_profit / total_loss) if total_loss > 0 else (None if total_profit == 0 else float("inf")),
        max_drawdown_pct=max_drawdown_pct,
        bars=tuple(bar_stats),
        closed_positions=closed_positions,
        signal_stats=signal_stats,
    )


def summarize_backtest_results(results: Sequence[StrategyBacktestResult]) -> dict[str, Any]:
    if not results:
        return {
            "result_count": 0,
            "trade_count": 0,
            "total_pnl": 0.0,
            "return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "max_drawdown_pct": 0.0,
            "by_market": [],
            "signal_stats": {},
        }

    by_market = [
        {
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "trade_count": result.trade_count,
            "realized_pnl": result.realized_pnl,
            "unrealized_pnl": result.unrealized_pnl,
            "total_pnl": result.total_pnl,
            "return_pct": result.return_pct,
            "win_rate_pct": result.win_rate_pct,
            "profit_factor": result.profit_factor,
            "max_drawdown_pct": result.max_drawdown_pct,
        }
        for result in results
    ]

    signal_counter: dict[str, Counter] = defaultdict(Counter)
    signal_pnl: Counter = Counter()
    for result in results:
        for key, stat in result.signal_stats.items():
            signal_counter[key].update(
                {
                    "seen": stat.seen,
                    "approved": stat.approved,
                    "blocked": stat.blocked,
                    "filled": stat.filled,
                    "closed": stat.closed,
                    "wins": stat.wins,
                    "losses": stat.losses,
                }
            )
            signal_pnl[key] += stat.total_pnl

    all_trade_pnls = [
        float(position.realized_pnl)
        for result in results
        for position in result.closed_positions
    ]
    total_profit = sum(pnl for pnl in all_trade_pnls if pnl > 0)
    total_loss = abs(sum(pnl for pnl in all_trade_pnls if pnl < 0))
    total_pnl = sum(result.total_pnl for result in results)
    total_starting_equity = sum(result.starting_equity for result in results)
    total_wins = sum(result.win_count for result in results)
    total_trades = sum(result.trade_count for result in results)

    return {
        "result_count": len(results),
        "trade_count": total_trades,
        "total_pnl": float(total_pnl),
        "return_pct": (float(total_pnl) / total_starting_equity * 100.0) if total_starting_equity else 0.0,
        "win_rate_pct": (total_wins / total_trades * 100.0) if total_trades else 0.0,
        "profit_factor": (total_profit / total_loss) if total_loss > 0 else (None if total_profit == 0 else float("inf")),
        "max_drawdown_pct": max((result.max_drawdown_pct for result in results), default=0.0),
        "by_market": by_market,
        "signal_stats": {
            key: {
                **dict(counter),
                "total_pnl": float(signal_pnl[key]),
            }
            for key, counter in sorted(signal_counter.items(), key=lambda item: item[0])
        },
    }


def _calculate_max_drawdown_pct(
    bar_stats: Sequence[BacktestBarStat],
    starting_equity: float,
    final_equity: float | None = None,
) -> float:
    peak = float(starting_equity)
    max_drawdown = 0.0
    for stat in bar_stats:
        peak = max(peak, float(stat.equity))
        if peak <= 0:
            continue
        drawdown = (peak - float(stat.equity)) / peak
        max_drawdown = max(max_drawdown, drawdown)
    if final_equity is not None:
        peak = max(peak, float(final_equity))
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - float(final_equity)) / peak)
    return max_drawdown * 100.0


def _new_signal_stat() -> dict[str, float]:
    return {
        "seen": 0,
        "approved": 0,
        "blocked": 0,
        "filled": 0,
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
    }


def _signal_bucket_key(signal: StrategySignal) -> str:
    return f"{signal.source}:{signal.direction}:{signal.trigger_mode}"


__all__ = [
    "BacktestBarStat",
    "BacktestSignalStat",
    "StrategyBacktestResult",
    "run_strategy_backtest",
    "summarize_backtest_results",
]
