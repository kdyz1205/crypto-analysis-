"""Rank + select winners from a round's backtest outcomes."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .backtest import BacktestOutcome


@dataclass
class LineStats:
    line_id: str
    n_configs: int
    n_filled: int
    best_r: float
    mean_r: float
    worst_r: float
    win_rate: float       # fraction of filled configs with realized_r > 0
    mfe_max: float
    mae_min: float

    def score(self) -> float:
        """Composite ranking score. Heavier weight on best_r and win_rate,
        penalise drawdown."""
        if self.n_filled == 0:
            return -1.0
        return (
            0.5 * self.best_r
            + 0.3 * self.win_rate * 3  # scale to similar magnitude as R
            + 0.1 * self.mean_r
            - 0.1 * abs(self.mae_min)
        )


def aggregate_per_line(outcomes: list[BacktestOutcome]) -> dict[str, LineStats]:
    by_line: dict[str, list[BacktestOutcome]] = defaultdict(list)
    for o in outcomes:
        by_line[o.line_id].append(o)
    stats: dict[str, LineStats] = {}
    for lid, rows in by_line.items():
        filled = [r for r in rows if r.filled]
        if not filled:
            stats[lid] = LineStats(lid, len(rows), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        rs = [r.realized_r for r in filled]
        mfes = [r.mfe_r for r in filled]
        maes = [r.mae_r for r in filled]
        stats[lid] = LineStats(
            line_id=lid,
            n_configs=len(rows),
            n_filled=len(filled),
            best_r=max(rs),
            mean_r=sum(rs) / len(rs),
            worst_r=min(rs),
            win_rate=sum(1 for r in rs if r > 0) / len(rs),
            mfe_max=max(mfes) if mfes else 0.0,
            mae_min=min(maes) if maes else 0.0,
        )
    return stats


def select_winners(
    outcomes: list[BacktestOutcome],
    *,
    top_frac: float = 0.2,
    min_score: float = 0.3,
) -> tuple[list[str], dict[str, LineStats]]:
    stats = aggregate_per_line(outcomes)
    ranked = sorted(stats.values(), key=lambda s: s.score(), reverse=True)
    target_n = max(1, int(len(ranked) * top_frac))
    winners = [s.line_id for s in ranked[:target_n] if s.score() >= min_score]
    return winners, stats


def summarize_round(outcomes: list[BacktestOutcome], stats: dict) -> dict:
    filled_stats = [s for s in stats.values() if s.n_filled > 0]
    if not filled_stats:
        return {"n_lines": len(stats), "n_filled_lines": 0}
    return {
        "n_lines": len(stats),
        "n_filled_lines": len(filled_stats),
        "best_r_median": sorted(s.best_r for s in filled_stats)[len(filled_stats)//2],
        "mean_r_avg": sum(s.mean_r for s in filled_stats) / len(filled_stats),
        "win_rate_avg": sum(s.win_rate for s in filled_stats) / len(filled_stats),
        "n_outcomes_total": len(outcomes),
    }
