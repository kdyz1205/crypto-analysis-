"""Self-improving loop: draw → backtest → select → sweep → redraw.

Each round produces:
  - a population of candidate trendlines drawn by sr_patterns under
    round-specific SRParams
  - a population of per-line trade configs (buffer, rr, sl_tick_pct)
  - realised PnL for every (line, config) pair via the existing
    drawing_outcome_labeler simulator
  - a selected "winners" subset ranked by (avg_realized_r, win_rate,
    confluence_with_user_manual)
  - next-round SRParams tuned by Bayesian optimisation

Round artefacts land in:
  data/evolve_rounds/round_XX/
    config.json       — SRParams + trade configs used this round
    lines.jsonl       — all candidate TrendlineRecord rows
    outcomes.jsonl    — one row per (line, config) pair with realized_r
    selected.jsonl    — winner lines (inputs to next round's tokeniser retrain)
    summary.json      — per-round metrics (win_rate, avg_r, sharpe, n_lines, …)
"""
from .rounds import run_round, load_seeds
from .select import select_winners
from .draw import draw_lines_for_symbol
from .backtest import backtest_lines

__all__ = [
    "run_round",
    "load_seeds",
    "select_winners",
    "draw_lines_for_symbol",
    "backtest_lines",
]
