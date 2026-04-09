"""Pure strategy core for trendline detection and signal generation."""

from .config import StrategyConfig
from .factors import (
    calculate_resistance_short_score,
    calculate_support_long_score,
)
from .pivots import detect_pivots
from .replay import ReplayResult, ReplaySnapshot, iter_replay_snapshots, replay_strategy
from .signals import (
    generate_failed_breakout_signals,
    generate_pre_limit_signals,
    generate_rejection_signals,
    generate_signals,
    prioritize_signals,
    resolve_signal_conflicts,
)
from .state_machine import (
    LineStateSnapshot,
    SignalStateSnapshot,
    advance_line_states,
    build_signal_state_snapshots,
    close_line_state,
    evaluate_line_arming,
)
from .trendlines import build_candidate_lines, detect_trendlines, select_active_lines

__all__ = [
    "LineStateSnapshot",
    "ReplayResult",
    "ReplaySnapshot",
    "SignalStateSnapshot",
    "StrategyConfig",
    "advance_line_states",
    "build_candidate_lines",
    "build_signal_state_snapshots",
    "calculate_resistance_short_score",
    "calculate_support_long_score",
    "close_line_state",
    "detect_pivots",
    "detect_trendlines",
    "evaluate_line_arming",
    "generate_failed_breakout_signals",
    "generate_pre_limit_signals",
    "generate_rejection_signals",
    "generate_signals",
    "iter_replay_snapshots",
    "prioritize_signals",
    "replay_strategy",
    "resolve_signal_conflicts",
    "select_active_lines",
]
