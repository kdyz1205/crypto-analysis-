"""Pure strategy core for trendline detection and signal generation."""

from .config import StrategyConfig, apply_strategy_overrides
from .display_filter import (
    DisplayLineMeta,
    build_display_line_meta,
    collapse_display_invalidations,
    filter_display_touch_indices,
)
from .factors import (
    calculate_resistance_short_score,
    calculate_support_long_score,
)
from .pivots import detect_pivots
from .replay import (
    ReplayResult,
    ReplaySnapshot,
    build_latest_snapshot,
    build_tail_snapshots,
    iter_replay_snapshots,
    replay_strategy,
)
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
    "DisplayLineMeta",
    "LineStateSnapshot",
    "ReplayResult",
    "ReplaySnapshot",
    "SignalStateSnapshot",
    "StrategyConfig",
    "advance_line_states",
    "apply_strategy_overrides",
    "build_display_line_meta",
    "build_latest_snapshot",
    "build_tail_snapshots",
    "build_candidate_lines",
    "build_signal_state_snapshots",
    "calculate_resistance_short_score",
    "calculate_support_long_score",
    "close_line_state",
    "collapse_display_invalidations",
    "detect_pivots",
    "detect_trendlines",
    "evaluate_line_arming",
    "filter_display_touch_indices",
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
