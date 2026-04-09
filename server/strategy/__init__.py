"""Pure strategy core for trendline detection and signal generation."""

from .config import StrategyConfig
from .factors import (
    calculate_resistance_short_score,
    calculate_support_long_score,
)
from .pivots import detect_pivots
from .signals import (
    generate_failed_breakout_signals,
    generate_pre_limit_signals,
    generate_rejection_signals,
    generate_signals,
    prioritize_signals,
    resolve_signal_conflicts,
)
from .trendlines import build_candidate_lines, detect_trendlines, select_active_lines

__all__ = [
    "StrategyConfig",
    "build_candidate_lines",
    "calculate_resistance_short_score",
    "calculate_support_long_score",
    "detect_pivots",
    "detect_trendlines",
    "generate_failed_breakout_signals",
    "generate_pre_limit_signals",
    "generate_rejection_signals",
    "generate_signals",
    "prioritize_signals",
    "resolve_signal_conflicts",
    "select_active_lines",
]
