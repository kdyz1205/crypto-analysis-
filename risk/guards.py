"""Risk Guards — enforce hard rules before any live action."""

from __future__ import annotations
from . import rules


def check_deployment(capital: float, current_live_count: int, current_total_deployed: float) -> tuple[bool, str]:
    """Check if a new deployment is allowed."""
    if capital > rules.MAX_SINGLE_STRATEGY_CAPITAL:
        return False, f"Capital ${capital} exceeds max ${rules.MAX_SINGLE_STRATEGY_CAPITAL}"
    if current_live_count >= rules.MAX_CONCURRENT_LIVE:
        return False, f"Already {current_live_count} live strategies (max {rules.MAX_CONCURRENT_LIVE})"
    if current_total_deployed + capital > rules.MAX_AUTO_DEPLOY_CAPITAL:
        return False, f"Total deployed would exceed ${rules.MAX_AUTO_DEPLOY_CAPITAL}"
    return True, "ok"


def check_strategy_health(pnl_pct: float, drawdown_pct: float, consecutive_losses: int) -> tuple[bool, str]:
    """Check if a running strategy should be killed."""
    if pnl_pct < -rules.MAX_STRATEGY_LOSS_PCT:
        return False, f"Loss {pnl_pct:.1f}% exceeds max {rules.MAX_STRATEGY_LOSS_PCT}%"
    if drawdown_pct > rules.MAX_STRATEGY_DRAWDOWN_PCT:
        return False, f"Drawdown {drawdown_pct:.1f}% exceeds max {rules.MAX_STRATEGY_DRAWDOWN_PCT}%"
    if consecutive_losses >= rules.MAX_CONSECUTIVE_LOSSES:
        return False, f"Consecutive losses {consecutive_losses} >= {rules.MAX_CONSECUTIVE_LOSSES}"
    return True, "ok"


def check_factor_promotion(backtest_trades: int, backtest_score: float, test_count: int) -> tuple[bool, str]:
    """Check if a candidate factor can be promoted."""
    if backtest_trades < rules.MIN_BACKTEST_TRADES:
        return False, f"Only {backtest_trades} trades (need {rules.MIN_BACKTEST_TRADES})"
    if backtest_score < rules.MIN_BACKTEST_SCORE:
        return False, f"Score {backtest_score:.2f} < {rules.MIN_BACKTEST_SCORE}"
    if test_count < rules.CANDIDATE_TO_VALIDATED_MIN_TESTS:
        return False, f"Only {test_count} tests (need {rules.CANDIDATE_TO_VALIDATED_MIN_TESTS})"
    return True, "ok"
