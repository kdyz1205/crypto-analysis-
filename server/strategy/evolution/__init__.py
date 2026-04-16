"""Self-Evolving Trading System -- integration hooks.

This package implements a three-layer self-improvement loop:

  Level 3 (daily_report):  Daily analysis + outcome matching + drift detection
  Level 4 (trainer):       ML classifier trained on historical trades
  Level 5 (scorer):        Setup quality scoring to gate/size trades
  Level 5 (adjuster):      Auto-tune buffer, RR, TF weights, blacklist, threshold

Integration points (designed to be called from mar_bb_runner's scan loop):

  after_scan()            -- call after each scan completes
  before_trade()          -- call before placing a trade (returns go/no-go + score)
  weekly_maintenance()    -- call once per week (retrains model)
  daily_maintenance()     -- call once per day (runs adjustments)

All functions are safe to call even if:
  - No trades exist yet (early bootstrapping)
  - No model has been trained yet
  - Optional dependencies (xgboost, sklearn) are not installed
  - Telegram is not configured
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Safe imports -- each module handles its own ImportError gracefully
# ---------------------------------------------------------------------------

from .daily_report import check_daily_report, generate_daily_report, load_report, list_reports
from .scorer import score_setup, should_take_trade, get_scorer_status
from .adjuster import (
    is_coin_blacklisted,
    get_tf_weight,
    run_adjustment_cycle,
    load_config as load_evolution_config,
)
from .trainer import train_model, retrain_if_needed, get_model_info
from .features import FEATURE_NAMES, FEATURE_COUNT, extract_features

# ---------------------------------------------------------------------------
# Integration hooks
# ---------------------------------------------------------------------------

_last_daily_maintenance_date: str = ""
_last_weekly_maintenance_ts: float = 0.0
_WEEKLY_INTERVAL = 7 * 24 * 3600  # 7 days in seconds


async def after_scan() -> dict[str, Any] | None:
    """Call after each scan cycle completes.

    Checks if it's time for a daily report. Lightweight (string
    comparison) when there's nothing to do.

    Returns the daily report dict if generated, None otherwise.
    """
    try:
        return await check_daily_report()
    except Exception as e:
        print(f"[evolution] after_scan err: {e}", flush=True)
        return None


def before_trade(
    trade_params: dict[str, Any],
    ohlcv_context: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Call before placing a trade. Returns a decision dict.

    Checks:
      1. Is the coin blacklisted?
      2. What's the TF weight? (used for position sizing)
      3. Does the setup pass the ML scorer?

    Returns
    -------
    dict with keys:
      ``take_trade``: bool -- final go/no-go
      ``score``: float -- ML quality score (0.0 - 1.0)
      ``tf_weight``: float -- TF allocation weight (0.0 - 1.0)
      ``blacklisted``: bool -- coin blacklist status
      ``reason``: str -- human-readable explanation if rejected
    """
    symbol = str(trade_params.get("symbol", "")).upper()
    timeframe = str(trade_params.get("timeframe", "1h"))

    result = {
        "take_trade": True,
        "score": 0.5,
        "tf_weight": 1.0,
        "blacklisted": False,
        "reason": "",
    }

    # 1. Blacklist check
    try:
        if is_coin_blacklisted(symbol):
            result["take_trade"] = False
            result["blacklisted"] = True
            result["reason"] = f"{symbol} is blacklisted (10+ consecutive losses)"
            return result
    except Exception:
        pass

    # 2. TF weight
    try:
        tf_w = get_tf_weight(timeframe)
        result["tf_weight"] = tf_w
        if tf_w < 0.15:
            result["take_trade"] = False
            result["reason"] = f"TF {timeframe} weight too low ({tf_w:.2f})"
            return result
    except Exception:
        pass

    # 3. ML scorer
    try:
        take, score = should_take_trade(trade_params, ohlcv_context)
        result["score"] = score
        if not take:
            result["take_trade"] = False
            result["reason"] = f"Score {score:.3f} below threshold"
            return result
    except Exception:
        # If scorer fails, allow the trade (fail-open)
        result["score"] = 0.5

    return result


async def daily_maintenance() -> dict[str, Any]:
    """Run daily maintenance tasks. Call once per UTC day.

    Tasks:
      1. Generate daily report (if not already done)
      2. Run adjustment cycle

    Returns a summary dict.
    """
    global _last_daily_maintenance_date
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today == _last_daily_maintenance_date:
        return {"ok": True, "action": "skipped", "reason": "Already ran today"}

    _last_daily_maintenance_date = today
    results: dict[str, Any] = {"date": today}

    # 1. Daily report
    try:
        report = await check_daily_report()
        results["daily_report"] = "generated" if report else "already_exists"
    except Exception as e:
        results["daily_report"] = f"error: {e}"

    # 2. Adjustment cycle
    try:
        adj = run_adjustment_cycle()
        results["adjustments"] = adj
    except Exception as e:
        results["adjustments"] = {"ok": False, "error": str(e)}

    return results


def weekly_maintenance() -> dict[str, Any]:
    """Run weekly maintenance tasks. Call from scan loop.

    Tasks:
      1. Retrain ML model if enough new data

    Automatically gates on time (at most once per week).

    Returns a summary dict.
    """
    global _last_weekly_maintenance_ts
    now = time.time()

    if (now - _last_weekly_maintenance_ts) < _WEEKLY_INTERVAL:
        return {"ok": True, "action": "skipped", "reason": "Less than 7 days since last run"}

    _last_weekly_maintenance_ts = now

    try:
        result = retrain_if_needed()
        return result
    except Exception as e:
        return {"ok": False, "action": "error", "reason": str(e)}


# ---------------------------------------------------------------------------
# Status / debug
# ---------------------------------------------------------------------------

def get_evolution_status() -> dict[str, Any]:
    """Return a comprehensive status dict for the dashboard."""
    return {
        "scorer": get_scorer_status(),
        "model": get_model_info(),
        "config": load_evolution_config(),
        "recent_reports": list_reports(last_n=7),
    }


__all__ = [
    # Integration hooks
    "after_scan",
    "before_trade",
    "daily_maintenance",
    "weekly_maintenance",
    # Status
    "get_evolution_status",
    # Re-exports for direct access
    "generate_daily_report",
    "load_report",
    "list_reports",
    "score_setup",
    "should_take_trade",
    "train_model",
    "retrain_if_needed",
    "run_adjustment_cycle",
    "is_coin_blacklisted",
    "get_tf_weight",
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    "extract_features",
]
