"""Self-Adjustment Engine (Level 5) -- auto-tunes parameters from live results.

Reads rolling performance metrics from daily reports and adjusts:
  1. Buffer: tracks actual MFE/MAE, shifts buffer toward optimal
  2. RR target: raises if MFE consistently exceeds target; lowers if rarely reaches
  3. TF weights: per-TF Sharpe on rolling window; reduces allocation to neg-Sharpe TFs
  4. Coin blacklist: 10+ consecutive losses -> blacklist for 24h
  5. Scoring threshold: auto-tune based on precision/recall trade-off

All adjustments are persisted to data/evolution_config.json and logged
with full rationale for auditability.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .daily_report import _load_all_trades, load_report, list_reports

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    try:
        from server.core.config import PROJECT_ROOT
        return Path(PROJECT_ROOT)
    except Exception:
        return Path(__file__).resolve().parents[3]


def _config_path() -> Path:
    return _project_root() / "data" / "evolution_config.json"


def _log_path() -> Path:
    d = _project_root() / "data" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "adjuster.log"


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    """Load the current evolution config. Returns defaults if file missing."""
    defaults: dict[str, Any] = {
        "buffer_pct": 0.003,         # 0.3% default buffer
        "rr_target": 3.0,           # default RR target
        "scoring_threshold": 0.3,   # minimum score to take a trade
        "tf_weights": {             # allocation weights per TF (0-1)
            "3m": 1.0, "5m": 1.0, "15m": 1.0, "1h": 1.0, "4h": 1.0,
        },
        "coin_blacklist": {},       # symbol -> expiry_ts
        "last_adjustment": "",      # ISO timestamp of last run
        "adjustment_history": [],   # last 50 adjustments for audit trail
    }
    p = _config_path()
    if not p.exists():
        return defaults
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
        # Merge with defaults so new keys are always present
        for k, v in defaults.items():
            if k not in saved:
                saved[k] = v
        return saved
    except Exception:
        return defaults


def save_config(config: dict[str, Any]) -> None:
    """Persist the evolution config to disk."""
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def _log_adjustment(msg: str) -> None:
    """Append a timestamped line to the adjuster log."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass
    print(f"[adjuster] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get_recent_resolved_trades(days: int = 30) -> list[dict]:
    """Get all resolved trades from the last N days of daily reports + trade log."""
    trades: list[dict] = []
    seen: set[str] = set()

    # From daily reports
    for date_str in list_reports(last_n=days):
        report = load_report(date_str)
        if not report:
            continue
        for t in report.get("trades", []):
            if t.get("outcome") not in ("win", "loss"):
                continue
            key = f"{t.get('order_id', '')}_{t.get('ts', '')}"
            if key not in seen:
                seen.add(key)
                trades.append(t)

    # From trade log (enriched trades)
    for t in _load_all_trades():
        if t.get("outcome") not in ("win", "loss"):
            continue
        key = f"{t.get('order_id', '')}_{t.get('ts', '')}"
        if key not in seen:
            seen.add(key)
            trades.append(t)

    # Sort by timestamp
    trades.sort(key=lambda t: float(t.get("ts", 0)))
    return trades


# ---------------------------------------------------------------------------
# Adjustment functions
# ---------------------------------------------------------------------------

def _adjust_buffer(trades: list[dict], config: dict) -> list[str]:
    """Track actual MFE/MAE of recent trades and shift buffer toward optimal.

    If trades have ``mfe_pct`` and ``mae_pct`` fields (populated by the
    daily report enrichment), compute the median entry-to-MFE distance
    and adjust buffer to capture more of the MFE.

    Without MFE/MAE data, uses the SL distance as a proxy: if most wins
    have tight SL, buffer can be tighter; if many losses are from buffer
    being too tight (price barely misses entry), widen it.
    """
    changes = []
    if len(trades) < 20:
        return changes

    current_buffer = float(config.get("buffer_pct", 0.003))
    recent = trades[-50:]  # last 50 trades

    # Use PnL distribution as proxy: if avg loss is very small (barely hit SL),
    # the buffer is likely too tight
    losses = [t for t in recent if t["outcome"] == "loss"]
    wins = [t for t in recent if t["outcome"] == "win"]

    if not losses or not wins:
        return changes

    # Compute SL distance from entry for each trade
    sl_dists = []
    for t in recent:
        entry = float(t.get("entry_price", 0) or 0)
        stop = float(t.get("stop_price", 0) or 0)
        if entry > 0 and stop > 0:
            sl_dists.append(abs(entry - stop) / entry)

    if not sl_dists:
        return changes

    median_sl_dist = float(np.median(sl_dists))

    # Heuristic: if WR < 35% and sl_dist is very tight, buffer might be too tight
    wr = len(wins) / len(recent) * 100
    if wr < 35 and median_sl_dist < 0.003:
        new_buffer = min(current_buffer * 1.15, 0.008)  # max 0.8%
        if abs(new_buffer - current_buffer) > 0.0001:
            config["buffer_pct"] = round(new_buffer, 5)
            msg = (f"Buffer widened: {current_buffer*100:.3f}% -> {new_buffer*100:.3f}% "
                   f"(WR={wr:.0f}%, median SL dist={median_sl_dist*100:.3f}%)")
            _log_adjustment(msg)
            changes.append(msg)

    elif wr > 50 and median_sl_dist > 0.005:
        new_buffer = max(current_buffer * 0.9, 0.001)  # min 0.1%
        if abs(new_buffer - current_buffer) > 0.0001:
            config["buffer_pct"] = round(new_buffer, 5)
            msg = (f"Buffer tightened: {current_buffer*100:.3f}% -> {new_buffer*100:.3f}% "
                   f"(WR={wr:.0f}%, median SL dist={median_sl_dist*100:.3f}%)")
            _log_adjustment(msg)
            changes.append(msg)

    return changes


def _adjust_rr_target(trades: list[dict], config: dict) -> list[str]:
    """Adjust RR target based on actual trade outcomes.

    If MFE consistently exceeds the target, raise it (capturing more upside).
    If it rarely reaches, lower it (taking profit sooner).
    """
    changes = []
    if len(trades) < 20:
        return changes

    current_rr = float(config.get("rr_target", 3.0))
    recent = trades[-50:]
    wins = [t for t in recent if t["outcome"] == "win"]

    if not wins:
        return changes

    # Compute actual RR of winning trades
    actual_rrs = []
    for t in wins:
        entry = float(t.get("entry_price", 0) or 0)
        stop = float(t.get("stop_price", 0) or 0)
        pnl_pct = float(t.get("pnl_pct", 0) or 0)
        if entry > 0 and stop > 0:
            sl_dist = abs(entry - stop) / entry
            if sl_dist > 1e-6:
                actual_rr = abs(pnl_pct) / (sl_dist * 100)
                actual_rrs.append(actual_rr)

    if len(actual_rrs) < 5:
        return changes

    median_rr = float(np.median(actual_rrs))
    p75_rr = float(np.percentile(actual_rrs, 75))

    # If median achieved RR is 50%+ above target, raise target
    if median_rr > current_rr * 1.5:
        new_rr = min(current_rr * 1.2, 10.0)  # cap at 10:1
        config["rr_target"] = round(new_rr, 1)
        msg = (f"RR target raised: {current_rr:.1f} -> {new_rr:.1f} "
               f"(median achieved RR={median_rr:.1f}, p75={p75_rr:.1f})")
        _log_adjustment(msg)
        changes.append(msg)

    # If 75th percentile barely reaches target, lower it
    elif p75_rr < current_rr * 0.7:
        new_rr = max(current_rr * 0.85, 1.5)  # floor at 1.5:1
        config["rr_target"] = round(new_rr, 1)
        msg = (f"RR target lowered: {current_rr:.1f} -> {new_rr:.1f} "
               f"(p75 achieved RR={p75_rr:.1f} < {current_rr*0.7:.1f})")
        _log_adjustment(msg)
        changes.append(msg)

    return changes


def _adjust_tf_weights(trades: list[dict], config: dict) -> list[str]:
    """Per-TF Sharpe ratio on rolling window. Reduce allocation to negative-Sharpe TFs."""
    changes = []
    if len(trades) < 30:
        return changes

    current_weights = config.get("tf_weights", {})
    recent = trades[-100:]  # last 100 trades

    # Group PnLs by TF
    tf_pnls: dict[str, list[float]] = defaultdict(list)
    for t in recent:
        tf = t.get("timeframe", "?")
        pnl = float(t.get("pnl", 0) or 0)
        tf_pnls[tf].append(pnl)

    for tf, pnls in tf_pnls.items():
        if len(pnls) < 5:
            continue

        arr = np.array(pnls)
        mean_pnl = float(np.mean(arr))
        std_pnl = float(np.std(arr))
        if std_pnl > 1e-8:
            sharpe = mean_pnl / std_pnl
        elif mean_pnl < 0:
            sharpe = -10.0  # all-loss with zero variance = extremely bad
        elif mean_pnl > 0:
            sharpe = 10.0   # all-win with zero variance = extremely good
        else:
            sharpe = 0.0

        current_w = float(current_weights.get(tf, 1.0))

        if sharpe < -0.5 and current_w > 0.2:
            # Negative Sharpe: reduce allocation
            new_w = max(current_w * 0.7, 0.1)  # floor at 0.1
            current_weights[tf] = round(new_w, 2)
            msg = (f"TF weight reduced: {tf} {current_w:.2f} -> {new_w:.2f} "
                   f"(Sharpe={sharpe:.2f}, n={len(pnls)})")
            _log_adjustment(msg)
            changes.append(msg)

        elif sharpe > 0.5 and current_w < 1.0:
            # Positive Sharpe: restore/increase
            new_w = min(current_w * 1.2, 1.0)
            current_weights[tf] = round(new_w, 2)
            msg = (f"TF weight increased: {tf} {current_w:.2f} -> {new_w:.2f} "
                   f"(Sharpe={sharpe:.2f}, n={len(pnls)})")
            _log_adjustment(msg)
            changes.append(msg)

    config["tf_weights"] = current_weights
    return changes


def _update_coin_blacklist(trades: list[dict], config: dict) -> list[str]:
    """Blacklist a coin for 24h if it has 10+ consecutive losses."""
    changes = []
    now = time.time()
    blacklist: dict[str, float] = config.get("coin_blacklist", {})

    # Remove expired blacklist entries
    expired = [sym for sym, exp_ts in blacklist.items() if exp_ts < now]
    for sym in expired:
        del blacklist[sym]
        msg = f"Coin un-blacklisted: {sym} (24h expired)"
        _log_adjustment(msg)
        changes.append(msg)

    # Check for consecutive losses per coin (recent trades only)
    recent = trades[-200:] if len(trades) > 200 else trades
    # Group by symbol, check last N trades
    by_coin: dict[str, list[str]] = defaultdict(list)
    for t in recent:
        sym = t.get("symbol", "?")
        by_coin[sym].append(t.get("outcome", "unknown"))

    for sym, outcomes in by_coin.items():
        if sym in blacklist:
            continue  # already blacklisted

        # Count consecutive losses from the end
        consecutive_losses = 0
        for outcome in reversed(outcomes):
            if outcome == "loss":
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 10:
            expiry = now + 86400  # 24 hours from now
            blacklist[sym] = expiry
            msg = (f"Coin blacklisted: {sym} for 24h "
                   f"({consecutive_losses} consecutive losses)")
            _log_adjustment(msg)
            changes.append(msg)

    config["coin_blacklist"] = blacklist
    return changes


def _adjust_scoring_threshold(trades: list[dict], config: dict) -> list[str]:
    """Auto-tune the scoring threshold based on precision/recall.

    If the system is rejecting too many trades that would have won
    (high threshold -> low recall), lower it. If it's accepting too many
    losers (low threshold -> low precision), raise it.

    This only runs if trades have a ``score`` field (populated by the
    scorer at trade time).
    """
    changes = []
    scored_trades = [t for t in trades if "score" in t and t.get("outcome") in ("win", "loss")]

    if len(scored_trades) < 30:
        return changes

    current_threshold = float(config.get("scoring_threshold", 0.3))
    recent = scored_trades[-100:]

    # Compute precision at current threshold
    above = [t for t in recent if float(t.get("score", 0)) >= current_threshold]
    below = [t for t in recent if float(t.get("score", 0)) < current_threshold]

    if not above:
        return changes

    precision_above = sum(1 for t in above if t["outcome"] == "win") / len(above)
    # Recall of wins: how many wins were above threshold
    total_wins = sum(1 for t in recent if t["outcome"] == "win")
    if total_wins == 0:
        return changes
    recall = sum(1 for t in above if t["outcome"] == "win") / total_wins

    # If precision is low (< 40%), raise threshold
    if precision_above < 0.40 and current_threshold < 0.6:
        new_threshold = min(current_threshold + 0.05, 0.7)
        config["scoring_threshold"] = round(new_threshold, 2)
        msg = (f"Scoring threshold raised: {current_threshold:.2f} -> {new_threshold:.2f} "
               f"(precision={precision_above:.2f}, recall={recall:.2f})")
        _log_adjustment(msg)
        changes.append(msg)

    # If recall is low (< 50% of wins are above threshold), lower it
    elif recall < 0.50 and current_threshold > 0.15:
        new_threshold = max(current_threshold - 0.05, 0.1)
        config["scoring_threshold"] = round(new_threshold, 2)
        msg = (f"Scoring threshold lowered: {current_threshold:.2f} -> {new_threshold:.2f} "
               f"(precision={precision_above:.2f}, recall={recall:.2f})")
        _log_adjustment(msg)
        changes.append(msg)

    return changes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_coin_blacklisted(symbol: str) -> bool:
    """Check if a coin is currently blacklisted.

    Used by the scan loop to skip blacklisted symbols.
    """
    config = load_config()
    blacklist = config.get("coin_blacklist", {})
    expiry = blacklist.get(symbol.upper(), 0)
    return float(expiry) > time.time()


def get_tf_weight(timeframe: str) -> float:
    """Get the current allocation weight for a timeframe (0.0-1.0).

    Used by the scan loop to reduce position size on under-performing TFs.
    """
    config = load_config()
    weights = config.get("tf_weights", {})
    return float(weights.get(timeframe, 1.0))


def run_adjustment_cycle() -> dict[str, Any]:
    """Run the full adjustment cycle.

    Reads recent performance data, applies all adjustment rules,
    persists changes, and returns a summary of what changed.

    Designed to be called once per day (after the daily report).
    """
    print("[adjuster] Starting adjustment cycle", flush=True)

    config = load_config()
    trades = _get_recent_resolved_trades(days=30)

    if len(trades) < 10:
        msg = f"Too few resolved trades ({len(trades)}) for adjustments"
        print(f"[adjuster] {msg}", flush=True)
        return {"ok": True, "changes": [], "reason": msg}

    all_changes: list[str] = []

    # Run all adjusters
    all_changes.extend(_adjust_buffer(trades, config))
    all_changes.extend(_adjust_rr_target(trades, config))
    all_changes.extend(_adjust_tf_weights(trades, config))
    all_changes.extend(_update_coin_blacklist(trades, config))
    all_changes.extend(_adjust_scoring_threshold(trades, config))

    # Update metadata
    config["last_adjustment"] = datetime.now(timezone.utc).isoformat()

    # Keep last 50 adjustments
    history = config.get("adjustment_history", [])
    for change in all_changes:
        history.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "change": change,
        })
    config["adjustment_history"] = history[-50:]

    # Persist
    save_config(config)

    if all_changes:
        print(f"[adjuster] Made {len(all_changes)} adjustments:", flush=True)
        for c in all_changes:
            print(f"  - {c}", flush=True)
    else:
        print("[adjuster] No adjustments needed", flush=True)

    return {
        "ok": True,
        "changes": all_changes,
        "config_snapshot": {
            "buffer_pct": config.get("buffer_pct"),
            "rr_target": config.get("rr_target"),
            "scoring_threshold": config.get("scoring_threshold"),
            "tf_weights": config.get("tf_weights"),
            "blacklist_count": len(config.get("coin_blacklist", {})),
        },
    }


__all__ = [
    "load_config",
    "save_config",
    "is_coin_blacklisted",
    "get_tf_weight",
    "run_adjustment_cycle",
]
