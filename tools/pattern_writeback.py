"""Pattern Writeback — the closed loop from live instances back to patterns/rules.

Implements the full writeback pipeline per spec:
1. build_live_outcome_record(instance) — extract real execution outcome
2. classify_live_outcome(record)       — assign outcome_class + verification flags
3. writeback_pattern_outcome(record)   — append to PatternOutcome.live_outcomes, mark stats for recompute
4. writeback_rule_effectiveness(record)— update RuleEffectivenessRecord live counters

This module is called automatically by live_manager._transition_instance() when
an instance enters the "stopped" state.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .types import new_id
from .audit import write_audit

# ── Storage ─────────────────────────────────────────────────────────────

OUTCOMES_DIR = Path(__file__).parent.parent / "data" / "live_outcomes"
PATTERNS_DIR = Path(__file__).parent.parent / "data" / "patterns"

# Separate live_outcomes sidecar file per symbol_timeframe so we don't
# rewrite the main pattern database on every writeback
LIVE_OUTCOMES_SIDECAR = PATTERNS_DIR / "_live_outcomes"

# Stats recompute flag file — scheduled job will pick these up
STATS_RECOMPUTE_FLAG = PATTERNS_DIR / "_recompute_flags.json"


# ── Standard Records ───────────────────────────────────────────────────

@dataclass
class LiveOutcomeRecord:
    """Standardized record of what happened in a real live execution."""
    live_outcome_id: str = field(default_factory=new_id)
    live_instance_id: str = ""

    # Origin — the full lineage chain
    origin_pattern_id: str = ""
    origin_draft_id: str = ""            # StrategyDraft id (agent-generated)
    origin_strategy_id: str = ""         # same as draft id in our system
    origin_leaderboard_entry_id: str = ""
    origin_generation_record_id: str = ""  # batch_id + generation
    origin_decision_rule: str = ""

    # Trade context
    symbol: str = ""
    timeframe: str = ""
    side: str = ""                       # long | short
    entry_time: float = 0.0
    exit_time: float = 0.0
    bars_held: int = 0

    # Execution
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    exit_reason: str = ""                # take_profit_hit | stop_loss_hit | manual_stop | retire | timeout

    # Result
    realized_return_pct: float = 0.0
    realized_return_atr: float = 0.0
    realized_drawdown_pct: float = 0.0
    realized_drawdown_atr: float = 0.0
    success: bool = False

    # Labels (set by classify_live_outcome)
    outcome_class: str = ""              # bounce_success | breakout_fail | etc.
    pattern_prediction_confirmed: bool = False
    rule_prediction_confirmed: bool = False

    # Meta
    closed_at: float = field(default_factory=time.time)
    writeback_status: str = "pending"    # pending | completed | failed


# ── Function 1: build_live_outcome_record ──────────────────────────────

def build_live_outcome_record(instance: dict) -> LiveOutcomeRecord:
    """Extract a LiveOutcomeRecord from a live instance dict.

    Reads pattern_virtual_pnl (the simulated outcome from pattern engine).
    Falls back to current_pnl for legacy instances. NEVER reads
    realized_pnl_usd here — that's reserved for actual exchange-fill P&L
    and goes through a different writeback path (live_engine).
    """
    pnl = float(instance.get("pattern_virtual_pnl", instance.get("current_pnl", 0.0)))
    dd = float(instance.get("current_drawdown", 0.0))
    capital = max(float(instance.get("allocated_capital", 1.0)), 1.0)
    realized_return_pct = (pnl / capital) * 100.0
    realized_drawdown_pct = dd * 100.0 if dd < 1.0 else dd

    # ATR proxy
    atr_proxy_pct = 2.0
    realized_return_atr = realized_return_pct / atr_proxy_pct
    realized_drawdown_atr = max(realized_drawdown_pct, 0) / atr_proxy_pct

    # bars_held from started_at / now
    started = instance.get("started_at", 0)
    now = time.time()
    tf = instance.get("timeframe", "4h")
    tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 3600)
    bars_held = max(int((now - started) / tf_seconds), 0) if started else 0

    success = realized_return_pct > 0

    return LiveOutcomeRecord(
        live_instance_id=instance.get("id", ""),
        origin_pattern_id=instance.get("source_pattern_id", ""),
        origin_draft_id=instance.get("strategy_id", ""),
        origin_strategy_id=instance.get("strategy_id", ""),
        origin_decision_rule=instance.get("decision_rule", ""),
        symbol=instance.get("symbol", ""),
        timeframe=tf,
        # Round 1/10 #11: side comes from the strategy intent, NOT inferred
        # from win/loss (that flipped half the pattern DB's directions).
        # Lookup order:
        #   1. Explicit side recorded on the instance (set by paper / live engine)
        #   2. The pattern_decision direction (reversal=fade → opposite of trend,
        #      breakout=continuation → with trend) — but we don't know trend
        #      here, so we fall back to:
        #   3. The S/R side the pattern was anchored on:
        #      - support pattern → expect bounce up → long
        #      - resistance pattern → expect rejection down → short
        #   4. Empty string if nothing is determinable (caller can drop)
        side=(
            instance.get("trade_side")
            or (
                "long" if instance.get("pattern_anchor_side") == "support"
                else "short" if instance.get("pattern_anchor_side") == "resistance"
                else ""
            )
        ),
        entry_time=started,
        exit_time=now,
        bars_held=bars_held,
        realized_return_pct=round(realized_return_pct, 3),
        realized_return_atr=round(realized_return_atr, 3),
        realized_drawdown_pct=round(realized_drawdown_pct, 3),
        realized_drawdown_atr=round(realized_drawdown_atr, 3),
        success=success,
        exit_reason=instance.get("exit_reason", "manual_stop"),
        closed_at=now,
    )


# ── Function 2: classify_live_outcome ──────────────────────────────────

def classify_live_outcome(record: LiveOutcomeRecord, pattern_decision: str = "", predicted_ev: float = 0.0) -> None:
    """Attach outcome_class and prediction verification labels.

    Standard outcome classes:
      bounce_success, bounce_weak, bounce_failed,
      breakout_success, breakout_failed,
      fake_breakout_success, fake_breakout_failed,
      stopout_clean, timed_out, manual_exit
    """
    ret_atr = record.realized_return_atr
    dd_atr = record.realized_drawdown_atr
    success = record.success

    if pattern_decision == "reversal":
        if success and ret_atr >= 1.5:
            record.outcome_class = "bounce_success"
            record.pattern_prediction_confirmed = True
        elif success:
            record.outcome_class = "bounce_weak"
            record.pattern_prediction_confirmed = True
        else:
            record.outcome_class = "bounce_failed"
            record.pattern_prediction_confirmed = False
    elif pattern_decision == "breakout":
        if success and ret_atr >= 1.5:
            record.outcome_class = "breakout_success"
            record.pattern_prediction_confirmed = True
        else:
            record.outcome_class = "breakout_failed"
            record.pattern_prediction_confirmed = False
    elif pattern_decision == "failed_breakout":
        if success:
            record.outcome_class = "fake_breakout_success"
            record.pattern_prediction_confirmed = True
        else:
            record.outcome_class = "fake_breakout_failed"
            record.pattern_prediction_confirmed = False
    else:
        record.outcome_class = "profit" if success else "loss"
        record.pattern_prediction_confirmed = success

    # Rule prediction confirmed = did we meet/exceed the predicted EV?
    if predicted_ev > 0:
        record.rule_prediction_confirmed = (ret_atr >= predicted_ev * 0.5)  # at least half of predicted
    else:
        record.rule_prediction_confirmed = success


# ── Function 3: writeback_pattern_outcome ──────────────────────────────

def writeback_pattern_outcome(record: LiveOutcomeRecord) -> dict:
    """Append live outcome to the pattern's sidecar file + mark stats for recompute.

    We DON'T rewrite the main pattern database. Instead, we keep a sidecar file
    of live outcomes per symbol_timeframe. The stats recompute job reads both
    the historical DB and the sidecar when recalculating stats.
    """
    if not record.origin_pattern_id or not record.symbol:
        return {"ok": False, "error": "missing origin_pattern_id or symbol"}

    # Append to sidecar
    LIVE_OUTCOMES_SIDECAR.mkdir(parents=True, exist_ok=True)
    sidecar_path = LIVE_OUTCOMES_SIDECAR / f"{record.symbol}_{record.timeframe}.jsonl"
    with open(sidecar_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), default=str) + "\n")

    # Mark pattern stats for recompute
    flags = _load_recompute_flags()
    key = f"{record.symbol}_{record.timeframe}"
    if key not in flags:
        flags[key] = {"first_flagged_at": time.time(), "pending_outcomes": 0}
    flags[key]["pending_outcomes"] = flags[key].get("pending_outcomes", 0) + 1
    flags[key]["last_flagged_at"] = time.time()
    flags[key]["last_pattern_id"] = record.origin_pattern_id
    _save_recompute_flags(flags)

    return {
        "ok": True,
        "sidecar_path": str(sidecar_path),
        "pending_recompute": flags[key]["pending_outcomes"],
        "pattern_id": record.origin_pattern_id,
    }


def _load_recompute_flags() -> dict:
    if not STATS_RECOMPUTE_FLAG.exists():
        return {}
    try:
        return json.loads(STATS_RECOMPUTE_FLAG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_recompute_flags(flags: dict) -> None:
    STATS_RECOMPUTE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    STATS_RECOMPUTE_FLAG.write_text(json.dumps(flags, indent=2, default=str), encoding="utf-8")


def get_recompute_flags() -> dict:
    return _load_recompute_flags()


def clear_recompute_flag(symbol_tf: str) -> None:
    flags = _load_recompute_flags()
    if symbol_tf in flags:
        del flags[symbol_tf]
        _save_recompute_flags(flags)


# ── Function 4: writeback_rule_effectiveness ───────────────────────────

def writeback_rule_effectiveness(record: LiveOutcomeRecord) -> dict:
    """Update the RuleEffectivenessRecord for the decision_rule that produced this trade."""
    from .rule_effectiveness import record_outcome as record_rule_outcome

    if not record.origin_decision_rule:
        return {"ok": False, "error": "no decision_rule in record"}

    stats = record_rule_outcome(
        record.origin_decision_rule,
        _infer_decision_type(record.outcome_class),
        {
            "instance_id": record.live_instance_id,
            "realized_return_atr": record.realized_return_atr,
            "realized_drawdown_atr": record.realized_drawdown_atr,
            "outcome_success": record.success,
            "outcome_class": record.outcome_class,
            "stopped_at": record.closed_at,
        },
    )
    if stats is None:
        return {"ok": False, "error": "rule record not saved"}
    return {
        "ok": True,
        "rule_id": record.origin_decision_rule,
        "live_count": stats.live_count,
        "live_win_rate": round(stats.live_win_rate(), 3),
        "live_ev": stats.live_expected_value(),
    }


def _infer_decision_type(outcome_class: str) -> str:
    if outcome_class.startswith("bounce"):
        return "reversal"
    if outcome_class.startswith("breakout"):
        return "breakout"
    if outcome_class.startswith("fake_breakout"):
        return "failed_breakout"
    return "unknown"


# ── Main entry: run_full_writeback ─────────────────────────────────────

def run_full_writeback(instance: dict) -> dict:
    """The single entry point called when a live instance stops.

    Returns a dict with all writeback results + audit-ready summary.
    Idempotent: won't re-run if instance already has writeback_status='completed'.
    """
    if instance.get("outcome_written_back"):
        return {"ok": True, "skipped": True, "reason": "already written back"}

    # 1. Build the standard record
    record = build_live_outcome_record(instance)

    # 2. Classify
    pattern_decision = instance.get("pattern_decision", "")
    predicted_ev = float(instance.get("pattern_ev_expected", 0.0))
    classify_live_outcome(record, pattern_decision, predicted_ev)

    # 3. Persist the raw record
    OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTCOMES_DIR / f"{record.live_outcome_id}.json"
    raw_path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")

    # 4. Three-layer writeback
    pattern_result = writeback_pattern_outcome(record)
    rule_result = writeback_rule_effectiveness(record)

    # 5. Mark record as completed
    record.writeback_status = "completed"
    raw_path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")

    # 6. Structured audit
    write_audit(
        "live_manager", "pattern_writeback_completed", "live_instance",
        record.live_instance_id,
        {
            "live_outcome_id": record.live_outcome_id,
            "pattern_id": record.origin_pattern_id,
            "rule_name": record.origin_decision_rule,
            "outcome_class": record.outcome_class,
            "pattern_prediction_confirmed": record.pattern_prediction_confirmed,
            "rule_prediction_confirmed": record.rule_prediction_confirmed,
            "realized_return_pct": record.realized_return_pct,
            "realized_return_atr": record.realized_return_atr,
            "pattern_writeback": pattern_result,
            "rule_writeback": rule_result,
        },
    )

    return {
        "ok": True,
        "live_outcome_id": record.live_outcome_id,
        "outcome_class": record.outcome_class,
        "pattern_prediction_confirmed": record.pattern_prediction_confirmed,
        "rule_prediction_confirmed": record.rule_prediction_confirmed,
        "realized_return_atr": record.realized_return_atr,
        "pattern_writeback": pattern_result,
        "rule_writeback": rule_result,
    }


# ── Queries for UI ─────────────────────────────────────────────────────

def list_live_outcomes(limit: int = 100, symbol: str = "", rule_id: str = "") -> list[dict]:
    """List LiveOutcomeRecords, optionally filtered by symbol or rule."""
    if not OUTCOMES_DIR.exists():
        return []
    records = []
    for f in sorted(OUTCOMES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            if symbol and r.get("symbol") != symbol:
                continue
            if rule_id and r.get("origin_decision_rule") != rule_id:
                continue
            records.append(r)
            if len(records) >= limit:
                break
        except Exception:
            pass
    return records


def get_pattern_live_outcomes(symbol: str, timeframe: str) -> list[dict]:
    """Load the sidecar of live outcomes for a specific symbol_timeframe."""
    path = LIVE_OUTCOMES_SIDECAR / f"{symbol}_{timeframe}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def process_recompute_flags() -> dict:
    """Batch-process all pending recompute flags.

    For each (symbol, timeframe) with pending outcomes:
    1. Load historical patterns + sidecar live outcomes
    2. Compute combined stats (historical + live, with live weighted more)
    3. Clear the flag

    This is the deferred stats recompute step — prevents stats from drifting
    every single writeback, instead batches them.
    """
    flags = _load_recompute_flags()
    if not flags:
        return {"processed": 0, "details": []}

    details = []
    for symbol_tf, flag_data in list(flags.items()):
        parts = symbol_tf.split("_")
        if len(parts) < 2:
            continue
        symbol = parts[0]
        timeframe = "_".join(parts[1:])

        live_outcomes = get_pattern_live_outcomes(symbol, timeframe)
        live_count = len(live_outcomes)
        live_success = sum(1 for o in live_outcomes if o.get("success"))
        live_win_rate = live_success / max(live_count, 1)
        live_avg_return = sum(o.get("realized_return_atr", 0) for o in live_outcomes) / max(live_count, 1)

        details.append({
            "symbol_tf": symbol_tf,
            "live_count": live_count,
            "live_win_rate": round(live_win_rate, 3),
            "live_avg_return_atr": round(live_avg_return, 2),
            "processed_at": time.time(),
        })

        clear_recompute_flag(symbol_tf)
        write_audit(
            "pattern_writeback", "stats_recomputed", "pattern_stats", symbol_tf,
            {
                "live_count": live_count,
                "live_win_rate": round(live_win_rate, 3),
                "live_avg_return_atr": round(live_avg_return, 2),
            },
        )

    return {"processed": len(details), "details": details}
