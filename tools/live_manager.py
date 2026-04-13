"""Live Deployment Manager — lifecycle: leaderboard → draft → approved → instance → stopped/retired.

Three distinct objects:
1. LeaderboardEntry — research result, read-only
2. LiveDeploymentDraft — editable deployment candidate
3. LiveStrategyInstance — running/paused/stopped instance

State transitions:
  draft → pending_approval → approved → (creates instance) → running → paused → stopped → retired
"""

from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import asdict
from .types import LiveDeploymentDraft, LiveStrategyInstance, new_id
from .ranking import get_entry
from .audit import write_audit

DRAFTS_DIR = Path(__file__).parent.parent / "data" / "strategies" / "live_drafts"
INSTANCES_DIR = Path(__file__).parent.parent / "data" / "strategies" / "live_instances"


# ── Draft CRUD ──────────────────────────────────────────────────────────

def create_draft_from_leaderboard(entry_id: str, capital: float = 100.0,
                                   risk_per_trade: float = 0.01,
                                   auto_submit: bool = False,
                                   notes: str = "") -> dict:
    """Create a live deployment draft from a leaderboard entry."""
    entry = get_entry(entry_id)
    if not entry:
        return {"ok": False, "error": "leaderboard entry not found", "data": None}

    # Pull pattern lineage from the underlying strategy draft
    strategy_id = entry.get("strategy_id", "")
    strat_draft_path = Path(__file__).parent.parent / "data" / "strategies" / "drafts" / f"{strategy_id}.json"
    lineage = {}
    if strat_draft_path.exists():
        try:
            sd = json.loads(strat_draft_path.read_text(encoding="utf-8"))
            lineage = {
                "source_pattern_id": sd.get("source_pattern_id", ""),
                "decision_rule": sd.get("decision_rule", ""),
                "pattern_decision": sd.get("pattern_decision", ""),
                "pattern_ev": sd.get("pattern_ev", 0.0),
            }
        except Exception:
            pass

    draft = LiveDeploymentDraft(
        strategy_id=strategy_id,
        leaderboard_entry_id=entry_id,
        capital_allocation=capital,
        risk_per_trade=risk_per_trade,
        symbols=[entry.get("symbol", "")],
        timeframes=[entry.get("timeframe", "")],
        auto_submit=auto_submit,
        status="draft",
        notes=notes,
        source_pattern_id=lineage.get("source_pattern_id", ""),
        decision_rule=lineage.get("decision_rule", ""),
        pattern_decision=lineage.get("pattern_decision", ""),
        pattern_ev=lineage.get("pattern_ev", 0.0),
    )
    _save_draft(draft)
    write_audit("user", "live_draft_created", "deployment", draft.id, {
        "strategy_id": strategy_id,
        "leaderboard_entry_id": entry_id,
        "capital": capital,
        "source_pattern_id": lineage.get("source_pattern_id", ""),
        "decision_rule": lineage.get("decision_rule", ""),
    })
    return {"ok": True, "error": None, "data": asdict(draft)}


def list_live_drafts(status: str = None) -> dict:
    """List all live deployment drafts."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    drafts = []
    for f in DRAFTS_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if status and d.get("status") != status:
                continue
            drafts.append(d)
        except Exception:
            pass
    drafts.sort(key=lambda d: d.get("created_at", 0), reverse=True)
    return {"ok": True, "error": None, "data": drafts}


def get_live_draft(draft_id: str) -> dict:
    """Get a single live draft."""
    d = _load_draft(draft_id)
    if not d:
        return {"ok": False, "error": "draft not found", "data": None}
    return {"ok": True, "error": None, "data": d}


def update_live_draft(draft_id: str, **updates) -> dict:
    """Update draft fields (only in draft/pending_approval status)."""
    d = _load_draft(draft_id)
    if not d:
        return {"ok": False, "error": "draft not found", "data": None}
    if d.get("status") not in ("draft", "pending_approval"):
        return {"ok": False, "error": f"cannot edit draft in status '{d.get('status')}'", "data": None}
    allowed = {"capital_allocation", "risk_per_trade", "max_concurrent_positions",
               "auto_submit", "runtime_days", "notes", "symbols", "timeframes"}
    for k, v in updates.items():
        if k in allowed:
            d[k] = v
    d["updated_at"] = time.time()
    _save_draft_dict(d)
    return {"ok": True, "error": None, "data": d}


def approve_live_draft(draft_id: str) -> dict:
    """Approve a draft for deployment. Runs risk guard check."""
    from risk.guards import check_deployment

    d = _load_draft(draft_id)
    if not d:
        return {"ok": False, "error": "draft not found", "data": None}
    if d.get("status") not in ("draft", "pending_approval"):
        return {"ok": False, "error": f"cannot approve draft in status '{d.get('status')}'", "data": None}

    # Count current live
    live_instances = list_live_instances()["data"]
    running = [i for i in live_instances if i.get("running_status") == "running"]
    total_deployed = sum(i.get("allocated_capital", 0) for i in running)

    ok, reason = check_deployment(d.get("capital_allocation", 0), len(running), total_deployed)
    if not ok:
        d["status"] = "pending_approval"
        _save_draft_dict(d)
        return {"ok": False, "error": f"deployment blocked: {reason}", "data": d}

    d["status"] = "approved"
    d["approved_at"] = time.time()
    _save_draft_dict(d)
    write_audit("user", "live_draft_approved", "deployment", draft_id, {
        "capital": d.get("capital_allocation"), "strategy_id": d.get("strategy_id"),
    })
    return {"ok": True, "error": None, "data": d}


def delete_live_draft(draft_id: str) -> dict:
    """Delete a draft (only if not running)."""
    d = _load_draft(draft_id)
    if not d:
        return {"ok": False, "error": "draft not found", "data": None}
    if d.get("status") in ("running",):
        return {"ok": False, "error": "cannot delete running draft", "data": None}
    p = DRAFTS_DIR / f"{draft_id}.json"
    if p.exists():
        p.unlink()
    write_audit("user", "live_draft_deleted", "deployment", draft_id, {})
    return {"ok": True, "error": None, "data": {"deleted": draft_id}}


# ── Instance lifecycle ──────────────────────────────────────────────────

def create_instance_from_draft(draft_id: str) -> dict:
    """Create a live strategy instance from an approved draft."""
    d = _load_draft(draft_id)
    if not d:
        return {"ok": False, "error": "draft not found", "data": None}
    if d.get("status") != "approved":
        return {"ok": False, "error": f"draft must be approved first (current: {d.get('status')})", "data": None}

    # Extract symbol + timeframe for pattern writeback later
    syms = d.get("symbols", [])
    tfs = d.get("timeframes", [])

    instance = LiveStrategyInstance(
        deployment_draft_id=draft_id,
        strategy_id=d.get("strategy_id", ""),
        running_status="running",
        allocated_capital=d.get("capital_allocation", 0),
        # Propagate pattern lineage
        source_pattern_id=d.get("source_pattern_id", ""),
        decision_rule=d.get("decision_rule", ""),
        pattern_decision=d.get("pattern_decision", ""),
        pattern_ev_expected=d.get("pattern_ev", 0.0),
        symbol=syms[0] if syms else "",
        timeframe=tfs[0] if tfs else "",
    )
    _save_instance(instance)

    d["status"] = "running"
    _save_draft_dict(d)

    write_audit("system", "live_instance_created", "live_instance", instance.id, {
        "draft_id": draft_id, "strategy_id": d.get("strategy_id"),
        "capital": d.get("capital_allocation"),
        "source_pattern_id": instance.source_pattern_id,
        "decision_rule": instance.decision_rule,
    })
    return {"ok": True, "error": None, "data": asdict(instance)}


def list_live_instances(status: str = None) -> dict:
    """List all live strategy instances."""
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    instances = []
    for f in INSTANCES_DIR.glob("*.json"):
        try:
            i = json.loads(f.read_text(encoding="utf-8"))
            if status and i.get("running_status") != status:
                continue
            instances.append(i)
        except Exception:
            pass
    instances.sort(key=lambda i: i.get("started_at", 0), reverse=True)
    return {"ok": True, "error": None, "data": instances}


def get_live_instance(instance_id: str) -> dict:
    """Get a single live instance."""
    i = _load_instance(instance_id)
    if not i:
        return {"ok": False, "error": "instance not found", "data": None}
    return {"ok": True, "error": None, "data": i}


def pause_live_instance(instance_id: str) -> dict:
    """Pause a running instance."""
    return _transition_instance(instance_id, "paused", from_states=["running"])


def resume_live_instance(instance_id: str) -> dict:
    """Resume a paused instance."""
    return _transition_instance(instance_id, "running", from_states=["paused"])


def stop_live_instance(instance_id: str) -> dict:
    """Stop a running/paused instance."""
    return _transition_instance(instance_id, "stopped", from_states=["running", "paused"])


def retire_live_instance(instance_id: str) -> dict:
    """Retire a stopped instance."""
    return _transition_instance(instance_id, "retired", from_states=["stopped"])


def _transition_instance(instance_id: str, to_status: str, from_states: list[str]) -> dict:
    i = _load_instance(instance_id)
    if not i:
        return {"ok": False, "error": "instance not found", "data": None}
    current = i.get("running_status", "")
    if current not in from_states:
        return {"ok": False, "error": f"cannot transition from '{current}' to '{to_status}'", "data": None}
    old = current
    i["running_status"] = to_status
    i["last_action_at"] = time.time()

    # ═══ CLOSED LOOP: auto-writeback on stop ═══
    writeback_result = None
    if to_status == "stopped" and not i.get("outcome_written_back"):
        try:
            outcome = _compute_realized_outcome(i)
            # Persist realized outcome on the instance BEFORE writeback
            # (writeback function reads these fields)
            for k, v in outcome.items():
                i[k] = v
            i["stopped_at"] = time.time()
            # NOTE: don't set outcome_written_back=True yet — let run_full_writeback
            # do its work first, then set the flag after success

            # Delegate to the full writeback pipeline
            from .pattern_writeback import run_full_writeback
            writeback_result = run_full_writeback(i)
            if writeback_result and writeback_result.get("ok"):
                i["outcome_written_back"] = True
        except Exception as e:
            import traceback
            write_audit("system", "writeback_failed", "live_instance", instance_id, {
                "error": str(e)[:200],
                "traceback": traceback.format_exc()[-400:],
            })

    _save_instance_dict(i)

    # Also update the draft status
    draft_id = i.get("deployment_draft_id", "")
    if draft_id:
        d = _load_draft(draft_id)
        if d:
            d["status"] = to_status
            _save_draft_dict(d)

    audit_details = {"from": old, "to": to_status}
    if writeback_result:
        audit_details["writeback"] = writeback_result
    write_audit("user", f"live_instance_{to_status}", "live_instance", instance_id, audit_details)
    return {"ok": True, "error": None, "data": i}


def _compute_realized_outcome(instance: dict) -> dict:
    """Compute the realized outcome dict from a live instance's tracked state.

    Reads pattern_virtual_pnl — the simulated/pattern-engine outcome.
    realized_pnl_usd is reserved for real exchange fills and is NOT
    aggregated here (it goes through live_engine writeback instead).
    Legacy instances still on `current_pnl` are read for backward compat.
    """
    pnl = float(instance.get("pattern_virtual_pnl", instance.get("current_pnl", 0.0)))
    dd = float(instance.get("current_drawdown", 0.0))
    capital = max(float(instance.get("allocated_capital", 1.0)), 1.0)
    realized_return_pct = (pnl / capital) * 100.0
    realized_drawdown_pct = dd * 100.0 if dd < 1.0 else dd

    # Rough ATR conversion: assume 1 ATR ~ 2% of capital for 4h timeframe
    # (real conversion would use historical ATR from market data)
    atr_proxy_pct = 2.0
    realized_return_atr = realized_return_pct / atr_proxy_pct
    realized_drawdown_atr = max(realized_drawdown_pct, 0) / atr_proxy_pct

    # Determine bars_held from started_at / stopped_at
    started = instance.get("started_at", 0)
    now = time.time()
    tf = instance.get("timeframe", "4h")
    tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 3600)
    bars_held = max(int((now - started) / tf_seconds), 0) if started else 0

    # Classify outcome
    success = realized_return_pct > 0
    decision = instance.get("pattern_decision", "")
    if decision == "reversal":
        outcome_class = "bounce_success" if success else "bounce_failed"
    elif decision == "breakout":
        outcome_class = "breakout_success" if success else "breakout_failed"
    elif decision == "failed_breakout":
        outcome_class = "reclaim_success" if success else "reclaim_failed"
    else:
        outcome_class = "profit" if success else "loss"

    return {
        "realized_return_pct": round(realized_return_pct, 3),
        "realized_return_atr": round(realized_return_atr, 3),
        "realized_drawdown_pct": round(realized_drawdown_pct, 3),
        "realized_drawdown_atr": round(realized_drawdown_atr, 3),
        "bars_held": bars_held,
        "outcome_class": outcome_class,
        "outcome_success": success,
    }


def _writeback_to_pattern_engine(instance: dict, outcome: dict) -> dict:
    """Feed the realized outcome back into pattern DB and rule effectiveness tracker."""
    from .pattern_engine import writeback_trade_outcome
    from .rule_effectiveness import record_outcome as record_rule_outcome

    symbol = instance.get("symbol", "")
    timeframe = instance.get("timeframe", "4h")
    source_pattern_id = instance.get("source_pattern_id", "")
    rule_id = instance.get("decision_rule", "")
    decision_type = instance.get("pattern_decision", "")
    strategy_id = instance.get("strategy_id", "")
    instance_id = instance.get("id", "")

    result = {
        "symbol": symbol, "timeframe": timeframe,
        "source_pattern_id": source_pattern_id,
        "rule_id": rule_id,
        "decision_type": decision_type,
        "pattern_writeback": None,
        "rule_update": None,
    }

    # 1. Pattern DB writeback (only if we have a source pattern)
    if source_pattern_id and symbol:
        try:
            pattern_payload = {
                "profit_atr": outcome["realized_return_atr"],
                "max_return_atr": max(outcome["realized_return_atr"], 0),
                "max_drawdown_atr": outcome["realized_drawdown_atr"],
                "hit_target": outcome["outcome_success"],
                "hit_stop": not outcome["outcome_success"],
                "bars_in_trade": outcome["bars_held"],
            }
            wb = writeback_trade_outcome(
                symbol, timeframe, source_pattern_id, strategy_id, pattern_payload
            )
            result["pattern_writeback"] = wb
        except Exception as e:
            result["pattern_writeback"] = {"error": str(e)[:200]}

    # 2. Rule effectiveness update (always, even without pattern_id)
    if rule_id:
        try:
            rule_stats = record_rule_outcome(rule_id, decision_type, {
                **outcome,
                "instance_id": instance_id,
                "stopped_at": time.time(),
            })
            if rule_stats:
                result["rule_update"] = {
                    "rule_id": rule_id,
                    "live_count": rule_stats.live_count,
                    "live_win_rate": round(rule_stats.live_win_rate(), 3),
                    "live_ev": rule_stats.live_expected_value(),
                }
        except Exception as e:
            result["rule_update"] = {"error": str(e)[:200]}

    return result


# ── Storage helpers ─────────────────────────────────────────────────────

def _save_draft(draft: LiveDeploymentDraft):
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    (DRAFTS_DIR / f"{draft.id}.json").write_text(json.dumps(asdict(draft), indent=2), encoding="utf-8")

def _save_draft_dict(d: dict):
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    fid = d.get("id", new_id())
    (DRAFTS_DIR / f"{fid}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")

def _load_draft(draft_id: str) -> dict | None:
    p = DRAFTS_DIR / f"{draft_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def _save_instance(inst: LiveStrategyInstance):
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    (INSTANCES_DIR / f"{inst.id}.json").write_text(json.dumps(asdict(inst), indent=2), encoding="utf-8")

def _save_instance_dict(i: dict):
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    fid = i.get("id", new_id())
    (INSTANCES_DIR / f"{fid}.json").write_text(json.dumps(i, indent=2), encoding="utf-8")

def _load_instance(instance_id: str) -> dict | None:
    p = INSTANCES_DIR / f"{instance_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None
