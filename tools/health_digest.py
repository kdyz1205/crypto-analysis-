"""Closed-Loop Health Digest — daily automated summary of the research system.

Answers these questions:
1. Today's new pattern snapshots (how much research data was added)?
2. Today's new live outcomes (how much did we actually learn from real trades)?
3. Which rule has the highest live win rate?
4. Which rule has the most drift (expected EV vs actual EV)?
5. Which symbol/timeframe is degrading?
6. Anything that needs manual intervention?

This is the ONE daily check-in to "see" the system.
Call it from a scheduled task or manually.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

PATTERNS_DIR = Path(__file__).parent.parent / "data" / "patterns"
LIVE_OUTCOMES_DIR = Path(__file__).parent.parent / "data" / "live_outcomes"
LIVE_OUTCOMES_SIDECAR = PATTERNS_DIR / "_live_outcomes"
AUDIT_DIR = Path(__file__).parent.parent / "data" / "logs" / "audit"
DIGEST_DIR = Path(__file__).parent.parent / "data" / "digests"


# ── Data collectors ────────────────────────────────────────────────────

def _within_hours(timestamp: float, hours: int) -> bool:
    if not timestamp:
        return False
    return (time.time() - float(timestamp)) <= hours * 3600


def _count_new_patterns(hours: int = 24) -> dict:
    """Count pattern records added in the last N hours across all databases."""
    total = 0
    by_db: dict[str, int] = {}
    if not PATTERNS_DIR.exists():
        return {"total": 0, "by_db": {}}
    for db_file in PATTERNS_DIR.glob("*.jsonl"):
        if db_file.name.startswith("_"):
            continue
        # Fast approximation: check file mtime, if old skip
        if time.time() - db_file.stat().st_mtime > hours * 3600:
            continue
        try:
            name = db_file.stem
            for line in db_file.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                # time_position is a fraction; we also check the raw file age
                # but real "new" means time_position == 1.0 (live_scan or live)
                bucket = r.get("split_bucket", "")
                if bucket in ("live_scan", "live"):
                    total += 1
                    by_db[name] = by_db.get(name, 0) + 1
        except Exception:
            pass
    return {"total": total, "by_db": by_db}


def _list_live_outcomes_recent(hours: int = 24) -> list[dict]:
    """Load live outcomes closed within the last N hours."""
    if not LIVE_OUTCOMES_DIR.exists():
        return []
    out = []
    cutoff = time.time() - hours * 3600
    for f in LIVE_OUTCOMES_DIR.glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            if (r.get("closed_at") or 0) >= cutoff:
                out.append(r)
        except Exception:
            pass
    out.sort(key=lambda r: -(r.get("closed_at") or 0))
    return out


def _rule_performance_report(recent_outcomes: list[dict]) -> dict:
    """Build per-rule performance from recent live outcomes."""
    from .rule_effectiveness import list_rules
    all_rules = list_rules()

    # Group recent outcomes by rule
    by_rule: dict[str, list[dict]] = {}
    for o in recent_outcomes:
        rid = o.get("origin_decision_rule", "")
        if not rid:
            continue
        by_rule.setdefault(rid, []).append(o)

    report = []
    for r in all_rules:
        rid = r["rule_id"]
        recent_for_rule = by_rule.get(rid, [])
        recent_count = len(recent_for_rule)
        recent_wins = sum(1 for o in recent_for_rule if o.get("success"))
        recent_win_rate = recent_wins / max(recent_count, 1)
        recent_ev = sum(o.get("realized_return_atr", 0) for o in recent_for_rule) / max(recent_count, 1)

        report.append({
            "rule_id": rid,
            "decision_type": r["decision_type"],
            "lifetime_count": r["live_count"],
            "lifetime_win_rate": r["live_win_rate"],
            "lifetime_ev": r["live_expected_value"],
            "recent_count": recent_count,
            "recent_win_rate": round(recent_win_rate, 3),
            "recent_ev": round(recent_ev, 2),
            "drift": round(recent_ev - r["live_expected_value"], 2) if recent_count > 0 else 0,
        })

    report.sort(key=lambda r: -r["lifetime_ev"])
    return {
        "top_performers": [r for r in report if r["lifetime_ev"] > 0.5][:5],
        "degrading": [r for r in report if r["drift"] < -0.5 and r["recent_count"] >= 2][:5],
        "all": report,
    }


def _symbol_performance_report(recent_outcomes: list[dict]) -> list[dict]:
    """Group recent outcomes by symbol_timeframe."""
    grouped: dict[str, list[dict]] = {}
    for o in recent_outcomes:
        key = f"{o.get('symbol','?')}_{o.get('timeframe','?')}"
        grouped.setdefault(key, []).append(o)

    out = []
    for key, items in grouped.items():
        wins = sum(1 for o in items if o.get("success"))
        avg_ret = sum(o.get("realized_return_atr", 0) for o in items) / len(items)
        out.append({
            "symbol_timeframe": key,
            "count": len(items),
            "win_rate": round(wins / len(items), 3),
            "avg_return_atr": round(avg_ret, 2),
        })
    out.sort(key=lambda r: -r["avg_return_atr"])
    return out


def _needs_attention_alerts(rule_report: dict, outcomes: list[dict]) -> list[dict]:
    """Surface items that genuinely need human review."""
    alerts = []

    # 1. Degrading rules (recent EV much worse than lifetime)
    for r in rule_report.get("degrading", []):
        alerts.append({
            "severity": "warning",
            "type": "rule_drift",
            "rule_id": r["rule_id"],
            "message": f"规则 {r['rule_id']} live 漂移: 历史 EV {r['lifetime_ev']} → 近期 EV {r['recent_ev']}",
        })

    # 2. Rules with 0 live count but used in generation
    # (captured by checking audit for pattern_driven_draft events vs rule writebacks)
    # Skip for now — noisy

    # 3. High volume of failed outcomes
    failures = [o for o in outcomes if not o.get("success")]
    if len(failures) >= 5 and len(failures) > len(outcomes) * 0.6:
        alerts.append({
            "severity": "warning",
            "type": "high_fail_rate",
            "message": f"近期实例失败率 {len(failures)}/{len(outcomes)} = {int(len(failures)/len(outcomes)*100)}%",
        })

    # 4. Anomaly-tagged outcomes
    anom = [o for o in outcomes if not o.get("pattern_prediction_confirmed")]
    if len(anom) >= 3:
        alerts.append({
            "severity": "info",
            "type": "prediction_misses",
            "message": f"{len(anom)} 笔交易 pattern 预测未兑现 — 样本不够大前正常",
        })

    return alerts


# ── Main digest builder ────────────────────────────────────────────────

def build_digest(hours: int = 24) -> dict:
    """Build a full closed-loop health digest for the last N hours."""
    now = time.time()

    new_patterns = _count_new_patterns(hours)
    recent_outcomes = _list_live_outcomes_recent(hours)
    rule_report = _rule_performance_report(recent_outcomes)
    symbol_report = _symbol_performance_report(recent_outcomes)
    alerts = _needs_attention_alerts(rule_report, recent_outcomes)

    # Lifetime rule stats for context
    from .rule_effectiveness import list_rules
    lifetime_rules = list_rules()

    # Pattern DB sizes
    db_sizes = {}
    total_patterns = 0
    if PATTERNS_DIR.exists():
        for f in PATTERNS_DIR.glob("*.jsonl"):
            if f.name.startswith("_"):
                continue
            try:
                n = sum(1 for _ in f.open(encoding="utf-8"))
                db_sizes[f.stem] = n
                total_patterns += n
            except Exception:
                pass

    # Agent state
    agent_state = {}
    state_path = Path(__file__).parent.parent / "data" / "agent_state.json"
    if state_path.exists():
        try:
            agent_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Build digest
    digest = {
        "generated_at": now,
        "period_hours": hours,
        "period_start": now - hours * 3600,

        "research_volume": {
            "total_patterns": total_patterns,
            "pattern_dbs": len(db_sizes),
            "new_live_scans_24h": new_patterns["total"],
        },

        "trade_volume": {
            "closed_last_24h": len(recent_outcomes),
            "total_wins": sum(1 for o in recent_outcomes if o.get("success")),
            "win_rate": round(sum(1 for o in recent_outcomes if o.get("success")) / max(len(recent_outcomes), 1), 3),
            "avg_return_atr": round(sum(o.get("realized_return_atr", 0) for o in recent_outcomes) / max(len(recent_outcomes), 1), 2),
        },

        "rule_performance": rule_report,
        "symbol_performance": symbol_report,

        "lifetime_rules": lifetime_rules,

        "agent_state": {
            "worker_status": agent_state.get("worker_status", "unknown"),
            "current_generation": agent_state.get("current_generation", 0),
            "total_strategies_generated": agent_state.get("total_strategies_generated", 0),
            "total_results_produced": agent_state.get("total_results_produced", 0),
            "last_run_at": agent_state.get("last_run_at", 0),
            "last_error": agent_state.get("last_error", ""),
        },

        "alerts": alerts,
        "alert_count": len(alerts),

        "summary_line": _one_line_summary(recent_outcomes, rule_report, alerts),
    }

    return digest


def _one_line_summary(outcomes: list[dict], rule_report: dict, alerts: list[dict]) -> str:
    n = len(outcomes)
    if n == 0:
        return "过去 24h 无实例结束 — pattern 数据库持续积累中"
    wins = sum(1 for o in outcomes if o.get("success"))
    ev = sum(o.get("realized_return_atr", 0) for o in outcomes) / n
    alert_str = f"，{len(alerts)} 项需关注" if alerts else "，一切正常"
    return f"过去 24h: {n} 笔平仓, 胜率 {int(wins/n*100)}%, 平均 {ev:+.2f} ATR{alert_str}"


def save_digest(digest: dict) -> str:
    """Persist a digest to disk for historical comparison."""
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(digest.get("generated_at", time.time()))
    path = DIGEST_DIR / f"digest_{ts}.json"
    path.write_text(json.dumps(digest, indent=2, default=str), encoding="utf-8")
    return str(path)


def list_recent_digests(limit: int = 7) -> list[dict]:
    """List recent digests for trend comparison."""
    if not DIGEST_DIR.exists():
        return []
    digests = []
    for f in sorted(DIGEST_DIR.glob("digest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            digests.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return digests
