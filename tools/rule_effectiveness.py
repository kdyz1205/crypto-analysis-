"""Rule Effectiveness Tracker — live performance of decision rules.

Each decision rule (reversal_strict, breakout_trend_aligned, etc.) accumulates
real-world outcome stats. When a rule produces many winners, it gets boosted;
when it produces losers, it gets suppressed.

This is the actual feedback loop that makes the system self-improving.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data" / "rule_effectiveness"
DB_FILE = DATA_ROOT / "rules.json"


@dataclass
class RuleStats:
    """Live effectiveness record for one decision rule."""
    rule_id: str = ""
    decision_type: str = ""  # reversal | breakout | failed_breakout
    live_count: int = 0              # total live instances this rule triggered
    live_successes: int = 0          # instances that finished in profit
    live_failures: int = 0           # instances that finished in loss
    total_realized_return_atr: float = 0.0
    total_realized_drawdown_atr: float = 0.0
    best_return_atr: float = 0.0
    worst_drawdown_atr: float = 0.0
    last_updated: float = 0.0
    recent_outcomes: list[dict] = field(default_factory=list)  # last 20 outcomes
    # Rule lifecycle + trust management
    status: str = "active"           # active | warning | suppressed | retired | needs_recalibration
    trust_multiplier: float = 1.0    # applied to pattern engine confidence
    status_changed_at: float = 0.0
    status_reason: str = ""
    consecutive_losses: int = 0
    last_success_at: float = 0.0

    def update_from_outcome(self, outcome: dict) -> None:
        """Incorporate one live outcome into this rule's stats."""
        self.live_count += 1
        success = outcome.get("outcome_success", False)
        ret_atr = outcome.get("realized_return_atr", 0.0)
        dd_atr = outcome.get("realized_drawdown_atr", 0.0)
        if success:
            self.live_successes += 1
            self.consecutive_losses = 0
            self.last_success_at = outcome.get("stopped_at", time.time())
        else:
            self.live_failures += 1
            self.consecutive_losses += 1
        self.total_realized_return_atr += ret_atr
        self.total_realized_drawdown_atr += dd_atr
        self.best_return_atr = max(self.best_return_atr, ret_atr)
        self.worst_drawdown_atr = max(self.worst_drawdown_atr, dd_atr)
        self.last_updated = time.time()
        # Keep only most recent 20
        self.recent_outcomes.append({
            "instance_id": outcome.get("instance_id", ""),
            "outcome_class": outcome.get("outcome_class", ""),
            "return_atr": round(ret_atr, 2),
            "drawdown_atr": round(dd_atr, 2),
            "success": success,
            "at": outcome.get("stopped_at", time.time()),
        })
        if len(self.recent_outcomes) > 20:
            self.recent_outcomes = self.recent_outcomes[-20:]
        # Auto-update lifecycle status based on new data
        self._evaluate_status()

    def _evaluate_status(self) -> None:
        """Auto-transition status based on performance thresholds."""
        # Don't touch manually-retired rules
        if self.status == "retired":
            return

        # Not enough data yet
        if self.live_count < 5:
            self.status = "active"
            self.trust_multiplier = 1.0
            return

        ev = self.live_expected_value()
        win_rate = self.live_win_rate()

        old_status = self.status

        # Very bad: EV very negative AND enough samples AND recent losses
        if self.live_count >= 10 and ev < -1.5:
            self.status = "retired"
            self.trust_multiplier = 0.0
            self.status_reason = f"retired: live_ev={ev} (<-1.5) with {self.live_count} samples"
        elif self.live_count >= 10 and ev < -0.5:
            self.status = "suppressed"
            self.trust_multiplier = 0.4
            self.status_reason = f"suppressed: live_ev={ev} after {self.live_count} samples"
        # Warning: negative EV or many consecutive losses
        elif ev < 0 or self.consecutive_losses >= 3:
            self.status = "warning"
            self.trust_multiplier = 0.7
            self.status_reason = f"warning: ev={ev} consec_losses={self.consecutive_losses}"
        # Needs recalibration: win rate dropped from recent vs lifetime
        elif len(self.recent_outcomes) >= 10:
            recent_half = self.recent_outcomes[-10:]
            recent_wr = sum(1 for o in recent_half if o.get("success", False)) / len(recent_half)
            if recent_wr < win_rate * 0.6 and win_rate > 0.3:
                self.status = "needs_recalibration"
                self.trust_multiplier = 0.8
                self.status_reason = f"recalibrate: recent_wr={recent_wr:.2f} < lifetime_wr={win_rate:.2f}"
            else:
                self.status = "active"
                self.trust_multiplier = min(1.2, 1.0 + ev / 3)  # boost proven rules
                self.status_reason = ""
        else:
            self.status = "active"
            self.trust_multiplier = 1.0
            self.status_reason = ""

        if old_status != self.status:
            self.status_changed_at = time.time()

    def live_win_rate(self) -> float:
        return self.live_successes / max(self.live_count, 1)

    def live_avg_return(self) -> float:
        return self.total_realized_return_atr / max(self.live_count, 1)

    def live_avg_drawdown(self) -> float:
        return self.total_realized_drawdown_atr / max(self.live_count, 1)

    def live_expected_value(self) -> float:
        """Net expected value per live instance in ATR units."""
        n = max(self.live_count, 1)
        win_rate = self.live_successes / n
        avg_ret = self.total_realized_return_atr / n
        avg_dd = self.total_realized_drawdown_atr / n
        return round(win_rate * avg_ret - (1 - win_rate) * avg_dd, 2)


def _load_all() -> dict[str, RuleStats]:
    if not DB_FILE.exists():
        return {}
    try:
        data = json.loads(DB_FILE.read_text(encoding="utf-8"))
        return {k: RuleStats(**v) for k, v in data.items()}
    except Exception:
        return {}


def _save_all(rules: dict[str, RuleStats]):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    DB_FILE.write_text(
        json.dumps({k: asdict(v) for k, v in rules.items()}, indent=2, default=str),
        encoding="utf-8",
    )


def record_outcome(rule_id: str, decision_type: str, outcome: dict) -> RuleStats:
    """Record a real live outcome against a rule. Creates the rule if new."""
    if not rule_id:
        return None  # type: ignore
    rules = _load_all()
    stats = rules.get(rule_id)
    if stats is None:
        stats = RuleStats(rule_id=rule_id, decision_type=decision_type)
    stats.update_from_outcome(outcome)
    rules[rule_id] = stats
    _save_all(rules)
    return stats


def get_rule(rule_id: str) -> RuleStats | None:
    return _load_all().get(rule_id)


def list_rules() -> list[dict]:
    rules = _load_all()
    out = []
    for rid, s in rules.items():
        out.append({
            **asdict(s),
            "live_win_rate": round(s.live_win_rate(), 3),
            "live_avg_return_atr": round(s.live_avg_return(), 2),
            "live_avg_drawdown_atr": round(s.live_avg_drawdown(), 2),
            "live_expected_value": s.live_expected_value(),
        })
    out.sort(key=lambda r: (r["status"] == "retired", -r["live_count"]))
    return out


def manually_retire_rule(rule_id: str, reason: str = "manual") -> dict:
    """Manually retire a rule, preventing future use."""
    rules = _load_all()
    if rule_id not in rules:
        return {"ok": False, "error": "rule not found"}
    rules[rule_id].status = "retired"
    rules[rule_id].trust_multiplier = 0.0
    rules[rule_id].status_reason = f"manually retired: {reason}"
    rules[rule_id].status_changed_at = time.time()
    _save_all(rules)
    return {"ok": True, "rule_id": rule_id, "status": "retired"}


def manually_reactivate_rule(rule_id: str) -> dict:
    """Reactivate a retired rule."""
    rules = _load_all()
    if rule_id not in rules:
        return {"ok": False, "error": "rule not found"}
    rules[rule_id].status = "active"
    rules[rule_id].trust_multiplier = 1.0
    rules[rule_id].status_reason = ""
    rules[rule_id].consecutive_losses = 0
    rules[rule_id].status_changed_at = time.time()
    _save_all(rules)
    return {"ok": True, "rule_id": rule_id, "status": "active"}


def get_rule_status(rule_id: str) -> dict:
    """Quick status check for the decision layer."""
    s = get_rule(rule_id)
    if s is None:
        return {"status": "unknown", "trust_multiplier": 1.0, "live_count": 0}
    return {
        "status": s.status,
        "trust_multiplier": s.trust_multiplier,
        "live_count": s.live_count,
        "live_ev": s.live_expected_value(),
        "reason": s.status_reason,
    }


def get_rule_multiplier(rule_id: str, min_samples: int = 5) -> float:
    """Return a confidence multiplier for a rule based on live track record.

    Returns 1.0 by default (no adjustment).
    After min_samples, adjusts based on live vs. expected performance.
    """
    stats = get_rule(rule_id)
    if stats is None or stats.live_count < min_samples:
        return 1.0
    ev = stats.live_expected_value()
    if ev > 2.0:
        return 1.2  # boost high-performing rules
    if ev > 0.5:
        return 1.1
    if ev < -1.0:
        return 0.6  # suppress losing rules
    if ev < 0:
        return 0.8
    return 1.0
