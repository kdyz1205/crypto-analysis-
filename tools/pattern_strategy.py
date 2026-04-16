"""Pattern → Strategy Decision Rules.

Core idea: Don't let pattern stats just display as probabilities.
Use them to auto-generate full StrategyConfig drafts based on decision rules.

The decision layer itself is auditable — each rule trigger is logged with the
pattern that caused it, so the rules themselves can be backtested and refined.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import (
    StrategyConfig, MarketScope, ConditionRule, EntryRules, ExitRules, RiskProfile
)


# ── Decision Rules ──────────────────────────────────────────────────────

DecisionType = str  # "reversal" | "breakout" | "failed_breakout" | "no_trade" | "watch_only"


def decide_strategy_type(match_result: dict) -> dict:
    """Given the output of pattern_engine.match_pattern, return decision + reason.

    Returns:
      {
        "decision": "reversal" | "breakout" | "failed_breakout" | "no_trade" | "watch_only",
        "rule_id": "<specific rule that fired>",
        "reason": "<explanation>",
        "triggered_rules": [...],
        "pattern_summary": {...},
        "rule_trust_multiplier": 1.0,  # from rule_effectiveness
      }

    If a rule would fire but has been suppressed/retired via rule_effectiveness,
    it is skipped and the next rule is evaluated.
    """
    from .rule_effectiveness import get_rule_status

    stats = match_result.get("stats", {})
    anomaly = match_result.get("anomaly", {})
    features = match_result.get("current_features", {})
    cluster = match_result.get("cluster_context") or {}

    def rule_is_usable(rule_id: str) -> tuple[bool, dict]:
        """Returns (usable, status_info). Suppressed/retired rules are NOT usable."""
        s = get_rule_status(rule_id)
        if s["status"] in ("retired", "suppressed"):
            return False, s
        return True, s

    # Extract key fields
    p_bounce = stats.get("p_bounce", 0.0)
    p_break = stats.get("p_break", 0.0)
    p_fake_break = stats.get("p_fake_break", 0.0)
    ev = stats.get("expected_value", 0.0)
    confidence = stats.get("confidence", 0.0)
    stability = stats.get("overfit_flag", "insufficient_samples")
    sample_size = stats.get("sample_size", 0)
    is_anomaly = anomaly.get("is_anomaly", False)
    side = features.get("side", "")
    trend = features.get("trend_context", "range")

    triggered = []

    # Compute spread: how much bounce probability exceeds break probability
    bounce_break_spread = p_bounce - p_break

    # EV override: very high EV can relax some conditions
    high_ev = ev >= 3.0
    strong_ev = ev >= 1.5

    # ── Hard filters (checked first) ──────────────────────────
    if is_anomaly:
        return _no_trade_result("anomalous structure — no comparable historical data",
                                stats, features, anomaly, cluster, ["anomaly_detected"])

    if sample_size < 15:
        return _no_trade_result(f"insufficient samples ({sample_size} < 15)",
                                stats, features, anomaly, cluster, ["sample_size_low"])

    if confidence < 0.45:
        return _no_trade_result(f"confidence too low ({confidence:.2f} < 0.45)",
                                stats, features, anomaly, cluster, ["low_confidence"])

    if stability == "overfit_detected":
        return _no_trade_result("train/test divergence — pattern may be overfit",
                                stats, features, anomaly, cluster, ["overfit"])

    if ev <= 0:
        return _no_trade_result(f"negative expected value (EV={ev:.2f} ATR)",
                                stats, features, anomaly, cluster, ["negative_ev"])

    # ── Decision tree (ordered by specificity) ────────────────
    #
    # Priority 1: Strong reversal (spread + EV based, not hard p_break cap)
    # - Traditional: p_bounce >= 60% AND p_break <= 50%
    # - Relaxed: spread >= 15% AND EV >= 1.5 ATR (lets SOL-1h-style trades through)
    # - High EV: spread >= 10% AND EV >= 3.0 ATR (lets very profitable trades through)
    strict_reversal = p_bounce >= 0.60 and p_break <= 0.50 and confidence >= 0.55
    spread_reversal = bounce_break_spread >= 0.15 and strong_ev and p_bounce >= 0.65
    high_ev_reversal = bounce_break_spread >= 0.10 and high_ev and p_bounce >= 0.70

    suppressed_rules = []

    if strict_reversal or spread_reversal or high_ev_reversal:
        if strict_reversal:
            rule_id = "reversal_strict"
        elif spread_reversal:
            rule_id = "reversal_spread"
        else:
            rule_id = "reversal_high_ev"

        usable, rstatus = rule_is_usable(rule_id)
        if usable:
            mode = rule_id.replace("reversal_", "")
            triggered.append(f"reversal_mode={mode}")
            triggered.append(f"p_bounce={p_bounce:.2f}, p_break={p_break:.2f}, spread={bounce_break_spread:.2f}, EV={ev:.2f}")
            return {
                "decision": "reversal",
                "rule_id": rule_id,
                "reason": f"[{mode}] bounce prob {p_bounce:.0%} dominates break {p_break:.0%} (spread {bounce_break_spread:+.0%}), EV={ev:.2f} ATR",
                "triggered_rules": triggered,
                "rule_trust_multiplier": rstatus.get("trust_multiplier", 1.0),
                "rule_status": rstatus.get("status", "active"),
                "pattern_summary": _summarize(stats, features, anomaly, cluster),
            }
        else:
            suppressed_rules.append(rule_id)

    # Priority 2: Failed Breakout (elevated fake-break rate)
    if p_fake_break >= 0.15 and 0.30 <= p_break <= 0.70:
        usable, rstatus = rule_is_usable("failed_breakout_elevated")
        if usable:
            triggered.append(f"p_fake_break={p_fake_break:.2f}>=0.15")
            triggered.append(f"p_break in [0.30, 0.70]")
            return {
                "decision": "failed_breakout",
                "rule_id": "failed_breakout_elevated",
                "reason": f"elevated fake-break rate ({p_fake_break:.0%}) with uncertain break ({p_break:.0%})",
                "triggered_rules": triggered,
                "rule_trust_multiplier": rstatus.get("trust_multiplier", 1.0),
                "rule_status": rstatus.get("status", "active"),
                "pattern_summary": _summarize(stats, features, anomaly, cluster),
            }
        else:
            suppressed_rules.append("failed_breakout_elevated")

    # Priority 3: Breakout (directional trend confirmation)
    if p_break >= 0.55 and trend in ("uptrend", "downtrend") and confidence >= 0.55:
        trend_aligns = (
            (side == "support" and trend == "downtrend") or
            (side == "resistance" and trend == "uptrend")
        )
        if trend_aligns:
            usable, rstatus = rule_is_usable("breakout_trend_aligned")
            if usable:
                triggered.append(f"p_break={p_break:.2f}>=0.55, trend_aligned={trend}")
                return {
                    "decision": "breakout",
                    "rule_id": "breakout_trend_aligned",
                    "reason": f"breakout likely ({p_break:.0%}) and trend-aligned ({trend})",
                    "triggered_rules": triggered,
                    "rule_trust_multiplier": rstatus.get("trust_multiplier", 1.0),
                    "rule_status": rstatus.get("status", "active"),
                    "pattern_summary": _summarize(stats, features, anomaly, cluster),
                }
            else:
                suppressed_rules.append("breakout_trend_aligned")

    # If we have suppressed rules, we hit watch_only with a note
    if suppressed_rules:
        return {
            "decision": "watch_only",
            "rule_id": "all_matching_rules_suppressed",
            "reason": f"所有匹配规则已被抑制/退役: {', '.join(suppressed_rules)} — 历史实盘表现不佳",
            "triggered_rules": [f"suppressed: {r}" for r in suppressed_rules],
            "pattern_summary": _summarize(stats, features, anomaly, cluster),
        }

    # Priority 4: High EV watch with signal (even without directional clarity)
    if high_ev:
        return {
            "decision": "watch_only",
            "rule_id": "watch_high_ev_ambiguous",
            "reason": f"very high EV ({ev:.2f} ATR) but no clear directional rule matched — manual review",
            "triggered_rules": ["high_ev_ambiguous"],
            "pattern_summary": _summarize(stats, features, anomaly, cluster),
        }

    # Default: watch only (has positive EV but mixed signals)
    return {
        "decision": "watch_only",
        "rule_id": "watch_ambiguous",
        "reason": f"EV positive ({ev:.2f}) but signals mixed (bounce={p_bounce:.0%}, break={p_break:.0%}, spread={bounce_break_spread:+.0%})",
        "triggered_rules": ["ambiguous_signals"],
        "pattern_summary": _summarize(stats, features, anomaly, cluster),
    }


def _no_trade_result(reason: str, stats, features, anomaly, cluster, rules) -> dict:
    # First rule in list becomes the rule_id (most specific trigger)
    rule_id = f"no_trade_{rules[0]}" if rules else "no_trade_unknown"
    return {
        "decision": "no_trade",
        "rule_id": rule_id,
        "reason": reason,
        "triggered_rules": rules,
        "pattern_summary": _summarize(stats, features, anomaly, cluster),
    }


def _summarize(stats, features, anomaly, cluster) -> dict:
    return {
        "p_bounce": stats.get("p_bounce"),
        "p_break": stats.get("p_break"),
        "p_fake_break": stats.get("p_fake_break"),
        "expected_value": stats.get("expected_value"),
        "confidence": stats.get("confidence"),
        "stability": stats.get("overfit_flag"),
        "sample_size": stats.get("sample_size"),
        "side": features.get("side"),
        "trend_context": features.get("trend_context"),
        "is_anomaly": anomaly.get("is_anomaly"),
        "cluster_label": cluster.get("label") if cluster else None,
    }


# ── StrategyConfig Generation ───────────────────────────────────────────

def generate_strategy_configs(decision: dict, match_result: dict,
                              symbol: str, timeframe: str,
                              both_variants: bool = True) -> list[dict]:
    """Given a decision + pattern match, auto-generate StrategyConfig draft(s).

    If both_variants=True, generates a conservative AND aggressive version
    (different entry modes, different risk levels).

    Returns list of configs as dicts (serializable).
    """
    dec_type = decision.get("decision")
    if dec_type in ("no_trade", "watch_only"):
        return []

    stats = match_result.get("stats", {})
    features = match_result.get("current_features", {})
    confidence = stats.get("confidence", 0.5)
    ev = stats.get("expected_value", 0.0)
    avg_ret = stats.get("avg_return_atr", 1.0)
    avg_dd = stats.get("avg_drawdown_atr", 1.0)
    p_bounce = stats.get("p_bounce", 0.0)
    side = features.get("side", "support")
    trend = features.get("trend_context", "range")

    # RR target: derived from historical reward profile
    # Use ratio of max_return to stop_loss (not worst-case drawdown)
    # This is the actual RR if you enter and exit at structure with a stop
    realistic_stop = 1.5
    rr_base = avg_ret / realistic_stop
    # Minimum 1.2 (tight but tradeable), max 2x historical ratio
    rr_target = max(1.2, min(rr_base, rr_base * 1.5))
    rr_target = round(rr_target, 2)

    # Risk per trade: scale by confidence + stability
    risk_base = 0.01  # 1% default
    stability = stats.get("overfit_flag", "stable")
    if stability == "high_variance":
        risk_base *= 0.5
    if confidence < 0.6:
        risk_base *= 0.7
    if ev < 0.5:
        risk_base *= 0.6
    risk_base = round(max(risk_base, 0.002), 4)  # floor 0.2%

    # Build base config shared across decision types
    base = {
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "pattern_summary": decision.get("pattern_summary"),
        "decision_reason": decision.get("reason"),
    }

    configs = []

    if dec_type == "reversal":
        configs.append(_make_reversal(base, rr_target, risk_base, confidence, aggressive=False))
        if both_variants:
            configs.append(_make_reversal(base, rr_target * 1.2, risk_base * 1.3, confidence, aggressive=True))
    elif dec_type == "breakout":
        configs.append(_make_breakout(base, rr_target, risk_base, trend, aggressive=False))
        if both_variants:
            configs.append(_make_breakout(base, rr_target * 1.3, risk_base * 1.3, trend, aggressive=True))
    elif dec_type == "failed_breakout":
        configs.append(_make_failed_breakout(base, rr_target, risk_base, confidence))

    return configs


def _make_reversal(base: dict, rr: float, risk: float, confidence: float, aggressive: bool) -> dict:
    cfg = StrategyConfig(
        market=MarketScope(symbols=[base["symbol"]], main_tf=base["timeframe"], confirm_tf="1d" if base["timeframe"] == "4h" else "4h"),
        logic_tags=["reversal", "third_touch"],
        logic_combine="AND",
        conditions=[
            ConditionRule(
                indicator="zone_distance",
                condition="lt",
                threshold=0.5,
                params={"unit": "atr"},
                enabled=True,
            ),
            ConditionRule(
                indicator="rsi",
                condition="lt" if base["side"] == "support" else "gt",
                threshold=35.0 if base["side"] == "support" else 65.0,
                params={"period": 14},
                enabled=True,
            ),
        ],
        entry=EntryRules(
            modes=["pre_limit", "rejection"] if aggressive else ["rejection"],
            logic_combine="OR",
            offset_pct=0.1 if aggressive else 0.0,
        ),
        exit=ExitRules(
            stop_type="structure",
            stop_atr_mult=1.2 if aggressive else 1.5,
            tp_type="rr",
            rr_target=round(rr, 2),
        ),
        risk=RiskProfile(
            risk_per_trade=round(risk, 4),
            max_concurrent=3 if aggressive else 2,
            max_drawdown_pct=10.0,
            auto_pause_on_dd=True,
        ),
    )
    return {
        "variant": "aggressive" if aggressive else "conservative",
        "name": f"{base['symbol']}-Reversal-{'Agg' if aggressive else 'Cons'}",
        "decision_type": "reversal",
        "config": asdict(cfg),
        "source_pattern": base.get("pattern_summary"),
        "decision_reason": base.get("decision_reason"),
    }


def _make_breakout(base: dict, rr: float, risk: float, trend: str, aggressive: bool) -> dict:
    cfg = StrategyConfig(
        market=MarketScope(symbols=[base["symbol"]], main_tf=base["timeframe"], confirm_tf="1d" if base["timeframe"] == "4h" else "4h"),
        logic_tags=["breakout", "trend_continuation"],
        logic_combine="AND",
        conditions=[
            ConditionRule(
                indicator="adx",
                condition="gt",
                threshold=20.0,
                params={"period": 14},
                enabled=True,
            ),
            ConditionRule(
                indicator="volume_ratio",
                condition="gt",
                threshold=1.3,
                params={},
                enabled=True,
            ),
        ],
        entry=EntryRules(
            modes=["failed_breakout" if aggressive else "retest"],
            logic_combine="OR",
            offset_pct=0.15 if aggressive else 0.05,
        ),
        exit=ExitRules(
            stop_type="structure",
            stop_atr_mult=1.5,
            tp_type="trailing" if aggressive else "rr",
            rr_target=round(rr, 2),
            trailing_pct=1.5,
        ),
        risk=RiskProfile(
            risk_per_trade=round(risk, 4),
            max_concurrent=2,
            max_drawdown_pct=12.0,
            auto_pause_on_dd=True,
        ),
    )
    return {
        "variant": "aggressive" if aggressive else "conservative",
        "name": f"{base['symbol']}-Breakout-{'Agg' if aggressive else 'Cons'}",
        "decision_type": "breakout",
        "config": asdict(cfg),
        "source_pattern": base.get("pattern_summary"),
        "decision_reason": base.get("decision_reason"),
    }


def _make_failed_breakout(base: dict, rr: float, risk: float, confidence: float) -> dict:
    cfg = StrategyConfig(
        market=MarketScope(symbols=[base["symbol"]], main_tf=base["timeframe"], confirm_tf="1d" if base["timeframe"] == "4h" else "4h"),
        logic_tags=["failed_breakout", "reclaim"],
        logic_combine="AND",
        conditions=[
            ConditionRule(
                indicator="wick_ratio",
                condition="gt",
                threshold=0.6,
                params={},
                enabled=True,
            ),
        ],
        entry=EntryRules(
            modes=["failed_breakout"],
            logic_combine="OR",
            offset_pct=0.0,
        ),
        exit=ExitRules(
            stop_type="structure",
            stop_atr_mult=1.0,  # tight stop beyond failed break wick
            tp_type="rr",
            rr_target=round(rr * 1.2, 2),  # usually get good RR on reclaims
        ),
        risk=RiskProfile(
            risk_per_trade=round(risk * 0.8, 4),  # slightly lower — higher uncertainty
            max_concurrent=2,
            max_drawdown_pct=10.0,
            auto_pause_on_dd=True,
        ),
    )
    return {
        "variant": "single",
        "name": f"{base['symbol']}-FailedBreak",
        "decision_type": "failed_breakout",
        "config": asdict(cfg),
        "source_pattern": base.get("pattern_summary"),
        "decision_reason": base.get("decision_reason"),
    }


# ── Second-layer guard: validate generated config ──────────────────────

def validate_generated_config(draft: dict, match_result: dict) -> dict:
    """Run sanity checks on a generated StrategyConfig before accepting.

    Returns: {"ok": bool, "reason": str, "warnings": [...]}
    """
    config = draft.get("config", {})
    stats = match_result.get("stats", {})
    warnings = []

    rr = config.get("exit", {}).get("rr_target", 0)
    avg_ret = stats.get("avg_return_atr", 0)

    # RR target must be grounded in historical max_return (normalized by assumed stop=1.5 ATR)
    # Max realistic RR = (historical avg_return / 1.5) * 1.5 buffer
    assumed_stop = 1.5
    historical_rr = avg_ret / assumed_stop
    if rr > historical_rr * 1.6:
        return {"ok": False, "reason": f"RR={rr} exceeds 1.6x historical ({historical_rr:.2f})", "warnings": warnings}
    if rr < 1.1:
        return {"ok": False, "reason": f"RR={rr} too low to cover fees/slippage", "warnings": warnings}

    # Risk must be reasonable
    risk = config.get("risk", {}).get("risk_per_trade", 0)
    if risk > 0.02:
        warnings.append(f"risk_per_trade={risk} above 2%")
    if risk < 0.001:
        return {"ok": False, "reason": "risk_per_trade below 0.1% — meaningless", "warnings": warnings}

    # Confidence gate
    confidence = stats.get("confidence", 0)
    if confidence < 0.45:
        return {"ok": False, "reason": f"confidence {confidence:.2f} below gate", "warnings": warnings}

    return {"ok": True, "reason": "passed all guards", "warnings": warnings}
