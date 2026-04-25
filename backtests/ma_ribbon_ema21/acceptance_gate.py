"""Phase 1 acceptance gate evaluator."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from backtests.ma_ribbon_ema21.cohort_report import CohortStats


_TARGET_BUCKET = "[0.0%, 0.5%)"


@dataclass
class GateResult:
    passed: bool
    threshold_pct: float
    min_symbol_pct: float
    horizon: int
    symbols_total: int
    symbols_passing: int
    passing_symbols: list[str]
    failing_symbols: list[str]
    reason: str


def evaluate_phase1_gate(
    cohorts: list[CohortStats],
    horizon: int = 20,
    threshold_pct: float = 0.01,
    min_symbol_pct: float = 0.30,
    target_bucket: str = _TARGET_BUCKET,
) -> GateResult:
    """A symbol counts as 'passing' iff, on at least one TF, the cohort
    (target_bucket, split='test') has mean_return_post_fee > threshold_pct.
    """
    by_symbol: dict[str, list[CohortStats]] = defaultdict(list)
    for c in cohorts:
        if c.split != "test":
            continue
        if c.bucket != target_bucket:
            continue
        by_symbol[c.symbol].append(c)

    passing: list[str] = []
    failing: list[str] = []
    for sym, sym_cohorts in by_symbol.items():
        ok = any(c.mean_return_post_fee > threshold_pct for c in sym_cohorts)
        (passing if ok else failing).append(sym)

    total = len(passing) + len(failing)
    pass_pct = (len(passing) / total) if total > 0 else 0.0
    passed = pass_pct >= min_symbol_pct

    reason = (
        f"{len(passing)}/{total} symbols ({pass_pct:.0%}) cleared mean +{horizon}-bar "
        f"return > {threshold_pct:.1%} on {target_bucket} (test split). "
        f"Required: {min_symbol_pct:.0%}."
    )
    return GateResult(
        passed=passed,
        threshold_pct=threshold_pct,
        min_symbol_pct=min_symbol_pct,
        horizon=horizon,
        symbols_total=total,
        symbols_passing=len(passing),
        passing_symbols=sorted(passing),
        failing_symbols=sorted(failing),
        reason=reason,
    )
