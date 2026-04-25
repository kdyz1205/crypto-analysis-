from __future__ import annotations
import pytest

from backtests.ma_ribbon_ema21.cohort_report import CohortStats
from backtests.ma_ribbon_ema21.acceptance_gate import (
    evaluate_phase1_gate,
    GateResult,
)


def _stat(sym, tf, bucket, split, mean_ret, count=5):
    return CohortStats(
        symbol=sym, tf=tf, bucket=bucket, split=split, count=count,
        mean_return_post_fee=mean_ret, median_return_post_fee=mean_ret,
        win_rate=0.5 if mean_ret > 0 else 0.4,
        worst_return_post_fee=mean_ret - 0.05,
    )


def test_gate_passes_when_30_percent_symbols_meet_threshold():
    cohorts = []
    syms_passing = ["AAA", "BBB", "CCC", "DDD"]   # 4
    syms_failing = ["EEE", "FFF", "GGG", "HHH", "III", "JJJ"]  # 6
    for s in syms_passing:
        cohorts.append(_stat(s, "1h", "[0.0%, 0.5%)", "test", 0.02))
    for s in syms_failing:
        cohorts.append(_stat(s, "1h", "[0.0%, 0.5%)", "test", 0.005))
    result = evaluate_phase1_gate(cohorts, horizon=20, threshold_pct=0.01,
                                  min_symbol_pct=0.30)
    # 4 / 10 = 40% pass → ≥ 30% → GATE PASS
    assert isinstance(result, GateResult)
    assert result.passed is True
    assert result.symbols_passing >= 3


def test_gate_fails_when_no_symbols_meet_threshold():
    cohorts = [_stat(s, "1h", "[0.0%, 0.5%)", "test", 0.005) for s in "ABCDE"]
    result = evaluate_phase1_gate(cohorts, horizon=20, threshold_pct=0.01,
                                  min_symbol_pct=0.30)
    assert result.passed is False


def test_gate_only_uses_test_split_not_train():
    cohorts = [
        _stat("AAA", "1h", "[0.0%, 0.5%)", "train", 0.05),  # ignored
        _stat("AAA", "1h", "[0.0%, 0.5%)", "test", 0.001),
    ]
    result = evaluate_phase1_gate(cohorts, horizon=20, threshold_pct=0.01,
                                  min_symbol_pct=0.30)
    assert result.passed is False


def test_gate_reports_failing_symbols_explicitly():
    cohorts = [
        _stat("WIN1", "1h", "[0.0%, 0.5%)", "test",  0.02),
        _stat("LOS1", "1h", "[0.0%, 0.5%)", "test", -0.005),
        _stat("LOS2", "1h", "[0.0%, 0.5%)", "test",  0.001),
    ]
    result = evaluate_phase1_gate(cohorts, horizon=20, threshold_pct=0.01,
                                  min_symbol_pct=0.50)
    assert "LOS1" in result.failing_symbols
    assert "LOS2" in result.failing_symbols
    assert "WIN1" in result.passing_symbols
