from __future__ import annotations
from pathlib import Path
import pytest

from backtests.ma_ribbon_ema21.phase1_engine import Phase1Event
from backtests.ma_ribbon_ema21.cohort_report import (
    aggregate_cohorts,
    write_markdown_report,
    CohortStats,
)


def _evt(sym, tf, idx, dist5, ret_post, split):
    return Phase1Event(
        symbol=sym, tf=tf, formation_bar_idx=idx,
        formation_timestamp=1_700_000_000 + idx,
        formation_close=100.0,
        distance_to_ma5_pct=dist5,
        distance_to_ma8_pct=dist5,
        distance_to_ema21_pct=dist5,
        distance_to_ma55_pct=dist5,
        ribbon_width_pct=0.01,
        forward_returns={20: ret_post + 0.0012},   # raw before round-trip cost
        forward_returns_post_fee={20: ret_post},
        split=split,
    )


def test_aggregate_cohorts_groups_by_symbol_tf_bucket():
    events = [
        _evt("AAA", "1h", 100, 0.001, 0.02,  "train"),
        _evt("AAA", "1h", 200, 0.001, 0.03,  "train"),
        _evt("AAA", "1h", 300, 0.025, -0.01, "test"),
    ]
    cohorts = aggregate_cohorts(events, horizon=20)
    keys = {(c.symbol, c.tf, c.bucket, c.split) for c in cohorts}
    assert ("AAA", "1h", "[0.0%, 0.5%)", "train") in keys
    assert ("AAA", "1h", "[2.0%, 4.0%)", "test") in keys


def test_aggregate_cohorts_computes_mean_and_winrate():
    events = [
        _evt("AAA", "1h", 100, 0.001, 0.02,  "train"),
        _evt("AAA", "1h", 200, 0.001, -0.01, "train"),
        _evt("AAA", "1h", 300, 0.001, 0.03,  "train"),
    ]
    cohorts = aggregate_cohorts(events, horizon=20)
    c = next(c for c in cohorts if c.split == "train")
    assert c.count == 3
    assert c.mean_return_post_fee == pytest.approx((0.02 - 0.01 + 0.03) / 3)
    assert c.win_rate == pytest.approx(2/3)


def test_write_markdown_report_creates_file(tmp_path):
    events = [_evt("AAA", "1h", 100, 0.001, 0.02, "train")]
    cohorts = aggregate_cohorts(events, horizon=20)
    out = tmp_path / "report.md"
    write_markdown_report(cohorts, output_path=str(out), horizon=20)
    text = out.read_text()
    assert out.exists()
    assert "AAA" in text and "1h" in text
    assert "0.0%" in text
