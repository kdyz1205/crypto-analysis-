"""Agent infrastructure tests: state persistence, style scoring, reports."""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

from trendline_tokenizer.agent.state import AgentState
from trendline_tokenizer.agent.report import IterationReport, tail_reports
from trendline_tokenizer.agent.style_score import StyleScorer
from trendline_tokenizer.schemas.trendline import TrendlineRecord


def _toy_record(rid: str = "x", role: str = "support",
                start_idx: int = 10, end_idx: int = 50,
                start_price: float = 100.0, end_price: float = 102.0) -> TrendlineRecord:
    return TrendlineRecord(
        id=rid, symbol="BTCUSDT", exchange="bitget", timeframe="5m",
        start_time=start_idx * 60, end_time=end_idx * 60,
        start_bar_index=start_idx, end_bar_index=end_idx,
        start_price=start_price, end_price=end_price,
        line_role=role, direction="up" if end_price > start_price else "down",
        touch_count=2,
        label_source="manual", created_at=0,
    )


# ── AgentState ───────────────────────────────────────────────────────────

def test_agent_state_round_trip(tmp_path: Path):
    s = AgentState(iteration=5, last_run_ts=1234, last_train_artifact="fusion-x",
                   last_seen_outcome_ts=42, last_seen_manual_count=78,
                   n_lines_auto_drawn_total=120, n_retrain_triggered=3)
    path = tmp_path / "state.json"
    s.save(path)
    loaded = AgentState.load(path)
    assert loaded == s


def test_agent_state_load_missing_returns_default(tmp_path: Path):
    s = AgentState.load(tmp_path / "nope.json")
    assert s.iteration == 0
    assert s.last_train_artifact == ""


def test_agent_state_record_iteration_caps_log(tmp_path: Path):
    s = AgentState()
    for i in range(250):
        s.record_iteration({"k": i})
    assert len(s.log) == 200    # capped


# ── IterationReport ──────────────────────────────────────────────────────

def test_iteration_report_round_trip(tmp_path: Path):
    rep = IterationReport(iteration=1, started_at=int(time.time()))
    rep.n_new_outcomes = 5
    rep.n_lines_auto_drawn = 12
    rep.retrained = True
    rep.new_artifact = "fusion-test"
    rep.backtest_summary = {"BTCUSDT_5m": {"hit_rate": 0.5}}
    path = tmp_path / "rep.jsonl"
    rep.append(path)
    assert rep.duration_s >= 0
    assert rep.finished_at >= rep.started_at
    rows = tail_reports(path)
    assert len(rows) == 1
    assert rows[0]["new_artifact"] == "fusion-test"
    assert rows[0]["backtest_summary"]["BTCUSDT_5m"]["hit_rate"] == 0.5


def test_iteration_report_append_grows(tmp_path: Path):
    path = tmp_path / "rep.jsonl"
    for i in range(5):
        rep = IterationReport(iteration=i, started_at=int(time.time()))
        rep.append(path)
    rows = tail_reports(path, n=3)
    assert len(rows) == 3
    assert rows[-1]["iteration"] == 4   # most recent


# ── StyleScorer ──────────────────────────────────────────────────────────

def test_style_scorer_empty_centroid():
    s = StyleScorer(manual_records=[])
    assert s.n_manual == 0
    # Returns 0.5 (neutral) when no centroid
    assert s.score_one(_toy_record()) == 0.5


def test_style_scorer_self_high_similarity():
    """A record IS the centroid -> score should be near max."""
    rec = _toy_record(role="support", start_price=100, end_price=102)
    s = StyleScorer(manual_records=[rec])
    assert s.score_one(rec) > 0.95


def test_style_scorer_distinguishes_dissimilar():
    """A wildly-different record scores lower than a similar one."""
    base = _toy_record(role="support", start_price=100, end_price=102)
    s = StyleScorer(manual_records=[base])
    similar = _toy_record(role="support", start_price=100, end_price=102.1)
    different = _toy_record(role="resistance", start_idx=10, end_idx=200,
                            start_price=100, end_price=80)
    assert s.score_one(similar) > s.score_one(different)


def test_style_scorer_filter_by_style():
    base = _toy_record(role="support", start_price=100, end_price=102)
    s = StyleScorer(manual_records=[base])
    candidates = [
        _toy_record("a", role="support", start_price=100, end_price=102.0),    # high
        _toy_record("b", role="resistance", start_idx=10, end_idx=200,          # low
                    start_price=100, end_price=80),
    ]
    kept = s.filter_by_style(candidates, min_score=0.7)
    kept_ids = {r.id for r in kept}
    assert "a" in kept_ids
    # 'b' should be filtered (different role + direction + magnitude)


# ── loop helpers ─────────────────────────────────────────────────────────

def test_count_manual_lines_handles_missing(tmp_path: Path):
    from trendline_tokenizer.agent.loop import count_manual_lines
    assert count_manual_lines(tmp_path / "absent.json") == 0


def test_count_manual_lines_list_format(tmp_path: Path):
    from trendline_tokenizer.agent.loop import count_manual_lines
    p = tmp_path / "manual.json"
    p.write_text(json.dumps([{"id": "x"}, {"id": "y"}]), encoding="utf-8")
    assert count_manual_lines(p) == 2


def test_count_manual_lines_dict_wrapper(tmp_path: Path):
    from trendline_tokenizer.agent.loop import count_manual_lines
    p = tmp_path / "manual.json"
    p.write_text(json.dumps({"drawings": [{}, {}, {}]}), encoding="utf-8")
    assert count_manual_lines(p) == 3
