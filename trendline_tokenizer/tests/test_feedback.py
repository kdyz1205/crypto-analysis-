"""Feedback store + retrain trigger tests."""
from __future__ import annotations
import time
from pathlib import Path

from trendline_tokenizer.schemas.trendline import TrendlineRecord
from trendline_tokenizer.feedback.schemas import (
    CorrectedTrendline, SignalAccepted, SignalRejected, parse_feedback_line,
)
from trendline_tokenizer.feedback.store import FeedbackStore
from trendline_tokenizer.retrain.trigger import (
    count_new_feedback_since, should_retrain,
)
from trendline_tokenizer.registry.manifest import ArtifactManifest


def _toy_record(rid: str = "x") -> TrendlineRecord:
    return TrendlineRecord(
        id=rid, symbol="BTCUSDT", exchange="bitget", timeframe="5m",
        start_time=0, end_time=100, start_bar_index=0, end_bar_index=10,
        start_price=100.0, end_price=101.0,
        line_role="support", direction="up", touch_count=2,
        label_source="manual", created_at=0,
    )


def test_feedback_schemas_parse():
    raw_corrected = {
        "event_type": "corrected_trendline", "created_at": 1,
        "corrected": _toy_record().model_dump(),
        "reason_code": "wrong_anchor", "notes": "moved start anchor 3 bars left",
    }
    parsed = parse_feedback_line(raw_corrected)
    assert isinstance(parsed, CorrectedTrendline)

    raw_accepted = {
        "event_type": "signal_accepted", "created_at": 2, "signal_id": "s1",
        "artifact_name": "fusion-x", "tokenizer_version": "rule.v1",
        "symbol": "BTC", "timeframe": "5m", "action": "BOUNCE",
    }
    parsed = parse_feedback_line(raw_accepted)
    assert isinstance(parsed, SignalAccepted)


def test_feedback_store_round_trip(tmp_path: Path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append(CorrectedTrendline(created_at=1, corrected=_toy_record("c1")))
    store.append(SignalAccepted(created_at=2, signal_id="s1",
                                 artifact_name="m1", tokenizer_version="rule.v1",
                                 symbol="BTC", timeframe="5m", action="BOUNCE"))
    store.append(SignalRejected(created_at=3, signal_id="s2",
                                 artifact_name="m1", tokenizer_version="rule.v1",
                                 symbol="ETH", timeframe="5m", action="BREAK"))
    rows = list(store)
    assert len(rows) == 3
    assert isinstance(rows[0], CorrectedTrendline)
    assert isinstance(rows[1], SignalAccepted)
    assert isinstance(rows[2], SignalRejected)
    assert store.count() == 3
    assert store.count_by_type() == {
        "corrected_trendline": 1, "signal_accepted": 1, "signal_rejected": 1,
    }


def test_should_retrain_threshold(tmp_path: Path):
    fp = tmp_path / "fb.jsonl"
    store = FeedbackStore(fp)
    # No file yet -> no retrain
    decide, n = should_retrain(feedback_path=fp, last_manifest=None, min_new=2)
    assert decide is False and n == 0

    # 3 events, 2 after the manifest's created_at -> retrain
    store.append(CorrectedTrendline(created_at=10, corrected=_toy_record("a")))
    store.append(CorrectedTrendline(created_at=20, corrected=_toy_record("b")))
    store.append(CorrectedTrendline(created_at=30, corrected=_toy_record("c")))

    fake_manifest = ArtifactManifest(
        artifact_name="x", model_kind="fusion", model_version="v1",
        tokenizer_version="rule.v1", training_dataset_version="d",
        feature_norm_version="none",
        ckpt_path="x", config_path="x", metrics={}, created_at=15,
    )
    decide, n = should_retrain(feedback_path=fp, last_manifest=fake_manifest, min_new=2)
    assert decide is True
    assert n == 2


def test_count_new_feedback_since(tmp_path: Path):
    fp = tmp_path / "fb.jsonl"
    store = FeedbackStore(fp)
    for ts in (5, 15, 25):
        store.append(CorrectedTrendline(created_at=ts, corrected=_toy_record(f"r{ts}")))
    assert count_new_feedback_since(store, 0) == 3
    assert count_new_feedback_since(store, 10) == 2
    assert count_new_feedback_since(store, 100) == 0
