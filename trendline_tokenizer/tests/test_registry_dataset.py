"""DatasetManifest + ExperimentManifest + splits — institutional-grade
provenance scaffolding. These are the audit trail every training run
must produce, per CLAUDE.md DO NOT LIE + the user's research-grade spec.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pytest

from trendline_tokenizer.registry.dataset import (
    DatasetManifest, RawSource, SplitSpec, sha256_file,
    build_manifest_from_collect_records,
)
from trendline_tokenizer.registry.experiment import (
    ExperimentManifest, EpochMetric, begin_experiment, finalize_experiment,
)
from trendline_tokenizer.training.splits import (
    random_split, time_forward_split, symbol_heldout_split,
    regime_heldout_split, split_records,
)
from trendline_tokenizer.schemas.trendline import TrendlineRecord


def _toy_record(rid: str, symbol: str = "BTC", end_time: int = 0,
                volatility: float | None = 0.01) -> TrendlineRecord:
    return TrendlineRecord(
        id=rid, symbol=symbol, exchange="bitget", timeframe="5m",
        start_time=0, end_time=end_time,
        start_bar_index=0, end_bar_index=end_time // 60 if end_time else 1,
        start_price=100.0, end_price=101.0,
        line_role="support", direction="up", touch_count=2,
        volatility_atr_pct=volatility,
        label_source="auto", created_at=0,
    )


# ── DatasetManifest ──────────────────────────────────────────────────

def test_sha256_of_file_is_stable(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world\n")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2 and len(h1) == 64


def test_sha256_missing_file_returns_empty(tmp_path: Path):
    assert sha256_file(tmp_path / "nope.bin") == ""


def test_dataset_manifest_round_trip(tmp_path: Path):
    m = DatasetManifest(
        dataset_name="trendline-prototype",
        dataset_version="trendline-structure-v0.1.0",
        created_at=int(time.time()),
        git_commit="abc123",
        raw_sources=[RawSource(path="data/x.jsonl", sha256="d34db33f",
                               size_bytes=42, role="auto_patterns")],
        symbols=["BTCUSDT"], timeframes=["5m"],
        n_trendline_records_total=271773,
        n_manual_records=82, n_auto_records=271773,
        split_policy=SplitSpec(policy="time_forward",
                                val_start_ts=1700000000,
                                test_start_ts=1750000000),
    )
    p = tmp_path / "manifest.json"
    m.save(p)
    loaded = DatasetManifest.load(p)
    assert loaded == m
    # manifest_sha256 stable + nonempty
    h1 = m.manifest_sha256()
    h2 = loaded.manifest_sha256()
    assert h1 == h2 and len(h1) == 64


def test_build_manifest_from_collect_records_stats(tmp_path: Path):
    # Simulate the dict shape that collect_records returns
    fake_stats = {
        "manual_loaded": 82, "manual_after_enrich": 82,
        "manual_oversample_factor": 50, "manual_total_in_pool": 4100,
        "auto_loaded": 271773, "auto_sweep_loaded": 1431676,
        "feedback_corrected": 0,
        "outcomes_coverage": {"total_outcome_rows": 1393, "n_records": 82,
                              "n_unique_ids": 82, "n_outcomes_join": 42,
                              "n_labels_join": 32,
                              "n_ever_filled": 33, "n_ever_won": 6},
    }
    raw = tmp_path / "fake_pattern.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    m = build_manifest_from_collect_records(
        dataset_name="x", dataset_version="x.v1",
        raw_paths={"auto_patterns": raw},
        symbols=["BTCUSDT"], timeframes=["5m"],
        stats=fake_stats,
    )
    assert m.n_manual_records == 82
    assert m.n_auto_records == 271773
    assert m.n_sweep_records == 1431676
    assert m.n_trendline_records_total == 82 + 271773 + 1431676
    assert m.n_outcome_records == 1393
    assert len(m.raw_sources) == 1
    assert m.raw_sources[0].sha256 != ""


# ── ExperimentManifest ──────────────────────────────────────────────────

def test_experiment_manifest_begin_and_finalize(tmp_path: Path):
    # Need a dataset manifest first
    dm = DatasetManifest(
        dataset_name="x", dataset_version="x.v1",
        created_at=0, n_trendline_records_total=100,
    )
    dm_path = tmp_path / "ds.json"
    dm.save(dm_path)

    exp = begin_experiment(
        experiment_kind="pretrain",
        cli_args={"epochs": 3, "batch_size": 32},
        model_cfg={"d_model": 96},
        dataset_manifest_path=dm_path,
        tokenizer_version="rule.v1+raw.v1",
    )
    assert exp.experiment_kind == "pretrain"
    assert exp.dataset_manifest_sha256 == dm.manifest_sha256()
    assert exp.tokenizer_version == "rule.v1+raw.v1"

    exp.add_epoch(epoch=0, train_loss=4.6, val_loss=None,
                  n_ok=844, n_skipped_nan=0, seconds=92.2)
    exp.add_epoch(epoch=1, train_loss=2.5, n_ok=844, n_skipped_nan=0)
    assert len(exp.epoch_metrics) == 2

    # Fake checkpoint to compute sha256
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"\x00" * 1024)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"d_model": 96}), encoding="utf-8")

    out = finalize_experiment(
        exp, checkpoint_path=ckpt, config_path=cfg,
        final_metrics={"val_loss": 2.32}, out_dir=tmp_path,
    )
    assert out.exists()
    loaded = ExperimentManifest.load(out)
    assert loaded.checkpoint_sha256 != ""
    assert loaded.final_metrics["val_loss"] == 2.32
    assert len(loaded.epoch_metrics) == 2


# ── splits ──────────────────────────────────────────────────────────────

def test_random_split_marked_as_smoke_only():
    recs = [_toy_record(f"r{i}") for i in range(100)]
    s = random_split(recs)
    assert s.policy == "random"
    assert "warning" in s.metadata
    assert len(s.train_idx) + len(s.val_idx) + len(s.test_idx) == 100


def test_time_forward_split_no_lookahead():
    """train.end_time < val.end_time < test.end_time (strict)."""
    recs = [_toy_record(f"r{i}", end_time=i * 60) for i in range(100)]
    s = time_forward_split(recs)
    train_max = max(recs[i].end_time for i in s.train_idx)
    val_min = min(recs[i].end_time for i in s.val_idx)
    val_max = max(recs[i].end_time for i in s.val_idx)
    test_min = min(recs[i].end_time for i in s.test_idx)
    assert train_max < val_min, "train should not overlap val time-wise"
    assert val_max < test_min, "val should not overlap test time-wise"


def test_symbol_heldout_split_no_overlap():
    recs = []
    for sym in ("BTC", "ETH", "SOL", "HYPE", "ADA"):
        for i in range(20):
            recs.append(_toy_record(f"{sym}-{i}", symbol=sym))
    s = symbol_heldout_split(recs)
    train_syms = {recs[i].symbol for i in s.train_idx}
    val_syms = {recs[i].symbol for i in s.val_idx}
    test_syms = {recs[i].symbol for i in s.test_idx}
    assert train_syms.isdisjoint(test_syms), "train ∩ test symbols must be empty"
    assert train_syms.isdisjoint(val_syms), "train ∩ val symbols must be empty"


def test_regime_heldout_split_high_vol():
    recs = (
        [_toy_record(f"low-{i}", volatility=0.001) for i in range(20)]
      + [_toy_record(f"normal-{i}", volatility=0.008) for i in range(50)]
      + [_toy_record(f"high-{i}", volatility=0.025) for i in range(15)]
    )
    s = regime_heldout_split(recs, test_regime="high_vol")
    # Test set should be only high-vol records
    for i in s.test_idx:
        assert recs[i].volatility_atr_pct >= 0.015
    # Train + val should be only low/normal
    for i in list(s.train_idx) + list(s.val_idx):
        v = recs[i].volatility_atr_pct
        assert v is None or v < 0.015


def test_split_records_dispatch():
    recs = [_toy_record(f"r{i}", end_time=i * 60) for i in range(50)]
    s = split_records(recs, policy="time_forward")
    assert s.policy == "time_forward"
    with pytest.raises(ValueError):
        split_records(recs, policy="bogus")
