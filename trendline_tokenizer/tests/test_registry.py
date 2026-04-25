"""Registry: every checkpoint must record tokenizer version, training
data version, and metrics. Loading a model demands the manifest.
"""
from pathlib import Path

import pytest

from trendline_tokenizer.registry.manifest import ArtifactManifest


def test_manifest_round_trip(tmp_path: Path):
    manifest = ArtifactManifest(
        artifact_name="fusion.v0.1-2026-04-24",
        model_kind="fusion",
        model_version="fusion.v0.1",
        tokenizer_version="rule.v1+vqvae.v0+raw.v1",
        training_dataset_version="2026-04-24-271k-auto",
        feature_norm_version="none",
        ckpt_path="checkpoints/fusion/fusion_v0.1.pt",
        config_path="checkpoints/fusion/fusion_v0.1.json",
        metrics={"val_next_coarse_acc": 0.42, "val_bounce_auc": 0.71,
                 "streams": {"rule": True, "learned": True, "raw": True}},
        created_at=1700000000,
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = ArtifactManifest.load(path)
    assert loaded == manifest


def test_manifest_rejects_missing_versions():
    with pytest.raises(ValueError):
        ArtifactManifest(
            artifact_name="bad", model_kind="fusion", model_version="",
            tokenizer_version="rule.v1", training_dataset_version="x",
            feature_norm_version="none",
            ckpt_path="x", config_path="x", metrics={}, created_at=0,
        )


def test_manifest_metrics_can_be_nested():
    m = ArtifactManifest(
        artifact_name="x", model_kind="fusion", model_version="v1",
        tokenizer_version="rule.v1", training_dataset_version="d1",
        feature_norm_version="none",
        ckpt_path="x", config_path="x",
        metrics={"by_symbol": {"BTCUSDT": {"acc": 0.5, "auc": 0.7}}},
        created_at=0,
    )
    assert m.metrics["by_symbol"]["BTCUSDT"]["acc"] == 0.5
