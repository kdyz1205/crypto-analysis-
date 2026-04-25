"""End-to-end inference pipeline test.

Trains a tiny model, saves the manifest, then loads it via
InferenceService, pushes synthetic bars, runs predict, and dispatches
the signal to the paper log. Asserts the manifest contract, version
matching, and the SignalRecord shape.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from trendline_tokenizer.models.config import FusionConfig
from trendline_tokenizer.models.full_model import TrendlineFusionModel
from trendline_tokenizer.registry.manifest import ArtifactManifest
from trendline_tokenizer.inference.feature_cache import FeatureCache
from trendline_tokenizer.inference.inference_service import InferenceService
from trendline_tokenizer.inference.signal_engine import SignalEngine
from trendline_tokenizer.inference.paper_dispatcher import PaperDispatcher
from trendline_tokenizer.inference.runtime_tokenizer import RuntimeTokenizer


def _save_tiny_model(tmp_path: Path) -> Path:
    """Save a randomly-initialised tiny model + a manifest."""
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1,
                       n_layers_fusion=1, price_seq_len=64, token_seq_len=4,
                       use_rule_tokens=True,
                       use_learned_tokens=False,    # no vqvae checkpoint in test
                       use_raw_features=True,
                       vqvae_checkpoint_path=None)
    model = TrendlineFusionModel(cfg)
    art_dir = tmp_path / "fusion_test"
    art_dir.mkdir()
    ckpt = art_dir / "fusion.pt"
    cfg_path = art_dir / "fusion.json"
    torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump()}, ckpt)
    cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    manifest = ArtifactManifest(
        artifact_name="fusion-test",
        model_kind="fusion",
        model_version=cfg.version,
        tokenizer_version="rule.v1+raw.v1",
        training_dataset_version="synthetic-test",
        feature_norm_version="none",
        ckpt_path=str(ckpt), config_path=str(cfg_path),
        metrics={"val_loss": 0.0, "n_train": 0, "n_val": 0},
        created_at=int(time.time()),
    )
    manifest_path = art_dir / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def _push_synthetic_bars(svc: InferenceService, symbol: str, tf: str, n: int = 200):
    """Push n closed bars with a clean uptrend so the runtime detector
    finds at least one trendline."""
    base = 100.0
    for i in range(n):
        c = base + 0.05 * i + 0.3 * np.sin(i / 5.0)
        bar = {
            "open_time": 1_700_000_000_000 + i * 60_000,
            "open": c, "high": c + 0.2, "low": c - 0.2,
            "close": c, "volume": 10.0 + (i % 7),
        }
        svc.push_bar(symbol, tf, bar)


def test_runtime_tokenizer_version_string():
    rt = RuntimeTokenizer(use_rule=True, use_learned=False, use_raw=True)
    assert rt.tokenizer_version() == "rule.v1+raw.v1"


def test_runtime_tokenizer_assert_matches():
    rt = RuntimeTokenizer(use_rule=True, use_learned=False, use_raw=True)
    rt.assert_version_matches("rule.v1+raw.v1")
    with pytest.raises(ValueError):
        rt.assert_version_matches("rule.v1+vqvae.v0+raw.v1")


def test_inference_service_loads_manifest_and_predicts(tmp_path: Path):
    manifest_path = _save_tiny_model(tmp_path)
    svc = InferenceService(manifest_path, device="cpu")

    # not enough bars yet -> None
    assert svc.predict("BTCUSDT", "5m") is None

    _push_synthetic_bars(svc, "BTCUSDT", "5m", n=200)
    pred = svc.predict("BTCUSDT", "5m")
    assert pred is not None
    assert pred.symbol == "BTCUSDT"
    assert pred.timeframe == "5m"
    assert pred.tokenizer_version == "rule.v1+raw.v1"
    assert 0.0 <= pred.bounce_prob <= 1.0
    assert 0.0 <= pred.break_prob <= 1.0
    assert pred.suggested_buffer_pct >= 0
    assert pred.n_bars_in_cache == 200


def test_inference_service_rejects_version_mismatch(tmp_path: Path):
    manifest_path = _save_tiny_model(tmp_path)
    # tamper the manifest to claim a tokenizer the runtime can't produce
    m = ArtifactManifest.load(manifest_path)
    m_data = m.model_dump()
    m_data["tokenizer_version"] = "rule.v1+vqvae.v0+raw.v1"  # vqvae expected but cfg disabled
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps(m_data), encoding="utf-8")
    with pytest.raises(ValueError):
        InferenceService(bad_manifest, device="cpu")


def test_signal_engine_and_paper_dispatcher(tmp_path: Path):
    manifest_path = _save_tiny_model(tmp_path)
    svc = InferenceService(manifest_path, device="cpu")
    _push_synthetic_bars(svc, "BTCUSDT", "5m", n=200)
    pred = svc.predict("BTCUSDT", "5m")
    assert pred is not None

    engine = SignalEngine()
    sig = engine.evaluate(pred)
    assert sig.action in ("BOUNCE", "BREAK", "WAIT")
    assert 0.0 <= sig.confidence <= 1.0
    assert "edge" in sig.reason

    dispatcher = PaperDispatcher(tmp_path / "signals.jsonl")
    dispatcher.dispatch(sig)
    rows = dispatcher.read_all()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTCUSDT"
    assert rows[0]["action"] == sig.action
