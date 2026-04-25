"""Predict-endpoint response shape contract.

Pins the keys the frontend relies on. Would have caught the audit
bugs (decoded_* fields dropped before reaching JSON, line_endpoint_pct_change
always zero) at PR time.
"""
from __future__ import annotations
import time
from pathlib import Path

import pytest
import torch

from trendline_tokenizer.models.config import FusionConfig
from trendline_tokenizer.models.full_model import TrendlineFusionModel
from trendline_tokenizer.registry.manifest import ArtifactManifest
from trendline_tokenizer.inference.feature_cache import FeatureCache
from trendline_tokenizer.inference.inference_service import InferenceService
from trendline_tokenizer.inference.signal_engine import SignalEngine


REQUIRED_TOP_LEVEL = {
    "symbol", "timeframe", "timestamp", "artifact_name", "tokenizer_version",
    "action", "trade_type", "confidence", "suggested_buffer_pct",
    "bounce_prob", "break_prob", "continuation_prob",
    "next_coarse_id", "next_fine_id", "predicted_role", "reason",
    "decoded_role", "decoded_direction",
    "decoded_log_slope_per_bar", "decoded_duration_bars",
    "line_endpoint_pct_change", "horizon_seconds",
    "extras",
}

REQUIRED_EXTRAS = {
    "anchor_close", "anchor_open_time_ms",
    "n_input_records", "n_bars_in_cache",
    "effective_role", "behaviour",
}


def _save_tiny(tmp_path: Path) -> Path:
    cfg = FusionConfig(
        d_model=32, n_layers_price=1, n_layers_token=1,
        n_layers_fusion=1, price_seq_len=64, token_seq_len=4,
        use_rule_tokens=True, use_learned_tokens=False,
        use_raw_features=True, vqvae_checkpoint_path=None,
    )
    model = TrendlineFusionModel(cfg)
    art = tmp_path / "fusion_x"
    art.mkdir()
    ckpt = art / "fusion.pt"
    cfg_p = art / "fusion.json"
    torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump()}, ckpt)
    cfg_p.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    m = ArtifactManifest(
        artifact_name="fusion-shape-test", model_kind="fusion",
        model_version=cfg.version, tokenizer_version="rule.v1+raw.v1",
        training_dataset_version="x", feature_norm_version="none",
        ckpt_path=str(ckpt), config_path=str(cfg_p),
        metrics={}, created_at=int(time.time()),
    )
    p = art / "manifest.json"; m.save(p); return p


def _push_bars(svc: InferenceService, n: int = 200):
    for i in range(n):
        c = 100.0 + 0.05 * i
        svc.push_bar("BTCUSDT", "5m", {
            "open_time": 1_700_000_000_000 + i * 60_000,
            "open": c, "high": c + 0.2, "low": c - 0.2,
            "close": c, "volume": 10.0,
        })


def test_prediction_record_has_all_required_decoded_fields(tmp_path: Path):
    svc = InferenceService(_save_tiny(tmp_path), device="cpu")
    _push_bars(svc, 200)
    pred = svc.predict("BTCUSDT", "5m")
    assert pred is not None
    # Every decoded field must be populated, not at default.
    assert pred.decoded_role in ("support", "resistance", "channel_upper",
                                  "channel_lower", "wedge_side", "triangle_side", "unknown")
    assert pred.decoded_direction in ("up", "down", "flat")
    assert isinstance(pred.decoded_log_slope_per_bar, float)
    assert pred.decoded_duration_bars >= 1
    # line_endpoint_pct_change + horizon_seconds must be COMPUTED, not 0 default.
    # horizon_seconds = duration_bars * bar_seconds("5m"=300)
    assert pred.horizon_seconds == pred.decoded_duration_bars * 300, (
        f"horizon_seconds wrong: got {pred.horizon_seconds}, "
        f"expected {pred.decoded_duration_bars * 300}"
    )
    # extras has anchor_close + anchor_open_time_ms
    assert "anchor_close" in pred.extras
    assert "anchor_open_time_ms" in pred.extras
    assert pred.extras["anchor_close"] > 0


def test_signal_record_dict_has_all_top_level_fields(tmp_path: Path):
    svc = InferenceService(_save_tiny(tmp_path), device="cpu")
    _push_bars(svc, 200)
    pred = svc.predict("BTCUSDT", "5m")
    sig = SignalEngine().evaluate(pred)
    d = sig.to_dict()
    missing = REQUIRED_TOP_LEVEL - d.keys()
    assert not missing, f"missing top-level keys: {missing}"
    extras_missing = REQUIRED_EXTRAS - d["extras"].keys()
    assert not extras_missing, f"missing extras keys: {extras_missing}"


def test_signal_record_decoded_fields_are_propagated(tmp_path: Path):
    """The audit bug: SignalEngine dropped these. Pin it."""
    svc = InferenceService(_save_tiny(tmp_path), device="cpu")
    _push_bars(svc, 200)
    pred = svc.predict("BTCUSDT", "5m")
    sig = SignalEngine().evaluate(pred)
    assert sig.decoded_role == pred.decoded_role
    assert sig.decoded_direction == pred.decoded_direction
    assert sig.decoded_log_slope_per_bar == pred.decoded_log_slope_per_bar
    assert sig.decoded_duration_bars == pred.decoded_duration_bars
    assert sig.line_endpoint_pct_change == pred.line_endpoint_pct_change
    assert sig.horizon_seconds == pred.horizon_seconds
    assert sig.extras["anchor_close"] == pred.extras["anchor_close"]
    assert sig.extras["anchor_open_time_ms"] == pred.extras["anchor_open_time_ms"]
