"""Backtest stack tests: replay determinism, simulator P&L logic, metrics."""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from trendline_tokenizer.models.config import FusionConfig
from trendline_tokenizer.models.full_model import TrendlineFusionModel
from trendline_tokenizer.registry.manifest import ArtifactManifest
from trendline_tokenizer.inference.inference_service import InferenceService
from trendline_tokenizer.inference.signal_engine import SignalEngine, SignalRecord
from trendline_tokenizer.backtest.replay_engine import replay, ReplayStep
from trendline_tokenizer.backtest.strategy_simulator import simulate, Trade
from trendline_tokenizer.backtest.metrics import compute_metrics, summarize_metrics


def _save_tiny(tmp_path: Path) -> Path:
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1,
                       n_layers_fusion=1, price_seq_len=64, token_seq_len=4,
                       use_rule_tokens=True, use_learned_tokens=False,
                       use_raw_features=True, vqvae_checkpoint_path=None)
    model = TrendlineFusionModel(cfg)
    art = tmp_path / "fusion_bt"
    art.mkdir()
    ckpt = art / "fusion.pt"
    cfg_p = art / "fusion.json"
    torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump()}, ckpt)
    cfg_p.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    m = ArtifactManifest(
        artifact_name="fusion-bt", model_kind="fusion", model_version=cfg.version,
        tokenizer_version="rule.v1+raw.v1", training_dataset_version="x",
        feature_norm_version="none",
        ckpt_path=str(ckpt), config_path=str(cfg_p),
        metrics={}, created_at=int(time.time()),
    )
    p = art / "manifest.json"
    m.save(p)
    return p


def _toy_df(n=200):
    t = np.arange(n) * 60_000 + 1_700_000_000_000
    c = 100 + 0.05 * np.arange(n) + 0.3 * np.sin(np.arange(n) / 5.0)
    return pd.DataFrame({
        "open_time": t, "open": c, "high": c + 0.3, "low": c - 0.3,
        "close": c, "volume": 10.0 + (np.arange(n) % 7),
    })


def test_replay_yields_one_step_per_bar(tmp_path: Path):
    svc = InferenceService(_save_tiny(tmp_path), device="cpu")
    df = _toy_df(150)
    steps = list(replay(df, symbol="BTCUSDT", timeframe="5m",
                        service=svc, predict_every=1, start_bar=64))
    assert len(steps) == len(df) - 64
    assert steps[0].bar_index == 64
    assert steps[-1].bar_index == len(df) - 1


def test_replay_no_lookahead(tmp_path: Path):
    """The service must only see bars up to the current step (and no
    bars before start_bar)."""
    svc = InferenceService(_save_tiny(tmp_path), device="cpu")
    df = _toy_df(150)
    start_bar = 70
    last_seen = -1
    last_cache = 0
    for step in replay(df, symbol="BTCUSDT", timeframe="5m",
                       service=svc, predict_every=10, start_bar=start_bar):
        if step.prediction is not None:
            # cache contains exactly the bars from start_bar..bar_index
            expected = step.bar_index - start_bar + 1
            assert step.prediction.n_bars_in_cache == expected
            assert step.prediction.n_bars_in_cache > last_cache
            last_cache = step.prediction.n_bars_in_cache
        assert step.bar_index > last_seen
        last_seen = step.bar_index


def _fake_step(bar_index, close, signal_action="WAIT", trade_type="wait",
               confidence=0.0, buffer_pct=0.01, signal=True):
    sig = None
    if signal:
        sig = SignalRecord(
            symbol="X", timeframe="5m", timestamp=bar_index,
            artifact_name="x", tokenizer_version="rule.v1",
            action=signal_action, trade_type=trade_type,
            confidence=confidence, suggested_buffer_pct=buffer_pct,
            bounce_prob=confidence if "bounce" in trade_type else 0.0,
            break_prob=confidence if "break" in trade_type or "down" in trade_type else 0.0,
            continuation_prob=0.5,
            next_coarse_id=0, next_fine_id=0,
            predicted_role="support", reason="",
        )
    return ReplayStep(bar_index=bar_index, open_time=bar_index,
                      prediction=None, signal=sig, close=close)


def test_simulator_long_stop_hit():
    steps = [
        _fake_step(0, 100.0, "LONG", "bounce_long", confidence=0.8, buffer_pct=0.02),
        _fake_step(1, 99.5, "WAIT"),
        _fake_step(2, 97.0, "WAIT"),       # below 98 -> stop
        _fake_step(3, 105.0, "WAIT"),
    ]
    trades = simulate(steps, hold_bars=10, min_confidence=0.5)
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long"
    assert t.reason == "stop"
    assert t.return_pct < 0


def test_simulator_short_expiry():
    steps = [_fake_step(0, 100.0, "SHORT", "breakdown_short",
                        confidence=0.8, buffer_pct=0.02)]
    for i in range(1, 30):
        steps.append(_fake_step(i, 100.0 - i * 0.05, "WAIT"))
    trades = simulate(steps, hold_bars=5, min_confidence=0.5)
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "short"
    assert t.reason == "expiry"
    assert t.return_pct > 0


def test_simulator_skips_low_confidence():
    steps = [_fake_step(0, 100.0, "LONG", "bounce_long", confidence=0.3)]
    for i in range(1, 10):
        steps.append(_fake_step(i, 100.0 + i, "WAIT"))
    trades = simulate(steps, hold_bars=5, min_confidence=0.55)
    assert trades == []


def test_simulator_skips_wait_action():
    """Even a high-confidence WAIT must not open a trade."""
    steps = [_fake_step(0, 100.0, "WAIT", "wait", confidence=0.9)]
    for i in range(1, 10):
        steps.append(_fake_step(i, 100.0 + i, "WAIT"))
    trades = simulate(steps, hold_bars=5, min_confidence=0.5)
    assert trades == []


def test_metrics_basic():
    trades = [
        Trade(0, 5, "long", 100, 102, 0.02, "expiry", "LONG", 0.6),
        Trade(6, 10, "long", 102, 100, -0.02, "stop", "LONG", 0.6),
        Trade(11, 15, "short", 100, 99, 0.01, "expiry", "SHORT", 0.7),
    ]
    m = compute_metrics(trades)
    assert m.n_trades == 3
    assert m.hit_rate == pytest.approx(2 / 3)
    assert m.cumulative_return_pct == pytest.approx(0.01, abs=1e-9)
    assert m.n_long == 2
    assert m.n_short == 1
    assert m.n_stop == 1
    assert m.n_expiry == 2
    s = summarize_metrics(m)
    assert "trades=3" in s
