"""Tests for the BERT-style mask reconstruction pretraining task."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from trendline_tokenizer.schemas.trendline import TrendlineRecord
from trendline_tokenizer.training.sequence_dataset import (
    SequenceDataset, build_examples,
)
from trendline_tokenizer.training.mask_pretrain import MaskingDataset, MaskReconModel
from trendline_tokenizer.models.config import FusionConfig


def _toy_df(n=200):
    return pd.DataFrame({
        "open_time": np.arange(n) * 60_000 + 1_700_000_000_000,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100.5, 110.5, n),
        "volume": np.full(n, 1.0),
    })


def _toy_record(start_idx, end_idx, role="support") -> TrendlineRecord:
    return TrendlineRecord(
        id=f"r-{start_idx}-{end_idx}",
        symbol="BTCUSDT", exchange="bitget", timeframe="5m",
        start_time=start_idx * 60, end_time=end_idx * 60,
        start_bar_index=start_idx, end_bar_index=end_idx,
        start_price=100.0, end_price=101.0,
        line_role=role, direction="up", touch_count=2,
        bounce_after=True, bounce_strength_atr=1.0,
        break_after=False, retested_after_break=False,
        volatility_atr_pct=0.01, volume_z_score=0.0,
        label_source="auto", auto_method="test", created_at=0,
    )


def _build_base_ds(n_records=12) -> SequenceDataset:
    df = _toy_df()
    records = [_toy_record(20 + 12 * i, 35 + 12 * i) for i in range(n_records)]
    examples = build_examples(df, records, price_seq_len=64, token_seq_len=8,
                              horizon_bars=10, raw_feat_dim=36)
    return SequenceDataset(examples)


def test_masking_dataset_produces_some_masks():
    base = _build_base_ds(n_records=8)
    ds = MaskingDataset(base, mask_prob=0.3, seed=42)
    masked_count = 0
    for i in range(len(ds)):
        s = ds[i]
        if s["mask_positions"].any():
            masked_count += 1
    assert masked_count > 0, "should mask at least some examples"


def test_masking_dataset_only_masks_non_padded_positions():
    base = _build_base_ds(n_records=8)
    ds = MaskingDataset(base, mask_prob=0.5, seed=42)
    for i in range(len(ds)):
        s = ds[i]
        # Mask must be a subset of NON-padded positions
        masked_padded = (s["mask_positions"] & s["token_pad"]).any().item()
        assert not masked_padded, f"example {i}: masked a padded position"


def test_masking_dataset_targets_are_minus100_outside_mask():
    base = _build_base_ds(n_records=8)
    ds = MaskingDataset(base, mask_prob=0.4, seed=42)
    for i in range(len(ds)):
        s = ds[i]
        mask = s["mask_positions"]
        # Where NOT masked, targets must be -100 (CE ignore_index)
        assert (s["masked_rule_coarse"][~mask] == -100).all()
        assert (s["masked_rule_fine"][~mask] == -100).all()
        # Where masked, targets must be valid ids
        assert (s["masked_rule_coarse"][mask] >= 0).all()
        assert (s["masked_rule_fine"][mask] >= 0).all()


def test_masking_dataset_zeros_inputs_at_mask_positions():
    base = _build_base_ds(n_records=8)
    ds = MaskingDataset(base, mask_prob=0.5, seed=42)
    for i in range(len(ds)):
        s = ds[i]
        mask = s["mask_positions"]
        # At masked positions, the input ids should be 0 (replaced)
        if mask.any():
            assert (s["rule_coarse"][mask] == 0).all()
            assert (s["rule_fine"][mask] == 0).all()
            assert (s["raw_feat"][mask] == 0).all()


def test_mask_recon_model_forward_shape():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1,
                       n_layers_fusion=1, price_seq_len=64, token_seq_len=8)
    model = MaskReconModel(cfg)
    base = _build_base_ds(n_records=4)
    ds = MaskingDataset(base, mask_prob=0.5, seed=42)
    sample = ds[0]
    # Build batch
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    out = model(batch)
    assert out["coarse_logits"].shape == (1, cfg.token_seq_len, cfg.rule_coarse_vocab_size)
    assert out["fine_logits"].shape == (1, cfg.token_seq_len, cfg.rule_fine_vocab_size)


def test_mask_recon_model_loss_is_finite():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1,
                       n_layers_fusion=1, price_seq_len=64, token_seq_len=8)
    model = MaskReconModel(cfg)
    base = _build_base_ds(n_records=8)
    ds = MaskingDataset(base, mask_prob=0.5, seed=42)
    # build a batch of 4
    samples = [ds[i] for i in range(4)]
    batch = {k: torch.stack([s[k] for s in samples], dim=0)
             for k in samples[0].keys()}
    loss, parts = model.compute_loss(batch)
    assert torch.isfinite(loss)
    assert "coarse_ce" in parts and "fine_ce" in parts
