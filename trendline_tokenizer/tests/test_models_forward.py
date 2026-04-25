"""Tests for the fusion model: forward shapes, config, multi-stream behaviour."""
import pytest
import torch

from trendline_tokenizer.models.config import FusionConfig


# ---------------------------------------------------------------------
# Task 1 — FusionConfig
# ---------------------------------------------------------------------

def test_fusion_config_defaults_match_existing_tokenizers():
    cfg = FusionConfig()
    # rule tokenizer vocab from tokenizer/vocab.py
    assert cfg.rule_coarse_vocab_size == 5040
    assert cfg.rule_fine_vocab_size == 21600
    # learned VQ-VAE vocab from learned/vqvae.py defaults
    assert cfg.learned_coarse_vocab_size == 256
    assert cfg.learned_fine_vocab_size == 1024
    # raw feature dim from features/vector.py
    assert cfg.raw_feat_dim == 36
    # all 3 streams on by default
    assert cfg.use_rule_tokens is True
    assert cfg.use_learned_tokens is True
    assert cfg.use_raw_features is True
    assert cfg.n_streams() == 3
    # OHLCV (5) + N_indicators must equal price_feat_dim
    assert cfg.price_feat_dim == 5 + cfg.n_indicators
    assert cfg.price_seq_len == 256
    assert cfg.token_seq_len == 32


def test_fusion_config_loads_from_dict_with_ablation():
    raw = {"d_model": 64, "n_layers_price": 2,
           "use_learned_tokens": False, "use_raw_features": False}
    cfg = FusionConfig(**raw)
    assert cfg.d_model == 64
    assert cfg.n_streams() == 1


def test_fusion_config_rejects_zero_streams():
    with pytest.raises(ValueError):
        FusionConfig(use_rule_tokens=False, use_learned_tokens=False,
                     use_raw_features=False)


# ---------------------------------------------------------------------
# Task 2 - PriceSequenceEncoder
# ---------------------------------------------------------------------

from trendline_tokenizer.models.price_seq_encoder import PriceSequenceEncoder


def test_price_seq_encoder_forward_shape():
    cfg = FusionConfig(d_model=64, n_layers_price=2, n_heads_price=4)
    enc = PriceSequenceEncoder(cfg)
    B, T, F = 2, cfg.price_seq_len, cfg.price_feat_dim
    x = torch.randn(B, T, F)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    h = enc(x, pad_mask)
    assert h.shape == (B, T, cfg.d_model)


def test_price_seq_encoder_handles_padding():
    cfg = FusionConfig(d_model=64, n_layers_price=2, n_heads_price=4)
    enc = PriceSequenceEncoder(cfg)
    B, T, F = 2, cfg.price_seq_len, cfg.price_feat_dim
    x = torch.randn(B, T, F)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    pad_mask[0, -10:] = True
    h = enc(x, pad_mask)
    assert torch.isfinite(h).all()
