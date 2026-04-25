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
