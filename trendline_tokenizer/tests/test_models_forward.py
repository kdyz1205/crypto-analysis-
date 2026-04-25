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


# ---------------------------------------------------------------------
# Task 3 - TrendlineMultiStreamEncoder
# ---------------------------------------------------------------------

from trendline_tokenizer.models.trendline_encoder import TrendlineMultiStreamEncoder


def _toy_trendline_batch(cfg, B):
    return {
        "rule_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B, cfg.token_seq_len)),
        "rule_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B, cfg.token_seq_len)),
        "learned_coarse": torch.randint(0, cfg.learned_coarse_vocab_size, (B, cfg.token_seq_len)),
        "learned_fine": torch.randint(0, cfg.learned_fine_vocab_size, (B, cfg.token_seq_len)),
        "raw_feat": torch.randn(B, cfg.token_seq_len, cfg.raw_feat_dim),
        "token_pad": torch.zeros(B, cfg.token_seq_len, dtype=torch.bool),
    }


def test_trendline_encoder_all_streams_forward_shape():
    cfg = FusionConfig(d_model=64, n_layers_token=2, n_heads_token=4)
    enc = TrendlineMultiStreamEncoder(cfg)
    B = 3
    batch = _toy_trendline_batch(cfg, B)
    h = enc(batch, batch["token_pad"])
    assert h.shape == (B, cfg.token_seq_len, cfg.d_model)


def test_trendline_encoder_rule_only_ablation():
    cfg = FusionConfig(d_model=32, n_layers_token=1, n_heads_token=2,
                       use_rule_tokens=True, use_learned_tokens=False,
                       use_raw_features=False)
    enc = TrendlineMultiStreamEncoder(cfg)
    B = 2
    batch = _toy_trendline_batch(cfg, B)
    h = enc(batch, batch["token_pad"])
    assert h.shape == (B, cfg.token_seq_len, cfg.d_model)
    assert enc.learned_coarse_emb is None
    assert enc.raw_proj is None


def test_trendline_encoder_raw_only_ablation():
    cfg = FusionConfig(d_model=32, n_layers_token=1, n_heads_token=2,
                       use_rule_tokens=False, use_learned_tokens=False,
                       use_raw_features=True)
    enc = TrendlineMultiStreamEncoder(cfg)
    B = 2
    batch = _toy_trendline_batch(cfg, B)
    h = enc(batch, batch["token_pad"])
    assert h.shape == (B, cfg.token_seq_len, cfg.d_model)


def test_trendline_encoder_clamps_out_of_range_ids():
    cfg = FusionConfig(d_model=32, n_layers_token=1, n_heads_token=2)
    enc = TrendlineMultiStreamEncoder(cfg)
    B = 1
    batch = _toy_trendline_batch(cfg, B)
    batch["rule_coarse"] = batch["rule_coarse"] + cfg.rule_coarse_vocab_size
    h = enc(batch, batch["token_pad"])
    assert torch.isfinite(h).all()


# ---------------------------------------------------------------------
# Task 4 - CrossAttentionFusion
# ---------------------------------------------------------------------

from trendline_tokenizer.models.fusion import CrossAttentionFusion


def test_fusion_forward_shape():
    cfg = FusionConfig(d_model=64, n_layers_fusion=2, n_heads_fusion=4)
    fusion = CrossAttentionFusion(cfg)
    B = 2
    price_h = torch.randn(B, cfg.price_seq_len, cfg.d_model)
    token_h = torch.randn(B, cfg.token_seq_len, cfg.d_model)
    price_pad = torch.zeros(B, cfg.price_seq_len, dtype=torch.bool)
    token_pad = torch.zeros(B, cfg.token_seq_len, dtype=torch.bool)
    fused = fusion(price_h, token_h, price_pad, token_pad)
    assert fused.shape == (B, cfg.price_seq_len, cfg.d_model)


def test_fusion_with_all_token_padding():
    cfg = FusionConfig(d_model=32, n_layers_fusion=1, n_heads_fusion=2)
    fusion = CrossAttentionFusion(cfg)
    B = 1
    price_h = torch.randn(B, cfg.price_seq_len, cfg.d_model)
    token_h = torch.zeros(B, cfg.token_seq_len, cfg.d_model)
    price_pad = torch.zeros(B, cfg.price_seq_len, dtype=torch.bool)
    token_pad = torch.ones(B, cfg.token_seq_len, dtype=torch.bool)
    fused = fusion(price_h, token_h, price_pad, token_pad)
    assert torch.isfinite(fused).all()


# ---------------------------------------------------------------------
# Task 5 - MultiTaskHeads
# ---------------------------------------------------------------------

from trendline_tokenizer.models.heads import MultiTaskHeads


def test_heads_forward_shapes():
    cfg = FusionConfig(d_model=64)
    heads = MultiTaskHeads(cfg)
    B = 4
    pooled = torch.randn(B, cfg.d_model)
    out = heads(pooled)
    assert out["next_coarse_logits"].shape == (B, cfg.rule_coarse_vocab_size)
    assert out["next_fine_logits"].shape == (B, cfg.rule_fine_vocab_size)
    assert out["bounce_logits"].shape == (B, 2)
    assert out["break_logits"].shape == (B, 2)
    assert out["continuation_logits"].shape == (B, 2)
    assert out["buffer_pct"].shape == (B,)
