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
    # Phase 2 heads (per the user's research-grade spec)
    assert out["regime_logits"].shape == (B, cfg.n_regime_classes)
    assert out["pattern_logits"].shape == (B, cfg.n_pattern_classes)
    assert out["invalidation_logits"].shape == (B, cfg.n_invalidation_classes)


def test_heads_phase2_can_be_disabled():
    cfg = FusionConfig(d_model=32, regime_head=False,
                       pattern_head=False, invalidation_head=False)
    heads = MultiTaskHeads(cfg)
    B = 2
    pooled = torch.randn(B, cfg.d_model)
    out = heads(pooled)
    assert "regime_logits" not in out
    assert "pattern_logits" not in out
    assert "invalidation_logits" not in out
    assert "bounce_logits" in out


def test_heads_phase2_default_class_counts():
    cfg = FusionConfig()
    assert cfg.n_regime_classes == 3
    assert cfg.n_pattern_classes == 6
    assert cfg.n_invalidation_classes == 5


def test_heads_buffer_pct_in_constrained_range():
    """multi_task_phase2_v1.json caught the bug: buffer head was unbounded
    so the model emitted negatives (pred_mean=-0.018) while targets are
    clipped to [0, 0.05]. The fix is sigmoid * buffer_max_pct."""
    cfg = FusionConfig(d_model=32, buffer_max_pct=0.05)
    heads = MultiTaskHeads(cfg)
    # Force extreme pre-sigmoid activations to prove the cap holds even
    # for blown-up logits — not just the random init regime.
    pooled = torch.randn(64, cfg.d_model) * 100.0
    out = heads(pooled)
    bp = out["buffer_pct"]
    assert (bp >= 0).all(), f"buffer_pct went negative: min={bp.min().item()}"
    assert (bp <= cfg.buffer_max_pct).all(), \
        f"buffer_pct exceeded cap {cfg.buffer_max_pct}: max={bp.max().item()}"


def test_heads_buffer_pct_respects_custom_max():
    cfg = FusionConfig(d_model=32, buffer_max_pct=0.10)
    heads = MultiTaskHeads(cfg)
    pooled = torch.randn(16, cfg.d_model) * 50.0
    out = heads(pooled)
    assert (out["buffer_pct"] <= 0.10).all()
    assert (out["buffer_pct"] >= 0).all()


# ---------------------------------------------------------------------
# Task 6 - Full TrendlineFusionModel
# ---------------------------------------------------------------------

from trendline_tokenizer.models.full_model import TrendlineFusionModel


def _full_batch(cfg, B):
    return {
        "price": torch.randn(B, cfg.price_seq_len, cfg.price_feat_dim),
        "price_pad": torch.zeros(B, cfg.price_seq_len, dtype=torch.bool),
        "rule_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B, cfg.token_seq_len)),
        "rule_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B, cfg.token_seq_len)),
        "learned_coarse": torch.randint(0, cfg.learned_coarse_vocab_size, (B, cfg.token_seq_len)),
        "learned_fine": torch.randint(0, cfg.learned_fine_vocab_size, (B, cfg.token_seq_len)),
        "raw_feat": torch.randn(B, cfg.token_seq_len, cfg.raw_feat_dim),
        "token_pad": torch.zeros(B, cfg.token_seq_len, dtype=torch.bool),
    }


def test_full_model_forward_returns_all_heads():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    out = model(_full_batch(cfg, 2))
    for k in ("next_coarse_logits", "next_fine_logits", "bounce_logits",
              "break_logits", "continuation_logits", "buffer_pct"):
        assert k in out


def test_full_model_compute_loss_returns_scalar():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    B = 4
    batch = _full_batch(cfg, B)
    targets = {
        "next_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B,)),
        "next_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B,)),
        "bounce": torch.randint(0, 2, (B,)),
        "brk": torch.randint(0, 2, (B,)),
        "cont": torch.randint(0, 2, (B,)),
        "buffer_pct": torch.rand(B) * 0.05,
    }
    total, parts = model.compute_loss(batch, targets)
    assert total.ndim == 0
    assert torch.isfinite(total)
    for k in ("next_coarse_ce", "next_fine_ce", "bounce_ce", "break_ce", "cont_ce", "buffer_mse"):
        assert k in parts


def test_full_model_runs_with_only_one_stream():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1,
                       use_rule_tokens=False, use_learned_tokens=False, use_raw_features=True)
    model = TrendlineFusionModel(cfg)
    B = 2
    batch = _full_batch(cfg, B)
    targets = {
        "next_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B,)),
        "next_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B,)),
        "bounce": torch.randint(0, 2, (B,)),
        "brk": torch.randint(0, 2, (B,)),
        "cont": torch.randint(0, 2, (B,)),
        "buffer_pct": torch.rand(B) * 0.05,
    }
    total, _ = model.compute_loss(batch, targets)
    assert torch.isfinite(total)


def test_full_model_class_weights_default_uniform():
    """Default class_weights buffers are all-ones — equivalent to plain CE."""
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    assert torch.allclose(model._cw_bounce, torch.ones(2))
    assert torch.allclose(model._cw_brk, torch.ones(2))
    assert torch.allclose(model._cw_cont, torch.ones(2))
    assert torch.allclose(model._cw_regime, torch.ones(cfg.n_regime_classes))
    assert torch.allclose(model._cw_pattern, torch.ones(cfg.n_pattern_classes))
    assert torch.allclose(model._cw_invalidation, torch.ones(cfg.n_invalidation_classes))


def test_full_model_set_class_weights_normalises_to_mean_one():
    """Inverse-freq weights normalised so the mean is ~1 (loss-scale stable)."""
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    # mocked inverse-freq for a 90/10 distribution: cls 0 weight low, cls 1 high
    raw = torch.tensor([0.5, 4.5])   # mean = 2.5
    model.set_class_weights({"bounce": raw})
    assert abs(model._cw_bounce.mean().item() - 1.0) < 1e-5
    # ratios preserved: 4.5 / 0.5 = 9 ; after normalisation, same ratio
    assert abs(model._cw_bounce[1].item() / model._cw_bounce[0].item() - 9.0) < 1e-5


def test_full_model_set_class_weights_handles_missing_classes():
    """Smoke runs can have classes with 0 samples; the previous all-classes
    normalisation inverted the weights (gave missing classes huge weight,
    present class near zero — visible in train_fusion smoke output before
    the fix). The new behaviour: normalise over PRESENT (non-zero) classes.
    """
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    # 3-class regime, only middle has data: raw weights [0, 1.5, 0]
    raw = torch.tensor([0.0, 1.5, 0.0])
    model.set_class_weights({"regime": raw})
    cw = model._cw_regime
    # Only the present class (idx 1) gets normalised to 1
    assert abs(cw[1].item() - 1.0) < 1e-5
    # Missing classes stay at 0 (no spurious upweighting)
    assert cw[0].item() == 0.0
    assert cw[2].item() == 0.0


def test_full_model_set_class_weights_rejects_wrong_shape():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    with pytest.raises(ValueError):
        model.set_class_weights({"regime": torch.ones(99)})


def test_full_model_set_class_weights_rejects_unknown_head():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    with pytest.raises(KeyError):
        model.set_class_weights({"nonexistent": torch.ones(2)})


def test_full_model_loss_changes_when_class_weights_change():
    """Sanity: weighted CE produces a different loss than uniform CE on
    the same imbalanced minibatch — proves the buffer is being read."""
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    torch.manual_seed(0)
    B = 16
    batch = _full_batch(cfg, B)
    targets = {
        "next_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B,)),
        "next_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B,)),
        # All zeros -> heavily imbalanced; weight should change loss
        "bounce": torch.zeros(B, dtype=torch.long),
        "brk": torch.randint(0, 2, (B,)),
        "cont": torch.randint(0, 2, (B,)),
        "buffer_pct": torch.rand(B) * 0.05,
        "regime": torch.randint(0, cfg.n_regime_classes, (B,)),
        "pattern": torch.randint(0, cfg.n_pattern_classes, (B,)),
        "invalidation": torch.randint(0, cfg.n_invalidation_classes, (B,)),
    }
    model.eval()
    with torch.no_grad():
        loss_uniform, parts_uniform = model.compute_loss(batch, targets)
    # Now skew the bounce weight heavily — class 1 weight 9x class 0
    model.set_class_weights({"bounce": torch.tensor([1.0, 9.0])})
    with torch.no_grad():
        loss_weighted, parts_weighted = model.compute_loss(batch, targets)
    # bounce_ce should differ now (all targets are class 0; class-0 weight
    # got down-scaled after normalisation so weighted bounce_ce < uniform).
    assert parts_uniform["bounce_ce"] != parts_weighted["bounce_ce"]


def test_full_model_predict_returns_probs_and_ids():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    B = 2
    batch = _full_batch(cfg, B)
    pred = model.predict(batch)
    assert pred["next_coarse_id"].shape == (B,)
    assert pred["next_fine_id"].shape == (B,)
    for k in ("bounce_prob", "break_prob", "continuation_prob"):
        assert pred[k].shape == (B,)
        assert (pred[k] >= 0).all() and (pred[k] <= 1).all()
    assert (pred["suggested_buffer_pct"] >= 0).all()
