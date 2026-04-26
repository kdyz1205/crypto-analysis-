"""Tests for Phase 3 autoregressive trendline generation."""
from __future__ import annotations
import pytest
import torch

from trendline_tokenizer.generation import AutoregressiveGenerator, GeneratedStep
from trendline_tokenizer.models.config import FusionConfig
from trendline_tokenizer.models.full_model import TrendlineFusionModel
from trendline_tokenizer.schemas.trendline import TrendlineRecord


def _toy_batch(cfg: FusionConfig, B: int = 1) -> dict:
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


def _small_model() -> TrendlineFusionModel:
    cfg = FusionConfig(
        d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1
    )
    return TrendlineFusionModel(cfg)


# ── shapes ────────────────────────────────────────────────────────────

def test_generator_returns_n_steps():
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    out = gen.generate(_toy_batch(model.cfg), n_steps=4, temperature=0.0)
    assert len(out) == 4
    for i, step in enumerate(out):
        assert step.step == i
        assert isinstance(step, GeneratedStep)


def test_generator_zero_steps_returns_empty():
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    assert gen.generate(_toy_batch(model.cfg), n_steps=0) == []


def test_generated_step_fields_in_valid_ranges():
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    out = gen.generate(_toy_batch(model.cfg), n_steps=3, temperature=1.0, seed=42)
    cfg = model.cfg
    for step in out:
        assert 0 <= step.rule_coarse_id < cfg.rule_coarse_vocab_size
        assert 0 <= step.rule_fine_id < cfg.rule_fine_vocab_size
        assert 0.0 <= step.bounce_prob <= 1.0
        assert 0.0 <= step.break_prob <= 1.0
        assert 0.0 <= step.continuation_prob <= 1.0
        # buffer head is sigmoid * cfg.buffer_max_pct
        assert 0.0 <= step.suggested_buffer_pct <= cfg.buffer_max_pct + 1e-6
        assert isinstance(step.decoded_record, TrendlineRecord)
        # top5 diagnostics
        assert len(step.top5_coarse_ids) == 5
        assert len(step.top5_coarse_probs) == 5
        assert all(0 <= p <= 1 for p in step.top5_coarse_probs)


# ── greedy determinism ────────────────────────────────────────────────

def test_generator_greedy_is_deterministic():
    """temperature=0 → argmax → exact same output across runs."""
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    batch = _toy_batch(model.cfg)
    out1 = gen.generate(batch, n_steps=3, temperature=0.0)
    out2 = gen.generate(batch, n_steps=3, temperature=0.0)
    for s1, s2 in zip(out1, out2):
        assert s1.rule_coarse_id == s2.rule_coarse_id
        assert s1.rule_fine_id == s2.rule_fine_id


def test_generator_seeded_sampling_reproduces():
    """seed=K → same multinomial samples."""
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    batch = _toy_batch(model.cfg)
    out1 = gen.generate(batch, n_steps=3, temperature=1.0, seed=123)
    out2 = gen.generate(batch, n_steps=3, temperature=1.0, seed=123)
    for s1, s2 in zip(out1, out2):
        assert s1.rule_coarse_id == s2.rule_coarse_id


def test_generator_different_seeds_diverge():
    """seeded sampling with different seeds should usually differ.
    Not guaranteed in pathological models but for an untrained random
    init the prob mass is spread enough that 3 seeds should differ."""
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    batch = _toy_batch(model.cfg)
    seeds = [1, 7, 99]
    coarse_ids = []
    for s in seeds:
        out = gen.generate(batch, n_steps=3, temperature=1.5, seed=s)
        coarse_ids.append(tuple(st.rule_coarse_id for st in out))
    # At least two of the three seeds should produce different sequences
    assert len(set(coarse_ids)) >= 2


# ── top-k truncation ──────────────────────────────────────────────────

def test_generator_top_k_restricts_sample_set():
    """Force top_k=1 → effectively greedy → same as temperature=0."""
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    batch = _toy_batch(model.cfg)
    greedy = gen.generate(batch, n_steps=3, temperature=0.0)
    top1 = gen.generate(batch, n_steps=3, temperature=1.0, top_k=1, seed=42)
    for g, t in zip(greedy, top1):
        assert g.rule_coarse_id == t.rule_coarse_id
        assert g.rule_fine_id == t.rule_fine_id


# ── context shifting ──────────────────────────────────────────────────

def test_generator_shifts_token_context():
    """After step k, the last position of rule_coarse should equal the
    sampled coarse_id from step k-1 (the just-emitted token feeds the
    next step). This proves the generator actually rolls context."""
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    cfg = model.cfg
    batch = _toy_batch(cfg)
    # Use temperature=0 so the trace is fully deterministic
    out = gen.generate(batch, n_steps=3, temperature=0.0)
    # Manually reproduce one shift to confirm the helper behaves
    coarse_id_step0 = torch.tensor([out[0].rule_coarse_id])
    fine_id_step0 = torch.tensor([out[0].rule_fine_id])
    shifted = gen._shift_in_token(batch, coarse_id_step0, fine_id_step0)
    assert int(shifted["rule_coarse"][0, -1].item()) == out[0].rule_coarse_id
    assert int(shifted["rule_fine"][0, -1].item()) == out[0].rule_fine_id
    # token_pad of the new slot is False (real, not padding)
    assert shifted["token_pad"][0, -1].item() is False or \
           shifted["token_pad"][0, -1].item() == 0
    # learned_coarse fallback to id 0 (unknown)
    assert int(shifted["learned_coarse"][0, -1].item()) == 0
    assert int(shifted["learned_fine"][0, -1].item()) == 0
    # raw_feat fallback to zeros
    assert torch.allclose(shifted["raw_feat"][0, -1, :], torch.zeros(cfg.raw_feat_dim))


# ── decode integration ────────────────────────────────────────────────

def test_decoded_record_carries_reference_symbol():
    """If a reference TrendlineRecord is passed, its symbol/exchange/
    created_at carry into every decoded step."""
    ref = TrendlineRecord(
        id="ref-line", symbol="HYPEUSDT", exchange="bitget", timeframe="1h",
        start_time=1700000000, end_time=1700036000,
        start_bar_index=100, end_bar_index=110,
        start_price=20.0, end_price=20.5,
        line_role="support", direction="up",
        touch_count=3, label_source="auto", created_at=1700000000,
    )
    model = _small_model()
    gen = AutoregressiveGenerator(model)
    out = gen.generate(_toy_batch(model.cfg), n_steps=2, temperature=0.0,
                       reference_record=ref)
    for step in out:
        assert step.decoded_record.symbol == "HYPEUSDT"
        assert step.decoded_record.exchange == "bitget"
        assert step.decoded_record.created_at == 1700000000


# ── pure-helper sampling ──────────────────────────────────────────────

def test_sample_logits_argmax_at_zero_temperature():
    """temperature=0 should equal argmax even with high entropy logits."""
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.3]])
    out = AutoregressiveGenerator._sample_logits(logits, temperature=0.0)
    assert out.item() == 2


def test_sample_logits_top_k_excludes_low_logits():
    """top_k=1 → must pick the argmax. Repeat 100 times to be sure."""
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.3, 0.5]])
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(100):
        out = AutoregressiveGenerator._sample_logits(
            logits, temperature=1.0, top_k=1, rng=rng,
        )
        assert out.item() == 2
