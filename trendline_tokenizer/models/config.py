"""Fusion model configuration. Versioned; bumping invalidates checkpoints.

Trendline branch supports THREE parallel streams:
  - rule tokens   (5040 coarse x 21600 fine, from tokenizer/rule.py)
  - learned tokens (256 coarse x 1024 fine, from learned/vqvae.py)
  - raw features  (36-dim continuous vector, from features/vector.py)

Each stream can be disabled for ablation. Default: all three on.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..tokenizer.vocab import coarse_vocab_size, fine_vocab_size
from ..features.vector import FEATURE_VECTOR_DIM
from ..learned.vqvae import VQVAEConfig


class FusionConfig(BaseModel):
    version: str = "fusion.v0.1"

    # trendline stream switches
    use_rule_tokens: bool = True
    use_learned_tokens: bool = True
    use_raw_features: bool = True

    # trendline stream sizes (locked to existing artifacts)
    rule_coarse_vocab_size: int = Field(default_factory=coarse_vocab_size)
    rule_fine_vocab_size: int = Field(default_factory=fine_vocab_size)
    learned_coarse_vocab_size: int = VQVAEConfig().coarse_codes
    learned_fine_vocab_size: int = VQVAEConfig().fine_codes
    raw_feat_dim: int = FEATURE_VECTOR_DIM
    vqvae_checkpoint_path: Optional[str] = "checkpoints/trendline_tokenizer/vqvae_v0.pt"

    # price branch
    n_indicators: int = 8
    price_feat_dim: int = 13
    price_seq_len: int = 256
    n_layers_price: int = 4
    n_heads_price: int = 4

    # trendline branch
    token_seq_len: int = 32
    n_layers_token: int = 3
    n_heads_token: int = 4

    # fusion
    d_model: int = 128
    n_layers_fusion: int = 2
    n_heads_fusion: int = 4
    dropout: float = 0.1

    # heads — Phase 1 supervised (already in production)
    next_token_head: bool = True
    bounce_head: bool = True
    break_head: bool = True
    continuation_head: bool = True
    buffer_head: bool = True
    # heads — Phase 2 multi-task (per the user's research-grade spec)
    # regime: low_vol / normal_vol / high_vol  (from volatility_atr_pct)
    # pattern: channel / triangle / wedge / parallel_same / diverging / unrelated
    # invalidation: valid / weak_pen / confirmed_break / break_retest / failed_breakout
    regime_head: bool = True
    n_regime_classes: int = 3
    pattern_head: bool = True
    n_pattern_classes: int = 6
    invalidation_head: bool = True
    n_invalidation_classes: int = 5

    def n_streams(self) -> int:
        return (int(self.use_rule_tokens)
                + int(self.use_learned_tokens)
                + int(self.use_raw_features))

    def model_post_init(self, _ctx):
        if self.price_feat_dim != 5 + self.n_indicators:
            raise ValueError(
                f"price_feat_dim ({self.price_feat_dim}) must equal "
                f"5 + n_indicators ({5 + self.n_indicators})"
            )
        if self.n_streams() == 0:
            raise ValueError(
                "at least one trendline stream must be enabled "
                "(use_rule_tokens / use_learned_tokens / use_raw_features)"
            )
