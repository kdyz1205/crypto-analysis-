"""Transformer encoder over OHLCV + indicators.

Input:  (B, T, price_feat_dim) - 5 OHLCV columns + n_indicators
Output: (B, T, d_model) - per-bar embedding

Causal masking is NOT applied here; the bar at position t may attend
to t' < t and t' > t within the same window because the window itself
is the model's "now" view. No-lookahead is enforced at the dataset
level (only bars with end_time <= prediction time are included).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import FusionConfig


class PriceSequenceEncoder(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.price_feat_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.price_seq_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads_price,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers_price)
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F). pad_mask: (B, T) bool (True where padded)."""
        h = self.input_proj(x) + self.pos_emb[:, : x.shape[1]]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.out_norm(h)
