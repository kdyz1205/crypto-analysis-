"""Cross-attention fusion: price as Query, tokens as Key/Value.

Each price-bar position learns to attend to the most relevant trendline
events. If the token branch is fully padded, the cross-attention is
short-circuited (skip connection only) so the model still has signal.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import FusionConfig


class _FusionBlock(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.norm_q = nn.LayerNorm(cfg.d_model)
        self.norm_kv = nn.LayerNorm(cfg.d_model)
        self.cross = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads_fusion,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )

    def forward(self, q, kv, kv_pad_mask, kv_all_padded):
        if bool(kv_all_padded.all().item()):
            return q + self.ff(self.norm_ff(q))
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross(q_norm, kv_norm, kv_norm, key_padding_mask=kv_pad_mask)
        if bool(kv_all_padded.any().item()):
            attn_out = attn_out.masked_fill(kv_all_padded.view(-1, 1, 1), 0.0)
        h = q + attn_out
        h = h + self.ff(self.norm_ff(h))
        return h


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([_FusionBlock(cfg) for _ in range(cfg.n_layers_fusion)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, price_h, token_h, price_pad_mask, token_pad_mask):
        kv_all_padded = token_pad_mask.all(dim=-1)
        h = price_h
        for blk in self.blocks:
            h = blk(h, token_h, token_pad_mask, kv_all_padded)
        return self.out_norm(h)
