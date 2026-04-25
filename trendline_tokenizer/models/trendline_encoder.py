"""Multi-stream trendline encoder.

A position represents one trendline event. Up to three representations
of that event are fused at the embedding stage:
    rule stream   : (rule_coarse_id, rule_fine_id) -> 2 embeddings, summed
    learned stream: (learned_coarse_id, learned_fine_id) -> same
    raw stream    : 36-dim continuous feature vector -> linear projection

Each stream is layer-normed before averaging across enabled streams.
That keeps any one stream from dominating just because its statistics
have a larger scale.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import FusionConfig


class TrendlineMultiStreamEncoder(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        if cfg.use_rule_tokens:
            self.rule_coarse_emb = nn.Embedding(cfg.rule_coarse_vocab_size, d)
            self.rule_fine_emb = nn.Embedding(cfg.rule_fine_vocab_size, d)
            self.rule_norm = nn.LayerNorm(d)
        else:
            self.rule_coarse_emb = None
            self.rule_fine_emb = None
            self.rule_norm = None

        if cfg.use_learned_tokens:
            self.learned_coarse_emb = nn.Embedding(cfg.learned_coarse_vocab_size, d)
            self.learned_fine_emb = nn.Embedding(cfg.learned_fine_vocab_size, d)
            self.learned_norm = nn.LayerNorm(d)
        else:
            self.learned_coarse_emb = None
            self.learned_fine_emb = None
            self.learned_norm = None

        if cfg.use_raw_features:
            self.raw_proj = nn.Linear(cfg.raw_feat_dim, d)
            self.raw_norm = nn.LayerNorm(d)
        else:
            self.raw_proj = None
            self.raw_norm = None

        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.token_seq_len, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads_token,
            dim_feedforward=d * 4, dropout=cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers_token)
        self.out_norm = nn.LayerNorm(d)

    def _embed_streams(self, batch: dict) -> torch.Tensor:
        embeds: list[torch.Tensor] = []
        cfg = self.cfg
        if self.rule_coarse_emb is not None:
            c = batch["rule_coarse"].clamp(0, cfg.rule_coarse_vocab_size - 1)
            f = batch["rule_fine"].clamp(0, cfg.rule_fine_vocab_size - 1)
            embeds.append(self.rule_norm(self.rule_coarse_emb(c) + self.rule_fine_emb(f)))
        if self.learned_coarse_emb is not None:
            c = batch["learned_coarse"].clamp(0, cfg.learned_coarse_vocab_size - 1)
            f = batch["learned_fine"].clamp(0, cfg.learned_fine_vocab_size - 1)
            embeds.append(self.learned_norm(self.learned_coarse_emb(c) + self.learned_fine_emb(f)))
        if self.raw_proj is not None:
            embeds.append(self.raw_norm(self.raw_proj(batch["raw_feat"])))
        stacked = torch.stack(embeds, dim=0)
        return stacked.mean(dim=0)

    def forward(self, batch: dict, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self._embed_streams(batch)
        h = h + self.pos_emb[:, : h.shape[1]]
        # Fully-padded rows would cause softmax(NaN) inside the Transformer.
        # Unmask the first position of any fully-padded row, then zero the
        # output for those rows so downstream cross-attention treats them
        # as padded anyway (we keep `pad_mask` unchanged for the caller).
        all_pad = pad_mask.all(dim=-1)
        if bool(all_pad.any().item()):
            safe_mask = pad_mask.clone()
            safe_mask[all_pad, 0] = False
        else:
            safe_mask = pad_mask
        h = self.encoder(h, src_key_padding_mask=safe_mask)
        if bool(all_pad.any().item()):
            h = h.masked_fill(all_pad.view(-1, 1, 1), 0.0)
        return self.out_norm(h)
