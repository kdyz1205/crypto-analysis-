"""Full fusion model: composes the four blocks and exposes
forward / compute_loss / predict."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig
from .price_seq_encoder import PriceSequenceEncoder
from .trendline_encoder import TrendlineMultiStreamEncoder
from .fusion import CrossAttentionFusion
from .heads import MultiTaskHeads
from .losses import LossWeights


class TrendlineFusionModel(nn.Module):
    def __init__(self, cfg: FusionConfig | None = None,
                 weights: LossWeights | None = None):
        super().__init__()
        self.cfg = cfg or FusionConfig()
        self.weights = weights or LossWeights()
        self.price_enc = PriceSequenceEncoder(self.cfg)
        self.token_enc = TrendlineMultiStreamEncoder(self.cfg)
        self.fusion = CrossAttentionFusion(self.cfg)
        self.heads = MultiTaskHeads(self.cfg)
        self.pool_attn = nn.Linear(self.cfg.d_model, 1)

    def _pool(self, fused, pad_mask):
        scores = self.pool_attn(fused).squeeze(-1)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = scores.softmax(dim=-1).unsqueeze(-1)
        return (fused * weights).sum(dim=1)

    def forward(self, batch: dict) -> dict:
        price_h = self.price_enc(batch["price"], batch["price_pad"])
        token_h = self.token_enc(batch, batch["token_pad"])
        fused = self.fusion(price_h, token_h, batch["price_pad"], batch["token_pad"])
        pooled = self._pool(fused, batch["price_pad"])
        return self.heads(pooled)

    def compute_loss(self, batch: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        out = self.forward(batch)
        w = self.weights

        nc = F.cross_entropy(out["next_coarse_logits"], targets["next_coarse"])
        nf = F.cross_entropy(out["next_fine_logits"], targets["next_fine"])
        bo = F.cross_entropy(out["bounce_logits"], targets["bounce"])
        br = F.cross_entropy(out["break_logits"], targets["brk"])
        co = F.cross_entropy(out["continuation_logits"], targets["cont"])
        bf = F.mse_loss(out["buffer_pct"], targets["buffer_pct"])

        total = (w.next_coarse * nc + w.next_fine * nf
                 + w.bounce * bo + w.brk * br
                 + w.cont * co + w.buffer * bf)
        parts = {
            "next_coarse_ce": nc.detach().item(),
            "next_fine_ce": nf.detach().item(),
            "bounce_ce": bo.detach().item(),
            "break_ce": br.detach().item(),
            "cont_ce": co.detach().item(),
            "buffer_mse": bf.detach().item(),
        }

        # Phase 2 multi-task heads — only contribute if BOTH the head is
        # active AND the target is supplied. Lets older datasets without
        # regime/pattern/invalidation labels still train Phase 1.
        if "regime_logits" in out and "regime" in targets:
            rg = F.cross_entropy(out["regime_logits"], targets["regime"])
            total = total + w.regime * rg
            parts["regime_ce"] = rg.detach().item()
        if "pattern_logits" in out and "pattern" in targets:
            pt = F.cross_entropy(out["pattern_logits"], targets["pattern"])
            total = total + w.pattern * pt
            parts["pattern_ce"] = pt.detach().item()
        if "invalidation_logits" in out and "invalidation" in targets:
            iv = F.cross_entropy(out["invalidation_logits"], targets["invalidation"])
            total = total + w.invalidation * iv
            parts["invalidation_ce"] = iv.detach().item()

        parts["total"] = total.detach().item()
        return total, parts

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        self.eval()
        out = self.forward(batch)
        return {
            "next_coarse_id": out["next_coarse_logits"].argmax(dim=-1),
            "next_fine_id": out["next_fine_logits"].argmax(dim=-1),
            "bounce_prob": out["bounce_logits"].softmax(dim=-1)[:, 1],
            "break_prob": out["break_logits"].softmax(dim=-1)[:, 1],
            "continuation_prob": out["continuation_logits"].softmax(dim=-1)[:, 1],
            "suggested_buffer_pct": out["buffer_pct"].clamp(min=0.0),
        }
