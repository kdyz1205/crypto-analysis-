"""Hierarchical VQ-VAE for trendline feature vectors.

Structure (Kronos-analogous, residual quantisation):
    feat(36) → encoder MLP → z(d_model)
    coarse_q  = VQ(256,  d_model, β=0.25)    # coarse market structure
    residual  = z - sg(coarse_q.emb)
    fine_q    = VQ(1024, d_model, β=0.25)    # fine geometry within coarse bin
    decoder_in = coarse_emb + fine_emb
    decoder → recon_feat(36), role_logits(7), bounce_logits(2), break_logits(2)

Losses: L1 recon + CE role + BCE bounce/break + VQ commitment (both)
        + codebook-usage entropy (anti-collapse).

CPU-trainable for the default sizes; GPU-trainable for larger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenizer.vocab import LINE_ROLES, TIMEFRAMES
from ..features.vector import FEATURE_VECTOR_DIM


@dataclass
class VQVAEConfig:
    feat_dim: int = FEATURE_VECTOR_DIM       # 36
    d_model: int = 64
    coarse_codes: int = 256
    fine_codes: int = 1024
    commitment_beta: float = 0.25
    entropy_weight: float = 0.01
    role_weight: float = 0.5
    bounce_weight: float = 0.3
    break_weight: float = 0.3
    recon_weight: float = 1.0
    hidden: int = 128
    dropout: float = 0.1
    ema_codebook: bool = True                # EMA codebook update (more stable than loss-based)
    ema_decay: float = 0.99


class VectorQuantizer(nn.Module):
    """VQ with EMA codebook + commitment loss + dead-code revival + kmeans init."""
    def __init__(self, num_codes: int, dim: int, *, beta: float = 0.25,
                 ema: bool = True, decay: float = 0.99,
                 dead_code_threshold: float = 1.0,
                 revive_every: int = 200):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.ema = ema
        self.decay = decay
        self.dead_code_threshold = dead_code_threshold
        self.revive_every = revive_every

        emb = torch.randn(num_codes, dim) * (1.0 / num_codes ** 0.5)
        self.register_buffer("codebook", emb.clone())
        if ema:
            self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
            self.register_buffer("_ema_weight", emb.clone())
        else:
            self.codebook = nn.Parameter(emb.clone())   # type: ignore[assignment]
        self.register_buffer("_step", torch.zeros(1))
        self.register_buffer("_initialised", torch.zeros(1))

    @torch.no_grad()
    def kmeans_init(self, z: torch.Tensor):
        """Seed the codebook by random sampling from a batch of encoder
        outputs. Beats random-init for 10x+ codebook utilisation on
        small datasets."""
        n = z.shape[0]
        if n == 0:
            return
        if n >= self.num_codes:
            idx = torch.randperm(n, device=z.device)[: self.num_codes]
            picks = z[idx]
        else:
            # repeat with jitter so we still fill the codebook
            picks = z.repeat((self.num_codes // n) + 1, 1)[: self.num_codes]
            picks = picks + 0.01 * torch.randn_like(picks)
        self.codebook.copy_(picks.detach())
        if self.ema:
            self._ema_weight.copy_(picks.detach())
            self._ema_cluster_size.fill_(1.0)
        self._initialised.fill_(1.0)

    @torch.no_grad()
    def _revive_dead_codes(self, z: torch.Tensor, cluster_size: torch.Tensor):
        """Any code whose EMA usage < threshold is replaced by a random
        row from the current batch. Prevents codebook collapse."""
        if not self.ema:
            return
        dead = cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        pool = z.detach()
        if pool.shape[0] == 0:
            return
        sel = torch.randint(0, pool.shape[0], (n_dead,), device=pool.device)
        new = pool[sel]
        new_codes = self.codebook.clone()
        new_codes[dead] = new
        self.codebook.copy_(new_codes)
        new_w = self._ema_weight.clone()
        new_w[dead] = new
        self._ema_weight.copy_(new_w)
        new_cs = self._ema_cluster_size.clone()
        new_cs[dead] = 1.0
        self._ema_cluster_size.copy_(new_cs)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """z: (B, dim). Returns (quantised, indices, commit_loss, aux)."""
        # distances
        # (B, num_codes) = ||z||² + ||e||² - 2 z·e
        z2 = (z ** 2).sum(dim=-1, keepdim=True)
        e2 = (self.codebook ** 2).sum(dim=-1).unsqueeze(0)
        ze = z @ self.codebook.t()
        dist = z2 + e2 - 2.0 * ze
        indices = dist.argmin(dim=-1)
        one_hot = F.one_hot(indices, num_classes=self.num_codes).float()

        quantised = one_hot @ self.codebook

        # EMA codebook update + dead-code revival
        if self.ema and self.training:
            with torch.no_grad():
                n = one_hot.sum(dim=0)
                self._ema_cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                dw = one_hot.t() @ z
                self._ema_weight.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                total = self._ema_cluster_size.sum()
                smoothed = (self._ema_cluster_size + 1e-5) / (total + self.num_codes * 1e-5) * total
                self.codebook.copy_(self._ema_weight / smoothed.unsqueeze(1))
                self._step += 1
                if int(self._step.item()) % self.revive_every == 0:
                    self._revive_dead_codes(z, self._ema_cluster_size)

        # straight-through estimator
        quantised_st = z + (quantised - z).detach()

        # commitment loss
        commit = self.beta * F.mse_loss(z, quantised.detach())
        if not self.ema:
            commit = commit + F.mse_loss(quantised, z.detach())

        # usage entropy (higher = more diverse use of codes)
        usage = one_hot.mean(dim=0)
        usage = usage + 1e-10
        entropy = -(usage * usage.log()).sum()

        aux = {
            "codebook_entropy": entropy.detach(),
            "codes_used": (one_hot.sum(dim=0) > 0).sum().detach(),
        }
        return quantised_st, indices, commit, aux


class Encoder(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.d_model),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(),
        )
        self.feat_head = nn.Linear(cfg.hidden, cfg.feat_dim)
        self.role_head = nn.Linear(cfg.hidden, len(LINE_ROLES))
        self.bounce_head = nn.Linear(cfg.hidden, 2)
        self.break_head = nn.Linear(cfg.hidden, 2)

    def forward(self, z):
        h = self.trunk(z)
        return {
            "feat": self.feat_head(h),
            "role_logits": self.role_head(h),
            "bounce_logits": self.bounce_head(h),
            "break_logits": self.break_head(h),
        }


class HierarchicalVQVAE(nn.Module):
    def __init__(self, cfg: VQVAEConfig | None = None):
        super().__init__()
        cfg = cfg or VQVAEConfig()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.coarse_q = VectorQuantizer(cfg.coarse_codes, cfg.d_model,
                                        beta=cfg.commitment_beta,
                                        ema=cfg.ema_codebook, decay=cfg.ema_decay)
        self.fine_q = VectorQuantizer(cfg.fine_codes, cfg.d_model,
                                      beta=cfg.commitment_beta,
                                      ema=cfg.ema_codebook, decay=cfg.ema_decay)
        self.decoder = Decoder(cfg)

    def forward(self, feat: torch.Tensor) -> dict:
        z = self.encoder(feat)
        # lazy kmeans init on first training batch, before the first VQ call
        if self.training and int(self.coarse_q._initialised.item()) == 0:
            self.coarse_q.kmeans_init(z)
        coarse_q, coarse_idx, commit_c, aux_c = self.coarse_q(z)
        residual = z - coarse_q.detach()
        if self.training and int(self.fine_q._initialised.item()) == 0:
            self.fine_q.kmeans_init(residual)
        fine_q, fine_idx, commit_f, aux_f = self.fine_q(residual)
        decoded_in = coarse_q + fine_q
        out = self.decoder(decoded_in)
        out.update({
            "z": z,
            "coarse_idx": coarse_idx,
            "fine_idx": fine_idx,
            "coarse_commit": commit_c,
            "fine_commit": commit_f,
            "coarse_entropy": aux_c["codebook_entropy"],
            "fine_entropy": aux_f["codebook_entropy"],
            "coarse_codes_used": aux_c["codes_used"],
            "fine_codes_used": aux_f["codes_used"],
        })
        return out

    def compute_loss(self, feat: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        """target dict:
            role_idx:   (B,) long
            bounce_lbl: (B,) long (0/1)
            break_lbl:  (B,) long (0/1)
        """
        out = self.forward(feat)
        cfg = self.cfg
        loss_recon = F.l1_loss(out["feat"], feat)
        loss_role = F.cross_entropy(out["role_logits"], target["role_idx"])
        loss_bounce = F.cross_entropy(out["bounce_logits"], target["bounce_lbl"])
        loss_break = F.cross_entropy(out["break_logits"], target["break_lbl"])
        loss_commit = out["coarse_commit"] + out["fine_commit"]
        loss_entropy = -(out["coarse_entropy"] + out["fine_entropy"]) * cfg.entropy_weight
        total = (cfg.recon_weight * loss_recon
                 + cfg.role_weight * loss_role
                 + cfg.bounce_weight * loss_bounce
                 + cfg.break_weight * loss_break
                 + loss_commit + loss_entropy)
        metrics = {
            "loss": total.detach().item(),
            "recon": loss_recon.detach().item(),
            "role_ce": loss_role.detach().item(),
            "bounce_ce": loss_bounce.detach().item(),
            "break_ce": loss_break.detach().item(),
            "commit": loss_commit.detach().item(),
            "coarse_codes_used": int(out["coarse_codes_used"].item()),
            "fine_codes_used": int(out["fine_codes_used"].item()),
        }
        return total, metrics

    @torch.no_grad()
    def tokenize(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(feat)
        _, coarse_idx, _, _ = self.coarse_q(z)
        residual = z - self.coarse_q.codebook[coarse_idx]
        _, fine_idx, _, _ = self.fine_q(residual)
        return coarse_idx, fine_idx
