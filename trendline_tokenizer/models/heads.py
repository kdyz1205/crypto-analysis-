"""Multi-task heads attached to the pooled fusion output.

next_coarse / next_fine: predict the next trendline event's RULE tokens
                         (the deterministic, explainable id space)
bounce / break / continuation: 2-class classification at horizon H
buffer_pct: regression - recommended SL buffer in % of price
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import FusionConfig


class MultiTaskHeads(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.d_model
        # Phase 1: next-token + bounce/break/cont + buffer
        self.next_coarse = nn.Linear(h, cfg.rule_coarse_vocab_size) if cfg.next_token_head else None
        self.next_fine = nn.Linear(h, cfg.rule_fine_vocab_size) if cfg.next_token_head else None
        self.bounce = nn.Linear(h, 2) if cfg.bounce_head else None
        self.brk = nn.Linear(h, 2) if cfg.break_head else None
        self.cont = nn.Linear(h, 2) if cfg.continuation_head else None
        self.buffer = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)) if cfg.buffer_head else None
        self._buffer_max_pct = cfg.buffer_max_pct
        # Phase 2: regime + pattern + invalidation
        self.regime = nn.Linear(h, cfg.n_regime_classes) if cfg.regime_head else None
        self.pattern = nn.Linear(h, cfg.n_pattern_classes) if cfg.pattern_head else None
        self.invalidation = nn.Linear(h, cfg.n_invalidation_classes) if cfg.invalidation_head else None

    def forward(self, pooled: torch.Tensor) -> dict:
        out: dict[str, torch.Tensor] = {}
        if self.next_coarse is not None:
            out["next_coarse_logits"] = self.next_coarse(pooled)
            out["next_fine_logits"] = self.next_fine(pooled)
        if self.bounce is not None:
            out["bounce_logits"] = self.bounce(pooled)
        if self.brk is not None:
            out["break_logits"] = self.brk(pooled)
        if self.cont is not None:
            out["continuation_logits"] = self.cont(pooled)
        if self.buffer is not None:
            # Targets are clipped to [0, buffer_max_pct] in
            # sequence_dataset._buffer_pct; constrain output to the same
            # range so MSE doesn't pull toward the unbounded space and
            # produce negatives. Verified by multi_task_phase2_v1.
            raw = self.buffer(pooled).squeeze(-1)
            out["buffer_pct"] = torch.sigmoid(raw) * self._buffer_max_pct
        # Phase 2 heads (per the user's spec)
        if self.regime is not None:
            out["regime_logits"] = self.regime(pooled)
        if self.pattern is not None:
            out["pattern_logits"] = self.pattern(pooled)
        if self.invalidation is not None:
            out["invalidation_logits"] = self.invalidation(pooled)
        return out
