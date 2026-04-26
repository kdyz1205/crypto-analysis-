"""Loss weights for the multi-task fusion model. Centralised so a
single place tunes how heads trade off."""
from __future__ import annotations
from pydantic import BaseModel


class LossWeights(BaseModel):
    next_coarse: float = 1.0
    next_fine: float = 0.5
    bounce: float = 0.5
    brk: float = 0.5
    cont: float = 0.3
    buffer: float = 0.2
    # Phase 2 multi-task heads (per the user's research-grade spec)
    regime: float = 0.3
    pattern: float = 0.3       # only used when pattern targets are supplied
    invalidation: float = 0.4
