"""Phase 3 — autoregressive generation.

Sample multi-step future trendlines from a trained TrendlineFusionModel,
decode each step's (coarse_id, fine_id) into a concrete TrendlineRecord,
and surface the per-step bounce/break/continuation probabilities.

Public API:
    AutoregressiveGenerator(model)
        .generate(batch, n_steps=..., temperature=..., top_k=...)
        -> list[GeneratedStep]
"""
from .autoregressive import AutoregressiveGenerator, GeneratedStep

__all__ = ["AutoregressiveGenerator", "GeneratedStep"]
