"""Runtime tokenizer.

Loads the rule tokenizer (always pure-Python) and optionally a learned
VQ-VAE checkpoint. Produces the same three-stream representation per
record that the training dataset produces.

Version is PINNED at construction. If a manifest expects
`rule.v1+vqvae.v0+raw.v1`, the runtime tokenizer must report exactly
those versions in its `tokenizer_version()` string. Mismatch is a
fatal error - inference output would be silently wrong.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..schemas.trendline import TrendlineRecord
from ..tokenizer.rule import encode_rule
from ..features.vector import build_feature_vector, FEATURE_VECTOR_DIM
from ..learned.vqvae import HierarchicalVQVAE, VQVAEConfig


class RuntimeTokenizer:
    def __init__(
        self,
        *,
        use_rule: bool = True,
        use_learned: bool = True,
        use_raw: bool = True,
        vqvae_path: Path | str | None = None,
    ):
        self.use_rule = use_rule
        self.use_learned = use_learned
        self.use_raw = use_raw
        self.vqvae: Optional[HierarchicalVQVAE] = None
        self._vqvae_coarse_codes = 0
        self._vqvae_fine_codes = 0

        if use_learned and vqvae_path is not None and Path(vqvae_path).exists():
            state = torch.load(vqvae_path, map_location="cpu", weights_only=False)
            cfg = VQVAEConfig(**state["cfg"])
            model = HierarchicalVQVAE(cfg)
            model.load_state_dict(state["model_state"])
            model.eval()
            self.vqvae = model
            self._vqvae_coarse_codes = cfg.coarse_codes
            self._vqvae_fine_codes = cfg.fine_codes
        elif use_learned:
            print(f"[runtime_tokenizer] WARNING: use_learned=True but vqvae_path "
                  f"missing/not-found ({vqvae_path}); learned tokens will be zeros")

    def tokenizer_version(self) -> str:
        parts = []
        if self.use_rule:
            parts.append("rule.v1")
        if self.use_learned and self.vqvae is not None:
            parts.append("vqvae.v0")
        if self.use_raw:
            parts.append("raw.v1")
        return "+".join(parts) or "none"

    def assert_version_matches(self, expected: str):
        actual = self.tokenizer_version()
        if actual != expected:
            raise ValueError(
                f"runtime tokenizer version mismatch: "
                f"manifest expects '{expected}', tokenizer produces '{actual}'"
            )

    def encode_records(self, records: list[TrendlineRecord]) -> dict:
        """Tokenise N records into the same dict shape the training
        dataset produces (without the price window).

        Returns:
            rule_coarse:    (N,) int64
            rule_fine:      (N,) int64
            learned_coarse: (N,) int64
            learned_fine:   (N,) int64
            raw_feat:       (N, 36) float32
        """
        n = len(records)
        rule_c = np.zeros(n, dtype=np.int64)
        rule_f = np.zeros(n, dtype=np.int64)
        learned_c = np.zeros(n, dtype=np.int64)
        learned_f = np.zeros(n, dtype=np.int64)
        raw = np.zeros((n, FEATURE_VECTOR_DIM), dtype=np.float32)
        if n == 0:
            return {"rule_coarse": rule_c, "rule_fine": rule_f,
                    "learned_coarse": learned_c, "learned_fine": learned_f,
                    "raw_feat": raw}
        for i, r in enumerate(records):
            tok = encode_rule(r)
            rule_c[i] = tok.coarse_token_id
            rule_f[i] = tok.fine_token_id
            raw[i] = build_feature_vector(r)
        if self.vqvae is not None:
            with torch.no_grad():
                feat_t = torch.from_numpy(raw).float()
                c_idx, f_idx = self.vqvae.tokenize(feat_t)
            for i in range(n):
                learned_c[i] = int(c_idx[i].item())
                learned_f[i] = int(f_idx[i].item())
        return {"rule_coarse": rule_c, "rule_fine": rule_f,
                "learned_coarse": learned_c, "learned_fine": learned_f,
                "raw_feat": raw}
