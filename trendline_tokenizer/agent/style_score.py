"""Style similarity scorer.

Computes the cosine similarity of a candidate's 36-dim feature vector
to the centroid of the user's hand-drawn lines. Used by auto_drawer
to prefer candidates that look like the user's drawing style.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np

from ..adapters.manual import load_manual_records
from ..features.vector import build_feature_vector, FEATURE_VECTOR_DIM
from ..registry.paths import ROOT
from ..schemas.trendline import TrendlineRecord


_MANUAL_PATH = ROOT / "data" / "manual_trendlines.json"


class StyleScorer:
    """Scores candidates 0..1 by similarity to user manual centroid."""

    def __init__(self, manual_records: list[TrendlineRecord] | None = None):
        if manual_records is None:
            manual_records = load_manual_records(_MANUAL_PATH)
        self.n_manual = len(manual_records)
        if not manual_records:
            self.centroid = np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)
            self.centroid_norm = 1.0
            return
        feats = np.stack([build_feature_vector(r) for r in manual_records], axis=0)
        self.centroid = feats.mean(axis=0)
        self.centroid_norm = float(np.linalg.norm(self.centroid)) or 1.0

    def score_one(self, record: TrendlineRecord) -> float:
        """Cosine similarity in [0, 1] (clipped from [-1, 1])."""
        if self.n_manual == 0:
            return 0.5
        v = build_feature_vector(record)
        nv = float(np.linalg.norm(v)) or 1.0
        cos = float(np.dot(v, self.centroid) / (nv * self.centroid_norm))
        # Map [-1, 1] -> [0, 1]
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))

    def score_batch(self, records: Iterable[TrendlineRecord]) -> list[float]:
        return [self.score_one(r) for r in records]

    def filter_by_style(self, records: list[TrendlineRecord],
                        min_score: float = 0.55) -> list[TrendlineRecord]:
        scores = self.score_batch(records)
        return [r for r, s in zip(records, scores) if s >= min_score]
