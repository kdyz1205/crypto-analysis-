"""PyTorch Dataset wrapping the legacy adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..adapters import iter_legacy_pattern_records, load_manual_records
from ..features.vector import build_feature_vector
from ..schemas.trendline import TrendlineRecord
from ..tokenizer.vocab import LINE_ROLES


class TrendlineFeatureDataset(Dataset):
    """Materialised (feature_vector, role_idx, bounce_lbl, break_lbl) tuples."""
    def __init__(self, records: list[TrendlineRecord]):
        feats = np.stack([build_feature_vector(r) for r in records], axis=0) if records else np.zeros((0, 36), dtype=np.float32)
        role_idx = np.array([LINE_ROLES.index(r.line_role) if r.line_role in LINE_ROLES else LINE_ROLES.index("unknown")
                             for r in records], dtype=np.int64)
        bounce = np.array([1 if r.bounce_after else 0 for r in records], dtype=np.int64)
        brk = np.array([1 if r.break_after else 0 for r in records], dtype=np.int64)
        self.feat = torch.from_numpy(feats)
        self.role_idx = torch.from_numpy(role_idx)
        self.bounce_lbl = torch.from_numpy(bounce)
        self.break_lbl = torch.from_numpy(brk)
        self.records = records

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        return {
            "feat": self.feat[idx],
            "role_idx": self.role_idx[idx],
            "bounce_lbl": self.bounce_lbl[idx],
            "break_lbl": self.break_lbl[idx],
        }


def build_dataset(
    *,
    patterns_dir: Path,
    manual_json: Optional[Path] = None,
    limit: Optional[int] = None,
) -> tuple[TrendlineFeatureDataset, list[TrendlineRecord]]:
    """Load auto + manual records into one dataset. Returns (dataset,
    raw_records_in_order)."""
    recs: list[TrendlineRecord] = []
    if manual_json and Path(manual_json).exists():
        for r in load_manual_records(manual_json):
            recs.append(r)
    if patterns_dir.exists():
        for f in sorted(Path(patterns_dir).glob("*.jsonl")):
            for r in iter_legacy_pattern_records(f):
                recs.append(r)
                if limit and len(recs) >= limit:
                    break
            if limit and len(recs) >= limit:
                break
    ds = TrendlineFeatureDataset(recs)
    return ds, recs
