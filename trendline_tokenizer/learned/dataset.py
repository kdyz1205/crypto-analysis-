"""PyTorch Dataset wrapping the legacy adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..adapters import (
    iter_legacy_pattern_records,
    load_manual_records,
    enrich_records_with_outcomes,
)
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
    outcomes_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    ml_events_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> tuple[TrendlineFeatureDataset, list[TrendlineRecord]]:
    """Load auto + (optionally outcome-enriched) manual records.

    Priority:
      1. If outcomes_path + labels_path provided, manual records are
         enriched with real trade outcomes as bounce/break labels. This
         makes them MUCH more valuable as supervision than the weak
         legacy pattern flags.
      2. Auto rows from data/patterns/*.jsonl are appended afterwards.
    """
    recs: list[TrendlineRecord] = []
    if manual_json and Path(manual_json).exists():
        manual_recs = load_manual_records(manual_json)
        if outcomes_path and labels_path and Path(outcomes_path).exists():
            manual_recs = enrich_records_with_outcomes(
                manual_recs,
                outcomes_path=Path(outcomes_path),
                labels_path=Path(labels_path),
                ml_path=Path(ml_events_path) if ml_events_path else None,
            )
        recs.extend(manual_recs)
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


def build_gold_eval_dataset(
    *,
    manual_json: Path,
    outcomes_path: Path,
    labels_path: Path,
    ml_events_path: Optional[Path] = None,
) -> tuple[TrendlineFeatureDataset, list[TrendlineRecord]]:
    """Gold eval set: ONLY the outcome-enriched manual records whose
    outcomes are known (ever_filled). Bounce/break labels come from
    real trade PnL, not legacy weak flags."""
    raw = load_manual_records(manual_json)
    enriched = enrich_records_with_outcomes(
        raw,
        outcomes_path=Path(outcomes_path),
        labels_path=Path(labels_path),
        ml_path=Path(ml_events_path) if ml_events_path else None,
    )
    # Keep only lines that actually have outcome info (bounce_after or break_after set)
    gold = [r for r in enriched if (r.bounce_after is not None or r.break_after is not None)]
    return TrendlineFeatureDataset(gold), gold
