"""Time-ordered train/test split. Out-of-sample is the last (1 - train_pct) fraction."""
from __future__ import annotations
import pandas as pd


def split_by_time(timestamps: list[int], train_pct: float = 0.70) -> tuple[list[bool], list[bool]]:
    """Returns (is_train, is_test) — equal-length boolean lists.
    The first `train_pct * len` bars are train; the rest are test.
    """
    if not (0.0 < train_pct < 1.0):
        raise ValueError(f"train_pct must be in (0, 1), got {train_pct}")
    n = len(timestamps)
    cutoff = int(n * train_pct)
    is_train = [i < cutoff for i in range(n)]
    is_test  = [not v for v in is_train]
    return is_train, is_test


def label_split_column(df: pd.DataFrame, train_pct: float = 0.70) -> pd.DataFrame:
    """Returns a NEW DataFrame with a 'split' column ('train' or 'test')."""
    if "timestamp" not in df.columns:
        raise ValueError("label_split_column requires a 'timestamp' column")
    out = df.copy()
    is_train, _ = split_by_time(list(out["timestamp"]), train_pct=train_pct)
    out["split"] = ["train" if t else "test" for t in is_train]
    return out
