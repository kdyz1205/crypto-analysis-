"""Bullish MA-ribbon alignment detection + formation event extraction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignmentConfig:
    require_close_above_ma5:   bool = True
    require_close_above_ma8:   bool = True
    require_close_above_ema21: bool = True
    require_close_above_ma55:  bool = True
    require_ma5_above_ma8:     bool = True
    require_ma8_above_ema21:   bool = True
    require_ema21_above_ma55:  bool = True

    @staticmethod
    def default() -> "AlignmentConfig":
        return AlignmentConfig()

    @staticmethod
    def from_dict(d: dict) -> "AlignmentConfig":
        return AlignmentConfig(**{k: bool(v) for k, v in d.items()})


def bullish_aligned(df: pd.DataFrame, cfg: AlignmentConfig) -> pd.Series:
    """Return a bool Series aligned with df.index where bullish ribbon holds.

    Required columns: close, ma5, ma8, ema21, ma55.
    NaN in any required column → False at that bar (no spurious truth).
    """
    required = ["close", "ma5", "ma8", "ema21", "ma55"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"bullish_aligned: df missing required columns {missing}")

    close = df["close"]
    ma5   = df["ma5"]
    ma8   = df["ma8"]
    ema21 = df["ema21"]
    ma55  = df["ma55"]

    aligned = pd.Series(True, index=df.index)
    if cfg.require_close_above_ma5:
        aligned &= close > ma5
    if cfg.require_close_above_ma8:
        aligned &= close > ma8
    if cfg.require_close_above_ema21:
        aligned &= close > ema21
    if cfg.require_close_above_ma55:
        aligned &= close > ma55
    if cfg.require_ma5_above_ma8:
        aligned &= ma5 > ma8
    if cfg.require_ma8_above_ema21:
        aligned &= ma8 > ema21
    if cfg.require_ema21_above_ma55:
        aligned &= ema21 > ma55

    nan_mask = (
        close.isna() | ma5.isna() | ma8.isna() | ema21.isna() | ma55.isna()
    )
    aligned = aligned.fillna(False)
    aligned[nan_mask] = False

    return aligned.astype(bool)


def formation_events(aligned: pd.Series) -> pd.Index:
    """Return integer-position indices where aligned transitions False → True.

    The first bar is never counted as an event (no prior False to compare).
    """
    a = aligned.to_numpy(dtype=bool)
    n = len(a)
    if n == 0:
        return pd.Index([], dtype=int)
    prev = np.concatenate(([False], a[:-1]))
    transitions = a & ~prev
    transitions[0] = False  # first bar can never be a "fresh formation"
    return pd.Index(np.where(transitions)[0], dtype=int)
