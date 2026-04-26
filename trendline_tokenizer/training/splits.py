"""Train/val/test split policies for trendline records.

Per institutional spec: random_split is for smoke tests ONLY.
Real validation must use:
  - time_forward: train-on-past, validate-on-future, test-on-tail
  - symbol_heldout: train symbols ∩ test symbols = empty
  - regime_heldout: train on normal/chop, test on crash/bubble/high-vol

Each split returns (train_idx, val_idx, test_idx) — index arrays into
the input record list. Caller materialises whichever subset they need.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ..schemas.trendline import TrendlineRecord


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    policy: str
    metadata: dict


def random_split(
    records: Sequence[TrendlineRecord], *,
    val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42,
) -> SplitResult:
    """SMOKE-ONLY: random-shuffle split. Includes a warning marker so
    audit reports flag this as non-rigorous."""
    n = len(records)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return SplitResult(
        train_idx=np.sort(train_idx),
        val_idx=np.sort(val_idx),
        test_idx=np.sort(test_idx),
        policy="random",
        metadata={"warning": "random_split is SMOKE-ONLY; use time_forward in real runs"},
    )


def time_forward_split(
    records: Sequence[TrendlineRecord], *,
    val_frac: float = 0.15, test_frac: float = 0.15,
) -> SplitResult:
    """Train on oldest fraction, val on next, test on most-recent.

    Sort records by `end_time`. Pick val_start_ts at the (1 - test_frac
    - val_frac) quantile and test_start_ts at the (1 - test_frac)
    quantile.
    """
    if not records:
        return SplitResult(
            train_idx=np.array([], dtype=np.int64),
            val_idx=np.array([], dtype=np.int64),
            test_idx=np.array([], dtype=np.int64),
            policy="time_forward", metadata={},
        )
    times = np.array([int(r.end_time or 0) for r in records])
    order = np.argsort(times, kind="stable")
    n = len(records)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_test - n_val
    if n_train <= 0:
        # Tiny pool: collapse to all-train
        return SplitResult(
            train_idx=order, val_idx=np.array([], dtype=np.int64),
            test_idx=np.array([], dtype=np.int64),
            policy="time_forward",
            metadata={"warning": f"too few records ({n}) for time-forward split"},
        )
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    return SplitResult(
        train_idx=np.sort(train_idx),
        val_idx=np.sort(val_idx),
        test_idx=np.sort(test_idx),
        policy="time_forward",
        metadata={
            "val_start_ts": int(times[order[n_train]]),
            "test_start_ts": int(times[order[n_train + n_val]]),
            "n_records": n,
        },
    )


def symbol_heldout_split(
    records: Sequence[TrendlineRecord], *,
    val_symbols: Optional[list[str]] = None,
    test_symbols: Optional[list[str]] = None,
    val_frac: float = 0.15, test_frac: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Hold out specific symbols for val + test.

    If val/test_symbols not provided, randomly pick symbols whose
    cumulative record count is closest to val_frac / test_frac.
    """
    by_sym: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_sym.setdefault(r.symbol, []).append(i)
    all_syms = sorted(by_sym.keys())

    if val_symbols is None or test_symbols is None:
        rng = np.random.default_rng(seed)
        rng.shuffle(all_syms)
        n_total = len(records)
        target_test = int(n_total * test_frac)
        target_val = int(n_total * val_frac)
        test_set: list[str] = []
        val_set: list[str] = []
        running = 0
        for s in all_syms:
            n_s = len(by_sym[s])
            if running + n_s <= target_test or not test_set:
                test_set.append(s); running += n_s
                if running >= target_test:
                    break
        val_running = 0
        for s in all_syms:
            if s in test_set:
                continue
            n_s = len(by_sym[s])
            if val_running + n_s <= target_val or not val_set:
                val_set.append(s); val_running += n_s
                if val_running >= target_val:
                    break
        val_symbols = val_symbols or val_set
        test_symbols = test_symbols or test_set

    val_set_l = set(val_symbols)
    test_set_l = set(test_symbols)
    train_idx, val_idx, test_idx = [], [], []
    for i, r in enumerate(records):
        if r.symbol in test_set_l:
            test_idx.append(i)
        elif r.symbol in val_set_l:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return SplitResult(
        train_idx=np.array(train_idx, dtype=np.int64),
        val_idx=np.array(val_idx, dtype=np.int64),
        test_idx=np.array(test_idx, dtype=np.int64),
        policy="symbol_heldout",
        metadata={
            "train_symbols": [s for s in all_syms if s not in val_set_l and s not in test_set_l],
            "val_symbols": list(val_symbols),
            "test_symbols": list(test_symbols),
        },
    )


def regime_heldout_split(
    records: Sequence[TrendlineRecord], *,
    test_regime: str = "high_vol",
    seed: int = 42,
) -> SplitResult:
    """Hold out a volatility regime. Records' regime is inferred from
    their `volatility_atr_pct` field:
      < 0.005   -> low_vol
      < 0.015   -> normal_vol
      otherwise -> high_vol

    Only those records ARE actually labelled (some have None). Records
    without volatility metadata go to train.
    """
    def _regime(r: TrendlineRecord) -> str:
        v = r.volatility_atr_pct
        if v is None:
            return "unknown"
        if v < 0.005:
            return "low_vol"
        if v < 0.015:
            return "normal_vol"
        return "high_vol"

    train_idx, test_idx = [], []
    for i, r in enumerate(records):
        if _regime(r) == test_regime:
            test_idx.append(i)
        else:
            train_idx.append(i)
    if not test_idx:
        # Fallback: empty test set is useless; mark as warning
        return SplitResult(
            train_idx=np.array(train_idx, dtype=np.int64),
            val_idx=np.array([], dtype=np.int64),
            test_idx=np.array([], dtype=np.int64),
            policy="regime_heldout",
            metadata={"warning": f"no records with regime={test_regime}"},
        )
    # Carve a small val out of train (random) so trainer has something to monitor
    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(len(train_idx))
    n_val = max(1, int(len(train_idx) * 0.1))
    val_actual = [train_idx[i] for i in train_perm[:n_val]]
    train_actual = [train_idx[i] for i in train_perm[n_val:]]
    return SplitResult(
        train_idx=np.array(train_actual, dtype=np.int64),
        val_idx=np.array(val_actual, dtype=np.int64),
        test_idx=np.array(test_idx, dtype=np.int64),
        policy="regime_heldout",
        metadata={"test_regime": test_regime,
                  "n_train": len(train_actual),
                  "n_val": len(val_actual),
                  "n_test": len(test_idx)},
    )


SPLIT_FNS = {
    "random": random_split,
    "time_forward": time_forward_split,
    "symbol_heldout": symbol_heldout_split,
    "regime_heldout": regime_heldout_split,
}


def split_records(records: Sequence[TrendlineRecord],
                  policy: str = "time_forward", **kwargs) -> SplitResult:
    """Dispatch by policy name. Default time_forward (the rigorous one)."""
    fn = SPLIT_FNS.get(policy)
    if fn is None:
        raise ValueError(f"unknown split policy: {policy} "
                         f"(choices: {list(SPLIT_FNS)})")
    return fn(records, **kwargs)
