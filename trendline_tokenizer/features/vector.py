"""TrendlineRecord → fixed-size feature vector for the learned tokeniser.

Shape: 17 continuous + 7 role_onehot + 12 tf_onehot = 36 dims.

All continuous fields clipped & normalised to O(1) so the VQ-VAE sees
a well-conditioned input. No look-ahead; everything is derivable from
the record itself.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..schemas.trendline import TrendlineRecord
from ..tokenizer.vocab import LINE_ROLES, TIMEFRAMES


CONTINUOUS_KEYS = [
    "normalized_start_price_shift",   # log(start/end) — relative, scale-free
    "normalized_end_price_shift",     # == -above, kept for redundancy tolerance
    "duration_bars_log",
    "slope",                          # log-slope per bar
    "angle_rad",                      # atan(slope * 100)
    "touch_count_clip",
    "rejection_strength_atr",
    "break_distance_atr",
    "bounce_strength_atr",
    "retested_after_break",
    "volatility_atr_pct",
    "volume_z_score",
    "distance_to_ma20_atr",
    "distance_to_recent_high_atr",
    "distance_to_recent_low_atr",
    "extend_right",
    "extend_left",
]

FEATURE_VECTOR_DIM = len(CONTINUOUS_KEYS) + len(LINE_ROLES) + len(TIMEFRAMES)  # 17 + 7 + 12 = 36


def feature_vector_keys() -> list[str]:
    keys = list(CONTINUOUS_KEYS)
    keys += [f"role_{r}" for r in LINE_ROLES]
    keys += [f"tf_{t}" for t in TIMEFRAMES]
    return keys


def _safe(v, default=0.0):
    if v is None:
        return default
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_feature_vector(r: TrendlineRecord) -> np.ndarray:
    """Produce the 36-dim feature vector for one record."""
    # log-price ratio captures slope direction + magnitude in one scale-free scalar
    if r.start_price > 0 and r.end_price > 0:
        log_ratio = math.log(r.end_price / r.start_price)
    else:
        log_ratio = 0.0
    slope = r.log_slope_per_bar()
    angle = math.atan(slope * 100.0)   # scale so slope of 0.01 → ~45°

    vec = np.array([
        _clip(log_ratio, -1.0, 1.0),                       # normalized_start_price_shift
        _clip(-log_ratio, -1.0, 1.0),                      # normalized_end_price_shift
        math.log1p(max(0, r.duration_bars())),             # duration_bars_log
        _clip(slope, -0.1, 0.1) * 10.0,                    # slope rescaled to ~[-1,1]
        angle / (math.pi / 2),                             # angle_rad in [-1,1]
        _clip(r.touch_count, 0, 8) / 8.0,                  # touch_count_clip
        _clip(_safe(r.rejection_strength_atr), 0, 5) / 5,
        _clip(_safe(r.break_distance_atr), -5, 5) / 5,
        _clip(_safe(r.bounce_strength_atr), 0, 5) / 5,
        1.0 if r.retested_after_break else 0.0,
        _clip(_safe(r.volatility_atr_pct), 0, 0.1) * 10.0,
        _clip(_safe(r.volume_z_score), -3, 3) / 3,
        _clip(_safe(r.distance_to_ma20_atr), -5, 5) / 5,
        _clip(_safe(r.distance_to_recent_high_atr), -5, 5) / 5,
        _clip(_safe(r.distance_to_recent_low_atr), -5, 5) / 5,
        1.0 if r.extend_right else 0.0,
        1.0 if r.extend_left else 0.0,
    ], dtype=np.float32)

    # one-hot role
    role_oh = np.zeros(len(LINE_ROLES), dtype=np.float32)
    try:
        role_oh[LINE_ROLES.index(r.line_role)] = 1.0
    except ValueError:
        pass

    # one-hot timeframe
    tf_oh = np.zeros(len(TIMEFRAMES), dtype=np.float32)
    try:
        tf_oh[TIMEFRAMES.index(r.timeframe)] = 1.0
    except ValueError:
        pass

    return np.concatenate([vec, role_oh, tf_oh], axis=0)


def records_to_tensor(records: Iterable[TrendlineRecord]) -> np.ndarray:
    arrs = [build_feature_vector(r) for r in records]
    if not arrs:
        return np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
    return np.stack(arrs, axis=0)
