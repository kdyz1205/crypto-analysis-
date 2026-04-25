"""Distance-from-MA features and bucketing."""
from __future__ import annotations
import numpy as np
import pandas as pd


# (lower_inclusive, upper_exclusive) in decimal form.
DEFAULT_BUCKETS: list[tuple[float, float]] = [
    (0.000, 0.005),
    (0.005, 0.010),
    (0.010, 0.020),
    (0.020, 0.040),
    (0.040, 0.070),
    (0.070, 1.000),
]


def distance_to_ma_pct(close: pd.Series, ma: pd.Series) -> pd.Series:
    """Returns (close - ma) / ma per bar, NaN where ma is NaN or zero."""
    ma_safe = ma.where(ma.abs() > 1e-12, np.nan)
    return (close - ma_safe) / ma_safe


def distance_bucket(distance: float, buckets: list[tuple[float, float]]) -> str:
    """Map a single distance value to a string bucket label.
    Negative distances return "<negative>". Above-max returns "<above_max>"."""
    if distance < 0:
        return "<negative>"
    for lo, hi in buckets:
        if lo <= distance < hi:
            return _label(lo, hi)
    return "<above_max>"


def _label(lo: float, hi: float) -> str:
    return f"[{lo*100:.1f}%, {hi*100:.1f}%)"
