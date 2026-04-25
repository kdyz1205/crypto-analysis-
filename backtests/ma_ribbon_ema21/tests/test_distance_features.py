from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.distance_features import (
    distance_to_ma_pct,
    distance_bucket,
    DEFAULT_BUCKETS,
)


def test_distance_to_ma_pct_simple():
    close = pd.Series([100.0, 105.0, 110.0])
    ma    = pd.Series([100.0, 100.0, 100.0])
    out = distance_to_ma_pct(close, ma)
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[1] == pytest.approx(0.05)
    assert out.iloc[2] == pytest.approx(0.10)


def test_distance_to_ma_pct_handles_nan_ma():
    close = pd.Series([100.0, 105.0])
    ma    = pd.Series([np.nan, 100.0])
    out = distance_to_ma_pct(close, ma)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.05)


def test_distance_bucket_returns_correct_label():
    assert distance_bucket(0.001, DEFAULT_BUCKETS) == "[0.0%, 0.5%)"
    assert distance_bucket(0.005, DEFAULT_BUCKETS) == "[0.5%, 1.0%)"
    assert distance_bucket(0.025, DEFAULT_BUCKETS) == "[2.0%, 4.0%)"
    assert distance_bucket(0.50,  DEFAULT_BUCKETS) == "[7.0%, 100.0%)"


def test_distance_bucket_negative_returns_outlier():
    assert distance_bucket(-0.01, DEFAULT_BUCKETS) == "<negative>"


def test_distance_bucket_at_lower_edge_starts_that_bucket():
    assert distance_bucket(0.005, DEFAULT_BUCKETS) == "[0.5%, 1.0%)"
