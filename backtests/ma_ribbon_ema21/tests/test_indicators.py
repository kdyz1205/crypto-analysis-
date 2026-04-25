from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.indicators import sma, ema


def test_sma_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, period=3)
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_sma_period_larger_than_series_returns_all_nan():
    s = pd.Series([1.0, 2.0, 3.0])
    out = sma(s, period=5)
    assert out.isna().all()


def test_ema_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(s, period=3)
    # alpha = 2/(3+1) = 0.5; seeded at idx 2 with SMA = 2.0
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    # idx 3 = 0.5 * 4 + 0.5 * 2 = 3.0
    assert out.iloc[3] == pytest.approx(3.0)
    # idx 4 = 0.5 * 5 + 0.5 * 3 = 4.0
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_index_preserved():
    s = pd.Series([10.0, 20.0, 30.0, 40.0], index=[100, 200, 300, 400])
    out = ema(s, period=2)
    assert list(out.index) == [100, 200, 300, 400]
