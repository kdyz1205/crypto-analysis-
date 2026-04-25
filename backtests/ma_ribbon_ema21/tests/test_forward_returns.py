from __future__ import annotations
import math
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.forward_returns import (
    forward_return,
    forward_returns_at,
    apply_round_trip_cost,
)


def test_forward_return_simple_long():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    r = forward_return(close, entry_idx=1, n_bars=3)
    assert r == pytest.approx((104.0 - 101.0) / 101.0)


def test_forward_return_returns_nan_when_horizon_exceeds_data():
    close = pd.Series([100.0, 101.0, 102.0])
    r = forward_return(close, entry_idx=1, n_bars=10)
    assert math.isnan(r)


def test_forward_returns_at_returns_dict_keyed_by_horizon():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    out = forward_returns_at(close, entry_idx=0, horizons=[1, 3, 5])
    assert out == pytest.approx({
        1: (101 - 100) / 100,
        3: (103 - 100) / 100,
        5: (105 - 100) / 100,
    })


def test_apply_round_trip_cost_subtracts_two_sides_plus_slippage():
    raw_return = 0.05
    cost = apply_round_trip_cost(raw_return, fee_per_side=0.0005, slippage_per_fill=0.0001)
    # 2 fills, each: 0.0005 + 0.0001 = 0.0006 → round trip 0.0012
    assert cost == pytest.approx(0.05 - 0.0012)
