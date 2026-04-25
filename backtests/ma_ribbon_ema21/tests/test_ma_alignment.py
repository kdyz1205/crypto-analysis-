from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.ma_alignment import (
    bullish_aligned,
    formation_events,
    AlignmentConfig,
)


def _make_aligned_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "close": [100, 100, 100, 100, 110],
        "ma5":   [100, 100, 100, 100, 105],
        "ma8":   [100, 100, 100, 100, 102],
        "ema21": [100, 100, 100, 100, 101],
        "ma55":  [100, 100, 100, 100,  99],
    })


def test_bullish_aligned_returns_bool_series_aligned_with_input():
    df = _make_aligned_frame()
    cfg = AlignmentConfig.default()
    aligned = bullish_aligned(df, cfg)
    assert isinstance(aligned, pd.Series)
    assert aligned.dtype == bool
    assert list(aligned.index) == list(df.index)


def test_bullish_aligned_only_last_bar_true():
    df = _make_aligned_frame()
    aligned = bullish_aligned(df, AlignmentConfig.default())
    assert list(aligned) == [False, False, False, False, True]


def test_bullish_aligned_handles_nan_gracefully():
    df = pd.DataFrame({
        "close": [np.nan, 100, 110],   # iloc[2] close 110 > ma5 105
        "ma5":   [np.nan, 100, 105],
        "ma8":   [np.nan, 100, 102],
        "ema21": [np.nan, 100, 101],
        "ma55":  [np.nan, 100,  99],
    })
    aligned = bullish_aligned(df, AlignmentConfig.default())
    assert aligned.iloc[0] == False  # NaN row → False
    assert aligned.iloc[1] == False  # equality fails strict gt
    assert aligned.iloc[2] == True   # 110 > 105 > 102 > 101 > 99


def test_formation_events_detects_false_to_true_transition_only():
    aligned = pd.Series([False, False, True, True, True, False, True])
    events = formation_events(aligned)
    assert list(events) == [2, 6]


def test_formation_events_first_bar_aligned_does_not_count():
    aligned = pd.Series([True, True, False, True])
    events = formation_events(aligned)
    assert list(events) == [3]


def test_alignment_config_subset_disables_check():
    df = pd.DataFrame({
        "close": [100],
        "ma5":   [99],   # would normally fail close > ma5
        "ma8":   [98],
        "ema21": [97],
        "ma55":  [96],
    })
    cfg = AlignmentConfig(
        require_close_above_ma5=False,
        require_close_above_ma8=True,
        require_close_above_ema21=True,
        require_close_above_ma55=True,
        require_ma5_above_ma8=True,
        require_ma8_above_ema21=True,
        require_ema21_above_ma55=True,
    )
    aligned = bullish_aligned(df, cfg)
    assert aligned.iloc[0] == True
