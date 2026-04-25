from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from backtests.ma_ribbon_ema21.tests.fixtures import (
    make_uptrend_with_formation, make_flat_ohlcv,
)
from server.strategy.ma_ribbon_auto_signals import (
    detect_new_signals_for_pair, BullSignalDetector, BearSignalDetector,
)


def test_bull_detector_emits_one_signal_at_first_formation():
    df, formation_at = make_uptrend_with_formation(
        n_bars=300, formation_at_bar=120, base_price=100.0
    )
    sigs = detect_new_signals_for_pair(df, "AAAUSDT", "1h", direction="long",
                                       last_processed_bar_ts=0)
    assert len(sigs) >= 1
    s = sigs[0]
    assert s.direction == "long"
    assert s.symbol == "AAAUSDT"
    assert s.tf == "1h"
    assert s.ema21_at_signal > 0
    assert s.next_bar_open_estimate > 0
    assert s.signal_bar_ts > 0


def test_bear_detector_emits_one_signal_at_first_bearish_formation():
    df, _ = make_uptrend_with_formation(
        n_bars=300, formation_at_bar=120, base_price=100.0,
        pre_drift=+0.001, post_drift=-0.005
    )
    sigs = detect_new_signals_for_pair(df, "BBBUSDT", "1h", direction="short",
                                       last_processed_bar_ts=0)
    assert len(sigs) >= 1
    assert all(s.direction == "short" for s in sigs)


def test_no_signal_on_flat_data():
    df = make_flat_ohlcv(n_bars=300, base_price=100.0)
    sigs = detect_new_signals_for_pair(df, "FLATUSDT", "1h", direction="long",
                                       last_processed_bar_ts=0)
    assert sigs == []


def test_dedup_via_last_processed_bar_ts():
    df, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=120, base_price=100.0)
    sigs1 = detect_new_signals_for_pair(df, "AAAUSDT", "1h", direction="long",
                                        last_processed_bar_ts=0)
    last_ts = sigs1[0].signal_bar_ts
    sigs2 = detect_new_signals_for_pair(df, "AAAUSDT", "1h", direction="long",
                                        last_processed_bar_ts=last_ts)
    assert all(s.signal_bar_ts > last_ts for s in sigs2)


def test_signal_id_is_unique_per_event():
    df, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=120, base_price=100.0)
    sigs = detect_new_signals_for_pair(df, "AAAUSDT", "1h", direction="long",
                                       last_processed_bar_ts=0)
    ids = [s.signal_id for s in sigs]
    assert len(set(ids)) == len(ids)


def test_bull_alignment_strict_inequality():
    """When ma5 == close, alignment must be False (strict gt per spec)."""
    df = pd.DataFrame({
        "timestamp": list(range(60)),
        "open":  [100.0] * 60, "high": [100.0] * 60, "low": [100.0] * 60,
        "close": [100.0] * 60, "volume": [1.0] * 60,
    })
    sigs = detect_new_signals_for_pair(df, "EQUSDT", "1h", direction="long",
                                       last_processed_bar_ts=0)
    assert sigs == []


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    sigs = detect_new_signals_for_pair(df, "X", "1h", direction="long",
                                       last_processed_bar_ts=0)
    assert sigs == []


def test_unknown_direction_raises():
    df = make_flat_ohlcv(n_bars=300, base_price=100.0)
    with pytest.raises(ValueError):
        detect_new_signals_for_pair(df, "X", "1h", direction="sideways",
                                    last_processed_bar_ts=0)
