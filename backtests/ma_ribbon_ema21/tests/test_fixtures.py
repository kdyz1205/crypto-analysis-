from __future__ import annotations
import pandas as pd
from backtests.ma_ribbon_ema21.tests.fixtures import (
    make_flat_ohlcv,
    make_uptrend_with_formation,
    make_real_csv_path,
)


def test_make_flat_ohlcv_shape():
    df = make_flat_ohlcv(n_bars=100, base_price=100.0)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 100
    assert (df["close"] == 100.0).all()
    assert df["timestamp"].is_monotonic_increasing


def test_make_uptrend_has_known_formation_bar():
    df, formation_bar_idx = make_uptrend_with_formation(
        n_bars=200, formation_at_bar=100, base_price=100.0
    )
    assert isinstance(df, pd.DataFrame)
    assert formation_bar_idx == 100
    # Closes after formation rise above closes before formation
    assert df.loc[formation_bar_idx + 20, "close"] > df.loc[formation_bar_idx - 1, "close"]


def test_real_csv_fixture_path_is_string():
    p = make_real_csv_path("BTCUSDT", "1h")
    # Path may or may not exist on first run — function only returns the path string
    assert isinstance(p, str)
    assert "BTCUSDT" in p and "1h" in p
