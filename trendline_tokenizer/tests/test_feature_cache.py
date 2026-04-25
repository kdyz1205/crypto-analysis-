"""FeatureCache tests: idempotency, no future leakage, shape parity
with the training-time _price_window helper."""
import numpy as np

from trendline_tokenizer.inference.feature_cache import FeatureCache


def _bar(t, c=100.0):
    return {"open_time": t, "open": c, "high": c + 0.5, "low": c - 0.5,
            "close": c, "volume": 1.0}


def test_cache_idempotent_on_same_open_time():
    fc = FeatureCache(capacity=10)
    assert fc.push("BTCUSDT", "5m", _bar(1000)) is True
    assert fc.push("BTCUSDT", "5m", _bar(1000)) is False  # duplicate
    assert fc.n_bars("BTCUSDT", "5m") == 1


def test_cache_evicts_oldest():
    fc = FeatureCache(capacity=3)
    for t in (1, 2, 3, 4, 5):
        fc.push("X", "5m", _bar(t))
    assert fc.n_bars("X", "5m") == 3
    df = fc.bars_df("X", "5m")
    assert df.iloc[0]["open_time"] == 3
    assert df.iloc[-1]["open_time"] == 5


def test_price_window_shape_matches_training():
    fc = FeatureCache(capacity=300)
    for t in range(200):
        fc.push("X", "5m", _bar(t * 60, c=100 + 0.01 * t))
    window, pad = fc.price_window("X", "5m", length=128)
    assert window.shape == (128, 13)
    assert pad.shape == (128,)
    assert np.isfinite(window).all()
    assert pad.sum() == 0  # 200 bars > 128, no padding


def test_price_window_pads_when_short():
    fc = FeatureCache(capacity=300)
    for t in range(20):
        fc.push("X", "5m", _bar(t * 60, c=100.0))
    window, pad = fc.price_window("X", "5m", length=64)
    assert window.shape == (64, 13)
    assert pad[:44].all()       # first 44 are padded
    assert not pad[44:].any()   # last 20 are real


def test_per_symbol_isolation():
    fc = FeatureCache(capacity=10)
    fc.push("A", "5m", _bar(1, c=100))
    fc.push("B", "5m", _bar(1, c=200))
    assert fc.n_bars("A", "5m") == 1
    assert fc.n_bars("B", "5m") == 1
    df_a = fc.bars_df("A", "5m")
    df_b = fc.bars_df("B", "5m")
    assert float(df_a.iloc[0]["close"]) == 100
    assert float(df_b.iloc[0]["close"]) == 200
