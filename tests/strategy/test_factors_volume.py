import pandas as pd
from server.strategy.factors import _volume_confirmation


def _make_candles(volumes: list[float], close: float = 100.0) -> pd.DataFrame:
    n = len(volumes)
    return pd.DataFrame({
        "timestamp": list(range(n)),
        "open": [close] * n,
        "high": [close + 1] * n,
        "low": [close - 1] * n,
        "close": [close] * n,
        "volume": volumes,
    })


def test_volume_surge_scores_high():
    vols = [100.0] * 20 + [300.0]
    df = _make_candles(vols)
    score = _volume_confirmation(df, bar_index=20, lookback=20, surge_threshold=1.5)
    assert score > 0.8


def test_volume_below_threshold_scores_zero():
    vols = [100.0] * 21
    df = _make_candles(vols)
    score = _volume_confirmation(df, bar_index=20, lookback=20, surge_threshold=1.5)
    assert score == 0.0


def test_volume_at_threshold_scores_low():
    vols = [100.0] * 20 + [150.0]
    df = _make_candles(vols)
    score = _volume_confirmation(df, bar_index=20, lookback=20, surge_threshold=1.5)
    assert 0.0 < score <= 0.2


def test_volume_with_short_history():
    vols = [100.0, 100.0, 300.0]
    df = _make_candles(vols)
    score = _volume_confirmation(df, bar_index=2, lookback=20, surge_threshold=1.5)
    assert score > 0.5
