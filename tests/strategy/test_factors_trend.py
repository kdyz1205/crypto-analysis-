import pandas as pd
from server.strategy.factors import _trend_context


def _make_trending_candles(direction: str, n: int = 60) -> pd.DataFrame:
    if direction == "up":
        closes = [100.0 + i * 0.5 for i in range(n)]
    elif direction == "down":
        closes = [100.0 - i * 0.5 for i in range(n)]
    else:
        closes = [100.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(n)]
    return pd.DataFrame({
        "timestamp": list(range(n)),
        "open": [c - 0.1 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


def test_long_in_uptrend_scores_high():
    df = _make_trending_candles("up")
    score = _trend_context(df, bar_index=59, direction="long", ema_period=50)
    assert score > 0.5


def test_long_in_downtrend_scores_zero():
    df = _make_trending_candles("down")
    score = _trend_context(df, bar_index=59, direction="long", ema_period=50)
    assert score == 0.0


def test_short_in_downtrend_scores_high():
    df = _make_trending_candles("down")
    score = _trend_context(df, bar_index=59, direction="short", ema_period=50)
    assert score > 0.5


def test_short_in_uptrend_scores_zero():
    df = _make_trending_candles("up")
    score = _trend_context(df, bar_index=59, direction="short", ema_period=50)
    assert score == 0.0


def test_sideways_scores_mid():
    df = _make_trending_candles("sideways")
    long_score = _trend_context(df, bar_index=59, direction="long", ema_period=50)
    short_score = _trend_context(df, bar_index=59, direction="short", ema_period=50)
    assert 0.0 <= long_score <= 0.7
    assert 0.0 <= short_score <= 0.7
