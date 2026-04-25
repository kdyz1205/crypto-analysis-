"""No-lookahead invariant: signal at bar i must depend only on bars [0..i].

This test walks through prefixes of a synthetic series and asserts that
indicators + alignment computed on each prefix yield the same value at the
prefix's last bar as the full-series compute. Any divergence means a future
bar leaked into a past signal (look-ahead bug). Runs every commit.
"""
from __future__ import annotations
import pandas as pd

from backtests.ma_ribbon_ema21.indicators import sma, ema
from backtests.ma_ribbon_ema21.ma_alignment import bullish_aligned, AlignmentConfig
from backtests.ma_ribbon_ema21.tests.fixtures import make_uptrend_with_formation


def _attach_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma5"]   = sma(out["close"], 5)
    out["ma8"]   = sma(out["close"], 8)
    out["ema21"] = ema(out["close"], 21)
    out["ma55"]  = sma(out["close"], 55)
    return out


def test_signal_at_bar_i_uses_only_bars_0_through_i():
    df, _ = make_uptrend_with_formation(
        n_bars=200, formation_at_bar=100, base_price=100.0
    )
    full = _attach_mas(df)
    full_aligned = bullish_aligned(full, AlignmentConfig.default())

    for i in range(60, len(df)):  # start after MA55 has at least one valid bar
        prefix = df.iloc[: i + 1].copy()
        prefix = _attach_mas(prefix)
        prefix_aligned = bullish_aligned(prefix, AlignmentConfig.default())
        assert prefix_aligned.iloc[i] == full_aligned.iloc[i], (
            f"alignment at bar {i} changed when future bars were hidden — look-ahead bug"
        )


def test_alignment_is_false_before_formation_bar():
    df, formation_at = make_uptrend_with_formation(
        n_bars=200, formation_at_bar=100, base_price=100.0,
        pre_drift=-0.001, post_drift=+0.005, noise_pct=0.0,
        seed=42,
    )
    full = _attach_mas(df)
    aligned = bullish_aligned(full, AlignmentConfig.default())
    # Several bars before the regime change, ribbon should not be aligned.
    assert aligned.iloc[formation_at - 5] == False
