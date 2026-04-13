"""Tests for server.strategy.evolved.v2_trader.

Validates canon enforcement, not parameter values:
  - params_for raises on unknown TF (no silent default)
  - zigzag_pivots is causal + alternating + deterministic
  - top_k_by_amplitude returns sorted-by-idx
  - score_line stays bounded (~0-30, never v1a's 4000)
  - detect_lines: max 6 lines, min 3 touches each, anchor spacing 8+
  - No fan: no two same-side lines share start_index within 8 bars
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from server.strategy.evolved.v2_trader import (
    V2Params,
    ZigZagPivot,
    _atr,
    detect_lines,
    params_for,
    score_line,
    top_k_by_amplitude,
    zigzag_pivots,
)


def _make_candles(
    n: int,
    start_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = [start_price]
    for _ in range(1, n):
        closes.append(closes[-1] * (1 + drift) + rng.normal(0, volatility))
    closes = np.array(closes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, volatility * 0.5, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, volatility * 0.5, n))
    return pd.DataFrame({
        "timestamp": np.arange(n) * 14400,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.ones(n) * 1000,
    })


# ── module sanity ──────────────────────────────────────────
def test_module_imports():
    from server.strategy.evolved import v2_trader
    assert hasattr(v2_trader, "detect_lines")
    assert hasattr(v2_trader, "zigzag_pivots")
    assert hasattr(v2_trader, "params_for")


def test_evolved_channel_dataclass_exists():
    from server.strategy.evolved.base import EvolvedChannel
    assert EvolvedChannel is not None


# ── params_for ─────────────────────────────────────────────
def test_params_for_known_tfs():
    for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        p = params_for(tf)
        assert isinstance(p, V2Params)
        assert p.min_genuine_touches == 3, "canon: 3-touch confirmation"
        assert 0 < p.min_swing_pct < 0.2
        assert p.top_k_pivots == 6


def test_params_for_unknown_raises():
    with pytest.raises(ValueError, match="no parameters for timeframe"):
        params_for("nonexistent_tf")


# ── zigzag_pivots ──────────────────────────────────────────
def test_zigzag_too_short_returns_empty():
    df = _make_candles(20)
    atr = _atr(df, 14)
    assert zigzag_pivots(df, atr, 0.02, 2.0) == []


def test_zigzag_alternates_kinds():
    df = _make_candles(300, drift=0.001, volatility=2.0)
    atr = _atr(df, 14)
    pivots = zigzag_pivots(df, atr, 0.02, 2.0)
    assert len(pivots) >= 4
    for a, b in zip(pivots, pivots[1:]):
        assert a.kind != b.kind, f"two consecutive {a.kind} pivots at idx {a.idx},{b.idx}"


def test_zigzag_deterministic():
    df = _make_candles(300, seed=42)
    atr = _atr(df, 14)
    p1 = zigzag_pivots(df, atr, 0.02, 2.0)
    p2 = zigzag_pivots(df, atr, 0.02, 2.0)
    assert [(p.idx, p.kind, p.price) for p in p1] == [(p.idx, p.kind, p.price) for p in p2]


def test_zigzag_count_modest():
    """A 500-bar chart should produce ~5-60 major pivots, not 100+."""
    df = _make_candles(500, drift=0.001, volatility=2.0)
    atr = _atr(df, 14)
    pivots = zigzag_pivots(df, atr, 0.02, 2.0)
    assert 4 <= len(pivots) <= 80, f"got {len(pivots)} pivots"


# ── top_k_by_amplitude ─────────────────────────────────────
def test_top_k_returns_sorted_by_idx():
    pivots = [
        ZigZagPivot(idx=10, kind="low", price=10.0, amplitude=5.0),
        ZigZagPivot(idx=20, kind="high", price=20.0, amplitude=10.0),
        ZigZagPivot(idx=30, kind="low", price=8.0, amplitude=15.0),
        ZigZagPivot(idx=40, kind="high", price=25.0, amplitude=8.0),
    ]
    highs, lows = top_k_by_amplitude(pivots, k=2)
    assert [p.idx for p in highs] == [20, 40]
    assert [p.idx for p in lows] == [10, 30]


def test_top_k_limits_size():
    pivots = [ZigZagPivot(idx=i, kind="low", price=10.0, amplitude=float(i)) for i in range(10)]
    _, lows = top_k_by_amplitude(pivots, k=3)
    assert len(lows) == 3
    # Largest amplitudes 9,8,7 → sorted by idx → [7,8,9]
    assert [p.idx for p in lows] == [7, 8, 9]


def test_top_k_handles_k_too_large():
    pivots = [ZigZagPivot(idx=i, kind="high", price=10.0, amplitude=1.0) for i in range(3)]
    highs, lows = top_k_by_amplitude(pivots, k=99)
    assert len(highs) == 3
    assert len(lows) == 0


# ── score_line ─────────────────────────────────────────────
def test_score_line_in_reasonable_range():
    p1 = ZigZagPivot(idx=10, kind="low", price=100.0, amplitude=10.0)
    p2 = ZigZagPivot(idx=100, kind="low", price=110.0, amplitude=15.0)
    params = params_for("4h")
    s = score_line(p1, p2, touch_count=5, body_violations=0,
                   dist_to_now_atr=1.0, params=params)
    # 5 + (10+15)/40 + max(0, 3-1) - 0 = 5 + 0.625 + 2 = 7.625
    assert 0 < s < 30


def test_score_line_body_penalty():
    p1 = ZigZagPivot(idx=10, kind="low", price=100.0, amplitude=10.0)
    p2 = ZigZagPivot(idx=100, kind="low", price=110.0, amplitude=15.0)
    params = params_for("4h")
    clean = score_line(p1, p2, 5, 0, 1.0, params)
    dirty = score_line(p1, p2, 5, 3, 1.0, params)
    assert dirty < clean - 5


# ── detect_lines (integration) ─────────────────────────────
def test_detect_lines_too_short_returns_empty():
    df = _make_candles(20)
    assert detect_lines(df, "4h", "TEST") == []


def test_detect_lines_unknown_tf_returns_empty():
    """detect_lines catches the params_for ValueError and returns []."""
    df = _make_candles(100)
    assert detect_lines(df, "nonexistent_tf", "TEST") == []


def test_detect_lines_max_6():
    df = _make_candles(500, drift=0.001, volatility=2.0)
    lines = detect_lines(df, "4h", "TEST")
    assert len(lines) <= 6


def test_detect_lines_canon_min_touches():
    df = _make_candles(500, drift=0.0005, volatility=1.5)
    for line in detect_lines(df, "4h", "TEST"):
        assert line.touch_count >= 3, "canon rule 2: 3rd touch required"


def test_detect_lines_no_fan_pattern():
    """Acceptance: no two same-side lines share start_index within 8 bars."""
    df = _make_candles(500, drift=0.0008, volatility=1.8)
    lines = detect_lines(df, "4h", "TEST")
    supports = [l for l in lines if l.side == "support"]
    resistances = [l for l in lines if l.side == "resistance"]
    for group in (supports, resistances):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                gap = abs(group[i].start_index - group[j].start_index)
                assert gap >= 8, (
                    f"fan: {group[i].side} lines share start_index "
                    f"within 8 bars: {group[i].start_index} vs {group[j].start_index}"
                )


def test_detect_lines_score_bounded():
    """Acceptance: scores never reach v1a's pathological 4000."""
    df = _make_candles(500, drift=0.001, volatility=2.0)
    for line in detect_lines(df, "4h", "TEST"):
        assert -10 <= line.score <= 30, f"score {line.score} out of range"


def test_detect_lines_proximity_within_limit():
    """Each line's projection at current bar within proximity_atr."""
    df = _make_candles(500, drift=0.0008, volatility=1.8)
    params = params_for("4h")
    atr = _atr(df, 14)
    current_close = float(df["close"].iloc[-1])
    current_atr = float(atr.iloc[-1])
    n = len(df)
    for line in detect_lines(df, "4h", "TEST"):
        span = max(1, line.end_index - line.start_index)
        slope = (line.end_price - line.start_price) / span
        proj = line.start_price + slope * (n - 1 - line.start_index)
        dist_atr = abs(proj - current_close) / max(current_atr, 1e-9)
        assert dist_atr <= params.proximity_atr + 0.01, (
            f"line proj {proj:.2f} is {dist_atr:.2f} ATR from current "
            f"{current_close:.2f}, > proximity {params.proximity_atr}"
        )


def test_detect_lines_deterministic():
    df = _make_candles(500, drift=0.001, volatility=2.0)
    a = detect_lines(df, "4h", "TEST")
    b = detect_lines(df, "4h", "TEST")
    assert len(a) == len(b)
    for la, lb in zip(a, b):
        assert la.side == lb.side
        assert la.start_index == lb.start_index
        assert la.end_index == lb.end_index
        assert abs(la.score - lb.score) < 1e-9
