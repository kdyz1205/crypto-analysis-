"""Feature extraction for the ML learning layer.

Takes a trade record + OHLCV context and produces ~15-20 numeric features
suitable for tree-based classifiers (XGBoost / GradientBoosting).

Features are grouped into:
  - Line features: slope, anchor gap, line age (trendline-specific)
  - Market context: ATR, volume, RSI, momentum, EMA distance, ribbon state
  - Time features: hour of day, day of week
  - Trade features: buffer used, RR target, timeframe code

All features are floats. Missing / uncomputable features get a default
value (typically 0.0 or 0.5) so the array is always the same length.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Canonical feature order -- every model trains and predicts on this exact list.
FEATURE_NAMES: list[str] = [
    "slope_normalized",
    "anchor_gap_bars",
    "anchor_gap_pct",
    "line_age_bars",
    "atr_pct",
    "volume_ratio",
    "rsi_14",
    "momentum_10bar",
    "distance_from_ema200",
    "ma_ribbon_state",
    "hour_of_day",
    "day_of_week",
    "buffer_used",
    "rr_target",
    "timeframe_code",
    "bb_width_pct",
    "adx_14",
]

FEATURE_COUNT = len(FEATURE_NAMES)

# Timeframe -> numeric code (monotonically increasing with bar duration).
TF_CODE: dict[str, int] = {
    "1m": 0, "3m": 1, "5m": 2, "15m": 3, "30m": 4,
    "1h": 5, "2h": 6, "4h": 7, "1d": 8,
}


# ---------------------------------------------------------------------------
# Indicator helpers (pure numpy, no pandas dependency in hot path)
# ---------------------------------------------------------------------------

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (forward-fill NaN at start)."""
    out = np.empty_like(arr, dtype=float)
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average (NaN-padded at start)."""
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < period:
        return out
    cs = np.cumsum(arr)
    out[period - 1] = cs[period - 1] / period
    out[period:] = (cs[period:] - cs[:-period]) / period
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    if len(c) < 2:
        return np.zeros(len(c))
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return _ema(tr, period)


def _rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    if len(c) < period + 1:
        return np.full(len(c), 50.0)
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)
    rs = avg_gain / np.where(avg_loss > 1e-12, avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.concatenate([[50.0], rsi])


def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index (simplified)."""
    n = len(c)
    if n < period + 1:
        return np.full(n, 25.0)
    atr_vals = _atr(h, l, c, period)
    up_move = np.diff(h)
    down_move = np.diff(-l)  # equivalent to l[:-1] - l[1:]
    down_move = -np.diff(l)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100.0 * _ema(plus_dm, period) / np.where(atr_vals[1:] > 1e-12, atr_vals[1:], 1e-12)
    minus_di = 100.0 * _ema(minus_dm, period) / np.where(atr_vals[1:] > 1e-12, atr_vals[1:], 1e-12)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 1e-12, plus_di + minus_di, 1e-12)
    adx_vals = _ema(dx, period)
    return np.concatenate([[25.0], adx_vals])


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_features(
    trade: dict[str, Any],
    ohlcv: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Extract a fixed-length feature vector from a trade record + OHLCV context.

    Parameters
    ----------
    trade : dict
        Must contain at least ``timeframe``, ``dt`` (or ``ts``).  May also
        contain ``entry_price``, ``stop_price``, ``tp_price``,
        ``line_slope``, ``line_intercept``, ``line_entry_bar``,
        ``buffer_pct``, etc.
    ohlcv : dict | None
        ``{"o": np.ndarray, "h": ..., "l": ..., "c": ..., "v": ...}``
        for the symbol/TF around the time of the trade.  If None, market-
        context features default to neutral values.

    Returns
    -------
    np.ndarray of shape (FEATURE_COUNT,)
    """
    feats = np.zeros(FEATURE_COUNT, dtype=float)

    # Helper to safely set a feature by name
    def _set(name: str, val: float) -> None:
        try:
            idx = FEATURE_NAMES.index(name)
            feats[idx] = val if np.isfinite(val) else 0.0
        except ValueError:
            pass

    # ── Line features ──────────────────────────────────────────────────
    slope = float(trade.get("line_slope", 0) or 0)
    intercept = float(trade.get("line_intercept", 0) or 0)
    entry_bar = int(trade.get("line_entry_bar", 0) or 0)

    if ohlcv is not None and len(ohlcv.get("c", [])) > 14:
        c = ohlcv["c"]
        h = ohlcv["h"]
        l = ohlcv["l"]
        v = ohlcv["v"]
        n = len(c)
        last_close = float(c[-1])

        atr_series = _atr(h, l, c, 14)
        atr_now = float(atr_series[-1]) if len(atr_series) > 0 else 1.0
        atr_now = max(atr_now, 1e-12)

        # slope_normalized: slope per bar / ATR
        _set("slope_normalized", slope / atr_now if atr_now > 0 else 0.0)

        # ATR as % of price
        _set("atr_pct", atr_now / last_close * 100.0 if last_close > 0 else 0.0)

        # Volume ratio: last bar vs 20-bar SMA
        vol_sma = _sma(v, 20)
        vol_sma_now = float(vol_sma[-1]) if not np.isnan(vol_sma[-1]) else 1.0
        _set("volume_ratio", float(v[-1]) / max(vol_sma_now, 1e-12))

        # RSI 14
        rsi_series = _rsi(c, 14)
        _set("rsi_14", float(rsi_series[-1]))

        # Momentum: 10-bar return (%)
        if n >= 11:
            mom = (last_close - float(c[-11])) / float(c[-11]) * 100.0 if c[-11] > 0 else 0.0
            _set("momentum_10bar", mom)

        # Distance from EMA 200 (as % of price)
        if n >= 200:
            ema200 = _ema(c, 200)
            dist = (last_close - float(ema200[-1])) / last_close * 100.0 if last_close > 0 else 0.0
            _set("distance_from_ema200", dist)
        else:
            # Use whatever EMA we can compute (half the available bars)
            fallback_period = max(n // 2, 10)
            ema_fb = _ema(c, fallback_period)
            dist = (last_close - float(ema_fb[-1])) / last_close * 100.0 if last_close > 0 else 0.0
            _set("distance_from_ema200", dist)

        # MA ribbon state: 1=bull, -1=bear, 0=neutral
        # Bull: EMA5 > EMA8 > EMA21 > EMA55.  Bear: reverse.
        if n >= 55:
            ema5 = _ema(c, 5)[-1]
            ema8 = _ema(c, 8)[-1]
            ema21 = _ema(c, 21)[-1]
            ema55 = _ema(c, 55)[-1]
            if ema5 > ema8 > ema21 > ema55:
                _set("ma_ribbon_state", 1.0)
            elif ema5 < ema8 < ema21 < ema55:
                _set("ma_ribbon_state", -1.0)
            else:
                _set("ma_ribbon_state", 0.0)

        # Bollinger Band width (as % of price) using 20-period SMA + 2 std
        if n >= 20:
            bb_sma = _sma(c, 20)
            bb_sma_now = float(bb_sma[-1])
            if not np.isnan(bb_sma_now) and bb_sma_now > 0:
                bb_std = float(np.std(c[-20:]))
                bb_width = 2 * bb_std / bb_sma_now * 100.0
                _set("bb_width_pct", bb_width)

        # ADX 14
        adx_series = _adx(h, l, c, 14)
        _set("adx_14", float(adx_series[-1]))

    else:
        # No OHLCV context -- fill with neutral defaults
        _set("slope_normalized", 0.0)
        _set("atr_pct", 1.0)        # ~1% is typical for crypto
        _set("volume_ratio", 1.0)   # neutral
        _set("rsi_14", 50.0)        # neutral
        _set("momentum_10bar", 0.0)
        _set("distance_from_ema200", 0.0)
        _set("ma_ribbon_state", 0.0)
        _set("bb_width_pct", 2.0)
        _set("adx_14", 25.0)

    # ── Anchor gap ─────────────────────────────────────────────────────
    a1 = int(trade.get("anchor1_bar", 0) or 0)
    a2 = int(trade.get("anchor2_bar", 0) or 0)
    anchor_gap = abs(a2 - a1) if (a1 or a2) else 0
    _set("anchor_gap_bars", float(anchor_gap))
    if anchor_gap > 0 and ohlcv is not None and len(ohlcv["c"]) > 0:
        total_bars = len(ohlcv["c"])
        _set("anchor_gap_pct", anchor_gap / total_bars * 100.0)
    else:
        _set("anchor_gap_pct", 0.0)

    # line_age_bars: bars between anchor2 and the signal bar
    if entry_bar > 0 and a2 > 0:
        _set("line_age_bars", float(entry_bar - a2))
    else:
        _set("line_age_bars", 0.0)

    # ── Time features ──────────────────────────────────────────────────
    dt_str = trade.get("dt", "")
    ts = trade.get("ts", 0)
    try:
        if dt_str:
            dt_obj = datetime.fromisoformat(str(dt_str))
        elif ts:
            dt_obj = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        else:
            dt_obj = None
    except Exception:
        dt_obj = None

    if dt_obj is not None:
        _set("hour_of_day", float(dt_obj.hour))
        _set("day_of_week", float(dt_obj.weekday()))
    else:
        _set("hour_of_day", 12.0)
        _set("day_of_week", 3.0)

    # ── Trade features ─────────────────────────────────────────────────
    _set("buffer_used", float(trade.get("buffer_pct", 0) or 0))

    # RR target: infer from prices if not explicit
    entry_p = float(trade.get("entry_price", 0) or 0)
    stop_p = float(trade.get("stop_price", 0) or 0)
    tp_p = float(trade.get("tp_price", 0) or 0)
    if entry_p > 0 and stop_p > 0 and tp_p > 0:
        sl_dist = abs(entry_p - stop_p)
        tp_dist = abs(tp_p - entry_p)
        rr = tp_dist / sl_dist if sl_dist > 1e-12 else 0.0
        _set("rr_target", rr)
    else:
        _set("rr_target", float(trade.get("rr_target", 3.0) or 3.0))

    tf = str(trade.get("timeframe", "1h"))
    _set("timeframe_code", float(TF_CODE.get(tf, 5)))

    return feats


def extract_features_dict(
    trade: dict[str, Any],
    ohlcv: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Same as ``extract_features`` but returns a name->value dict."""
    arr = extract_features(trade, ohlcv)
    return {name: float(arr[i]) for i, name in enumerate(FEATURE_NAMES)}


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    "extract_features",
    "extract_features_dict",
]
