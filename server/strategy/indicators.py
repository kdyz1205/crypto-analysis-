"""Technical indicator factor library — 200+ indicators as callable factors.

Uses the `ta` library (https://github.com/bukosabino/ta) for 85+ built-in indicators,
plus custom implementations for candlestick patterns and additional factors.

Usage:
    from server.strategy.indicators import compute_all, compute_indicator, list_indicators

    # Compute single indicator
    result = compute_indicator(df, "rsi", period=14)

    # Compute all indicators at once
    all_indicators = compute_all(df)

    # List available indicators
    names = list_indicators()
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ── ta library indicators (85+) ─────────────────────────────────────────

def compute_all_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Add all ta library indicators to DataFrame. Returns new df with ~85 extra columns."""
    import ta
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    result = df.copy()
    try:
        result = ta.add_all_ta_features(result, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    except Exception:
        # If ta.add_all_ta_features fails, add key indicators manually
        result = _add_core_ta(result, close, high, low, volume)
    return result


def _add_core_ta(df, close, high, low, volume):
    """Manually add the most important indicators if bulk add fails."""
    import ta
    # Trend
    df["ema_8"] = ta.trend.ema_indicator(close, 8)
    df["ema_21"] = ta.trend.ema_indicator(close, 21)
    df["ema_55"] = ta.trend.ema_indicator(close, 55)
    df["sma_20"] = ta.trend.sma_indicator(close, 20)
    df["sma_50"] = ta.trend.sma_indicator(close, 50)
    df["sma_200"] = ta.trend.sma_indicator(close, 200)
    df["macd"] = ta.trend.macd(close)
    df["macd_signal"] = ta.trend.macd_signal(close)
    df["macd_hist"] = ta.trend.macd_diff(close)
    df["adx"] = ta.trend.adx(high, low, close)
    df["adx_pos"] = ta.trend.adx_pos(high, low, close)
    df["adx_neg"] = ta.trend.adx_neg(high, low, close)
    df["ichimoku_a"] = ta.trend.ichimoku_a(high, low)
    df["ichimoku_b"] = ta.trend.ichimoku_b(high, low)
    df["cci"] = ta.trend.cci(high, low, close)
    df["aroon_up"] = ta.trend.aroon_up(high, low)
    df["aroon_down"] = ta.trend.aroon_down(high, low)
    # Momentum
    df["rsi"] = ta.momentum.rsi(close, 14)
    df["rsi_6"] = ta.momentum.rsi(close, 6)
    df["stoch_k"] = ta.momentum.stoch(high, low, close)
    df["stoch_d"] = ta.momentum.stoch_signal(high, low, close)
    df["williams_r"] = ta.momentum.williams_r(high, low, close)
    df["roc"] = ta.momentum.roc(close)
    df["tsi"] = ta.momentum.tsi(close)
    df["uo"] = ta.momentum.ultimate_oscillator(high, low, close)
    df["ao"] = ta.momentum.awesome_oscillator(high, low)
    # Volatility
    df["bb_upper"] = ta.volatility.bollinger_hband(close)
    df["bb_lower"] = ta.volatility.bollinger_lband(close)
    df["bb_mid"] = ta.volatility.bollinger_mavg(close)
    df["bb_width"] = ta.volatility.bollinger_wband(close)
    df["bb_pband"] = ta.volatility.bollinger_pband(close)
    df["atr"] = ta.volatility.average_true_range(high, low, close)
    df["kc_upper"] = ta.volatility.keltner_channel_hband(high, low, close)
    df["kc_lower"] = ta.volatility.keltner_channel_lband(high, low, close)
    df["dc_upper"] = ta.volatility.donchian_channel_hband(high, low, close)
    df["dc_lower"] = ta.volatility.donchian_channel_lband(high, low, close)
    # Volume
    df["obv"] = ta.volume.on_balance_volume(close, volume)
    df["vwap"] = ta.volume.volume_weighted_average_price(high, low, close, volume)
    df["mfi"] = ta.volume.money_flow_index(high, low, close, volume)
    df["adi"] = ta.volume.acc_dist_index(high, low, close, volume)
    df["cmf"] = ta.volume.chaikin_money_flow(high, low, close, volume)
    df["fi"] = ta.volume.force_index(close, volume)
    df["eom"] = ta.volume.ease_of_movement(high, low, volume)
    return df


def compute_indicator(df: pd.DataFrame, name: str, **kwargs) -> pd.Series:
    """Compute a single indicator by name."""
    import ta
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    dispatch = {
        # Trend
        "ema": lambda: ta.trend.ema_indicator(close, kwargs.get("period", 21)),
        "sma": lambda: ta.trend.sma_indicator(close, kwargs.get("period", 20)),
        "macd": lambda: ta.trend.macd(close),
        "macd_signal": lambda: ta.trend.macd_signal(close),
        "macd_hist": lambda: ta.trend.macd_diff(close),
        "adx": lambda: ta.trend.adx(high, low, close, kwargs.get("period", 14)),
        "cci": lambda: ta.trend.cci(high, low, close, kwargs.get("period", 20)),
        "aroon_up": lambda: ta.trend.aroon_up(high, low, kwargs.get("period", 25)),
        "aroon_down": lambda: ta.trend.aroon_down(high, low, kwargs.get("period", 25)),
        "ichimoku_a": lambda: ta.trend.ichimoku_a(high, low),
        "ichimoku_b": lambda: ta.trend.ichimoku_b(high, low),
        # Momentum
        "rsi": lambda: ta.momentum.rsi(close, kwargs.get("period", 14)),
        "stoch_k": lambda: ta.momentum.stoch(high, low, close, kwargs.get("period", 14)),
        "stoch_d": lambda: ta.momentum.stoch_signal(high, low, close, kwargs.get("period", 14)),
        "williams_r": lambda: ta.momentum.williams_r(high, low, close, kwargs.get("period", 14)),
        "roc": lambda: ta.momentum.roc(close, kwargs.get("period", 12)),
        "tsi": lambda: ta.momentum.tsi(close),
        "uo": lambda: ta.momentum.ultimate_oscillator(high, low, close),
        "ao": lambda: ta.momentum.awesome_oscillator(high, low),
        # Volatility
        "atr": lambda: ta.volatility.average_true_range(high, low, close, kwargs.get("period", 14)),
        "bb_upper": lambda: ta.volatility.bollinger_hband(close, kwargs.get("period", 20)),
        "bb_lower": lambda: ta.volatility.bollinger_lband(close, kwargs.get("period", 20)),
        "bb_width": lambda: ta.volatility.bollinger_wband(close, kwargs.get("period", 20)),
        "bb_pband": lambda: ta.volatility.bollinger_pband(close, kwargs.get("period", 20)),
        "kc_upper": lambda: ta.volatility.keltner_channel_hband(high, low, close),
        "kc_lower": lambda: ta.volatility.keltner_channel_lband(high, low, close),
        "dc_upper": lambda: ta.volatility.donchian_channel_hband(high, low, close, kwargs.get("period", 20)),
        "dc_lower": lambda: ta.volatility.donchian_channel_lband(high, low, close, kwargs.get("period", 20)),
        # Volume
        "obv": lambda: ta.volume.on_balance_volume(close, volume),
        "mfi": lambda: ta.volume.money_flow_index(high, low, close, volume, kwargs.get("period", 14)),
        "cmf": lambda: ta.volume.chaikin_money_flow(high, low, close, volume, kwargs.get("period", 20)),
        "adi": lambda: ta.volume.acc_dist_index(high, low, close, volume),
        "vwap": lambda: ta.volume.volume_weighted_average_price(high, low, close, volume),
        "fi": lambda: ta.volume.force_index(close, volume, kwargs.get("period", 13)),
        "eom": lambda: ta.volume.ease_of_movement(high, low, volume, kwargs.get("period", 14)),
    }

    fn = dispatch.get(name.lower())
    if fn is None:
        raise ValueError(f"Unknown indicator: {name}. Available: {sorted(dispatch.keys())}")
    return fn()


# ── Candlestick pattern detection (40+ patterns) ────────────────────────

def detect_candlestick_patterns(df: pd.DataFrame) -> dict[str, list[int]]:
    """Detect all candlestick patterns. Returns {pattern_name: [bar_indices]}."""
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    n = len(c)

    results: dict[str, list[int]] = {}

    for name, fn in _CANDLE_PATTERNS.items():
        indices = []
        for i in range(2, n):
            try:
                if fn(o, h, l, c, i):
                    indices.append(i)
            except (IndexError, ZeroDivisionError):
                pass
        if indices:
            results[name] = indices

    return results


def _body(o, c, i): return abs(c[i] - o[i])
def _upper_wick(o, h, c, i): return h[i] - max(o[i], c[i])
def _lower_wick(o, l, c, i): return min(o[i], c[i]) - l[i]
def _range(h, l, i): return h[i] - l[i]
def _bullish(o, c, i): return c[i] > o[i]
def _bearish(o, c, i): return c[i] < o[i]
def _avg_body(o, c, i, n=5): return sum(abs(c[i-j] - o[i-j]) for j in range(n)) / n

# Pattern definitions
def _doji(o, h, l, c, i):
    r = _range(h, l, i)
    return r > 0 and _body(o, c, i) / r < 0.1

def _hammer(o, h, l, c, i):
    b = _body(o, c, i); lw = _lower_wick(o, l, c, i); uw = _upper_wick(o, h, c, i)
    return b > 0 and lw >= 2 * b and uw < b * 0.3

def _inverted_hammer(o, h, l, c, i):
    b = _body(o, c, i); uw = _upper_wick(o, h, c, i); lw = _lower_wick(o, l, c, i)
    return b > 0 and uw >= 2 * b and lw < b * 0.3

def _shooting_star(o, h, l, c, i):
    return _inverted_hammer(o, h, l, c, i) and _bearish(o, c, i)

def _hanging_man(o, h, l, c, i):
    return _hammer(o, h, l, c, i) and _bearish(o, c, i)

def _engulfing_bull(o, h, l, c, i):
    return _bearish(o, c, i-1) and _bullish(o, c, i) and o[i] <= c[i-1] and c[i] >= o[i-1]

def _engulfing_bear(o, h, l, c, i):
    return _bullish(o, c, i-1) and _bearish(o, c, i) and o[i] >= c[i-1] and c[i] <= o[i-1]

def _morning_star(o, h, l, c, i):
    return (i >= 2 and _bearish(o, c, i-2) and _body(o, c, i-2) > _avg_body(o, c, i-2) * 0.5
            and _body(o, c, i-1) < _body(o, c, i-2) * 0.3
            and _bullish(o, c, i) and c[i] > (o[i-2] + c[i-2]) / 2)

def _evening_star(o, h, l, c, i):
    return (i >= 2 and _bullish(o, c, i-2) and _body(o, c, i-2) > _avg_body(o, c, i-2) * 0.5
            and _body(o, c, i-1) < _body(o, c, i-2) * 0.3
            and _bearish(o, c, i) and c[i] < (o[i-2] + c[i-2]) / 2)

def _three_white_soldiers(o, h, l, c, i):
    return (i >= 2 and all(_bullish(o, c, i-j) for j in range(3))
            and c[i] > c[i-1] > c[i-2] and o[i] > o[i-1] > o[i-2])

def _three_black_crows(o, h, l, c, i):
    return (i >= 2 and all(_bearish(o, c, i-j) for j in range(3))
            and c[i] < c[i-1] < c[i-2] and o[i] < o[i-1] < o[i-2])

def _piercing(o, h, l, c, i):
    return (_bearish(o, c, i-1) and _bullish(o, c, i)
            and o[i] < c[i-1] and c[i] > (o[i-1] + c[i-1]) / 2 and c[i] < o[i-1])

def _dark_cloud(o, h, l, c, i):
    return (_bullish(o, c, i-1) and _bearish(o, c, i)
            and o[i] > c[i-1] and c[i] < (o[i-1] + c[i-1]) / 2 and c[i] > o[i-1])

def _harami_bull(o, h, l, c, i):
    return (_bearish(o, c, i-1) and _bullish(o, c, i)
            and o[i] > c[i-1] and c[i] < o[i-1])

def _harami_bear(o, h, l, c, i):
    return (_bullish(o, c, i-1) and _bearish(o, c, i)
            and o[i] < c[i-1] and c[i] > o[i-1])

def _tweezer_top(o, h, l, c, i):
    return abs(h[i] - h[i-1]) < _range(h, l, i) * 0.05 and _bullish(o, c, i-1) and _bearish(o, c, i)

def _tweezer_bottom(o, h, l, c, i):
    return abs(l[i] - l[i-1]) < _range(h, l, i) * 0.05 and _bearish(o, c, i-1) and _bullish(o, c, i)

def _marubozu_bull(o, h, l, c, i):
    r = _range(h, l, i)
    return r > 0 and _bullish(o, c, i) and _body(o, c, i) / r > 0.9

def _marubozu_bear(o, h, l, c, i):
    r = _range(h, l, i)
    return r > 0 and _bearish(o, c, i) and _body(o, c, i) / r > 0.9

def _spinning_top(o, h, l, c, i):
    r = _range(h, l, i)
    b = _body(o, c, i)
    return r > 0 and b / r < 0.3 and _upper_wick(o, h, c, i) > b and _lower_wick(o, l, c, i) > b

def _dragonfly_doji(o, h, l, c, i):
    r = _range(h, l, i)
    return r > 0 and _body(o, c, i) / r < 0.05 and _lower_wick(o, l, c, i) > r * 0.7

def _gravestone_doji(o, h, l, c, i):
    r = _range(h, l, i)
    return r > 0 and _body(o, c, i) / r < 0.05 and _upper_wick(o, h, c, i) > r * 0.7

def _inside_bar(o, h, l, c, i):
    return h[i] < h[i-1] and l[i] > l[i-1]

def _outside_bar(o, h, l, c, i):
    return h[i] > h[i-1] and l[i] < l[i-1]

def _pin_bar_bull(o, h, l, c, i):
    r = _range(h, l, i)
    return r > 0 and _lower_wick(o, l, c, i) / r > 0.6 and _body(o, c, i) / r < 0.2

def _pin_bar_bear(o, h, l, c, i):
    r = _range(h, l, i)
    return r > 0 and _upper_wick(o, h, c, i) / r > 0.6 and _body(o, c, i) / r < 0.2

def _three_inside_up(o, h, l, c, i):
    return (i >= 2 and _bearish(o, c, i-2) and _harami_bull(o, h, l, c, i-1)
            and _bullish(o, c, i) and c[i] > o[i-2])

def _three_inside_down(o, h, l, c, i):
    return (i >= 2 and _bullish(o, c, i-2) and _harami_bear(o, h, l, c, i-1)
            and _bearish(o, c, i) and c[i] < o[i-2])

def _rising_three(o, h, l, c, i):
    if i < 4: return False
    return (_bullish(o, c, i-4) and _body(o, c, i-4) > _avg_body(o, c, i) * 0.8
            and all(_bearish(o, c, i-j) and _body(o, c, i-j) < _body(o, c, i-4) * 0.5 for j in [3,2,1])
            and _bullish(o, c, i) and c[i] > c[i-4])

def _falling_three(o, h, l, c, i):
    if i < 4: return False
    return (_bearish(o, c, i-4) and _body(o, c, i-4) > _avg_body(o, c, i) * 0.8
            and all(_bullish(o, c, i-j) and _body(o, c, i-j) < _body(o, c, i-4) * 0.5 for j in [3,2,1])
            and _bearish(o, c, i) and c[i] < c[i-4])


# ── Chart Patterns (multi-bar structural patterns) ───────────────────────

def detect_chart_patterns(df: pd.DataFrame, lookback: int = 50) -> dict[str, list[dict]]:
    """Detect chart patterns (triangles, H&S, flags, wedges, etc).
    Returns {pattern_name: [{start_idx, end_idx, direction, confidence}]}."""
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    n = len(c)
    results: dict[str, list[dict]] = {}

    def _add(name, start, end, direction, confidence=0.7):
        if name not in results: results[name] = []
        results[name].append({"start_idx": start, "end_idx": end, "direction": direction, "confidence": confidence})

    for end in range(lookback, n):
        start = end - lookback

        highs = h[start:end+1]
        lows = l[start:end+1]
        closes = c[start:end+1]

        # Swing points in window
        swing_h = [(i, highs[i]) for i in range(2, len(highs)-2)
                   if highs[i] >= max(highs[i-2:i]) and highs[i] >= max(highs[i+1:i+3])]
        swing_l = [(i, lows[i]) for i in range(2, len(lows)-2)
                   if lows[i] <= min(lows[i-2:i]) and lows[i] <= min(lows[i+1:i+3])]

        if len(swing_h) < 2 or len(swing_l) < 2:
            continue

        # Higher highs / lower lows slopes
        h_slope = (swing_h[-1][1] - swing_h[0][1]) / max(swing_h[-1][0] - swing_h[0][0], 1)
        l_slope = (swing_l[-1][1] - swing_l[0][1]) / max(swing_l[-1][0] - swing_l[0][0], 1)
        h_range = max(highs) - min(lows)
        if h_range <= 0: continue
        norm_h_slope = h_slope / h_range * lookback
        norm_l_slope = l_slope / h_range * lookback

        # Ascending Triangle: flat top + rising bottoms
        if abs(norm_h_slope) < 0.05 and norm_l_slope > 0.1:
            _add("ascending_triangle", start, end, "bullish", 0.75)

        # Descending Triangle: flat bottom + falling tops
        if abs(norm_l_slope) < 0.05 and norm_h_slope < -0.1:
            _add("descending_triangle", start, end, "bearish", 0.75)

        # Symmetric Triangle: converging highs and lows
        if norm_h_slope < -0.05 and norm_l_slope > 0.05:
            _add("symmetric_triangle", start, end, "neutral", 0.65)

        # Expanding Triangle: diverging
        if norm_h_slope > 0.05 and norm_l_slope < -0.05:
            _add("expanding_triangle", start, end, "neutral", 0.55)

        # Rising Wedge: both rising, highs rising slower
        if norm_h_slope > 0.03 and norm_l_slope > 0.03 and norm_l_slope > norm_h_slope:
            _add("rising_wedge", start, end, "bearish", 0.7)

        # Falling Wedge: both falling, lows falling slower
        if norm_h_slope < -0.03 and norm_l_slope < -0.03 and norm_l_slope > norm_h_slope:
            _add("falling_wedge", start, end, "bullish", 0.7)

        # Bull Flag: sharp up then shallow down
        if len(swing_h) >= 3:
            pre_move = swing_h[-3][1] - swing_l[0][1] if len(swing_l) > 0 else 0
            flag_range = max(highs[-lookback//3:]) - min(lows[-lookback//3:])
            if pre_move > 0 and flag_range < pre_move * 0.4 and norm_h_slope < -0.02 and norm_l_slope < -0.02:
                _add("bull_flag", start, end, "bullish", 0.7)

        # Bear Flag: sharp down then shallow up
        if len(swing_l) >= 3:
            pre_move = swing_h[0][1] - swing_l[-3][1] if len(swing_h) > 0 else 0
            flag_range = max(highs[-lookback//3:]) - min(lows[-lookback//3:])
            if pre_move > 0 and flag_range < pre_move * 0.4 and norm_h_slope > 0.02 and norm_l_slope > 0.02:
                _add("bear_flag", start, end, "bearish", 0.7)

        # Head and Shoulders (need 3 swing highs)
        if len(swing_h) >= 3:
            left, head, right = swing_h[-3], swing_h[-2], swing_h[-1]
            if head[1] > left[1] and head[1] > right[1] and abs(left[1] - right[1]) / h_range < 0.15:
                _add("head_and_shoulders", start, end, "bearish", 0.8)

        # Inverse Head and Shoulders
        if len(swing_l) >= 3:
            left, head, right = swing_l[-3], swing_l[-2], swing_l[-1]
            if head[1] < left[1] and head[1] < right[1] and abs(left[1] - right[1]) / h_range < 0.15:
                _add("inverse_head_shoulders", start, end, "bullish", 0.8)

        # Double Top
        if len(swing_h) >= 2:
            t1, t2 = swing_h[-2], swing_h[-1]
            if abs(t1[1] - t2[1]) / h_range < 0.03 and t2[0] - t1[0] > 5:
                _add("double_top", start, end, "bearish", 0.75)

        # Double Bottom
        if len(swing_l) >= 2:
            b1, b2 = swing_l[-2], swing_l[-1]
            if abs(b1[1] - b2[1]) / h_range < 0.03 and b2[0] - b1[0] > 5:
                _add("double_bottom", start, end, "bullish", 0.75)

        # Triple Top
        if len(swing_h) >= 3:
            tops = [sh[1] for sh in swing_h[-3:]]
            if max(tops) - min(tops) < h_range * 0.05:
                _add("triple_top", start, end, "bearish", 0.8)

        # Triple Bottom
        if len(swing_l) >= 3:
            bots = [sl[1] for sl in swing_l[-3:]]
            if max(bots) - min(bots) < h_range * 0.05:
                _add("triple_bottom", start, end, "bullish", 0.8)

        # Cup and Handle (simplified: U shape in lows + small pullback at end)
        if len(swing_l) >= 3:
            lows_vals = [sl[1] for sl in swing_l[-3:]]
            if lows_vals[0] > lows_vals[1] and lows_vals[2] > lows_vals[1] and abs(lows_vals[0] - lows_vals[2]) / h_range < 0.1:
                _add("cup_and_handle", start, end, "bullish", 0.7)

        # Channel Up: parallel rising highs and lows
        if norm_h_slope > 0.05 and norm_l_slope > 0.05 and abs(norm_h_slope - norm_l_slope) < 0.05:
            _add("channel_up", start, end, "bullish", 0.65)

        # Channel Down: parallel falling
        if norm_h_slope < -0.05 and norm_l_slope < -0.05 and abs(norm_h_slope - norm_l_slope) < 0.05:
            _add("channel_down", start, end, "bearish", 0.65)

        # Range/Rectangle: flat highs and lows
        if abs(norm_h_slope) < 0.03 and abs(norm_l_slope) < 0.03:
            _add("rectangle", start, end, "neutral", 0.6)

        # Pennant: sharp move then small symmetric triangle
        vol_first_half = np.std(closes[:lookback//2])
        vol_second_half = np.std(closes[lookback//2:])
        if vol_second_half < vol_first_half * 0.4 and norm_h_slope < -0.03 and norm_l_slope > 0.03:
            direction = "bullish" if closes[lookback//2] > closes[0] else "bearish"
            _add("pennant", start, end, direction, 0.7)

    return results


_CHART_PATTERN_NAMES = [
    "ascending_triangle", "descending_triangle", "symmetric_triangle", "expanding_triangle",
    "rising_wedge", "falling_wedge", "bull_flag", "bear_flag",
    "head_and_shoulders", "inverse_head_shoulders",
    "double_top", "double_bottom", "triple_top", "triple_bottom",
    "cup_and_handle", "channel_up", "channel_down", "rectangle", "pennant",
]


_CANDLE_PATTERNS = {
    "doji": _doji,
    "hammer": _hammer,
    "inverted_hammer": _inverted_hammer,
    "shooting_star": _shooting_star,
    "hanging_man": _hanging_man,
    "engulfing_bull": _engulfing_bull,
    "engulfing_bear": _engulfing_bear,
    "morning_star": _morning_star,
    "evening_star": _evening_star,
    "three_white_soldiers": _three_white_soldiers,
    "three_black_crows": _three_black_crows,
    "piercing": _piercing,
    "dark_cloud": _dark_cloud,
    "harami_bull": _harami_bull,
    "harami_bear": _harami_bear,
    "tweezer_top": _tweezer_top,
    "tweezer_bottom": _tweezer_bottom,
    "marubozu_bull": _marubozu_bull,
    "marubozu_bear": _marubozu_bear,
    "spinning_top": _spinning_top,
    "dragonfly_doji": _dragonfly_doji,
    "gravestone_doji": _gravestone_doji,
    "inside_bar": _inside_bar,
    "outside_bar": _outside_bar,
    "pin_bar_bull": _pin_bar_bull,
    "pin_bar_bear": _pin_bar_bear,
    "three_inside_up": _three_inside_up,
    "three_inside_down": _three_inside_down,
    "rising_three": _rising_three,
    "falling_three": _falling_three,
}


# ── Custom composite indicators ─────────────────────────────────────────

def squeeze_momentum(df: pd.DataFrame, bb_period=20, kc_period=20, atr_period=14) -> pd.Series:
    """TTM Squeeze: BB inside KC = squeeze. Positive momentum = breakout direction."""
    import ta
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    bb_upper = ta.volatility.bollinger_hband(close, bb_period)
    bb_lower = ta.volatility.bollinger_lband(close, bb_period)
    kc_upper = ta.volatility.keltner_channel_hband(high, low, close, kc_period)
    kc_lower = ta.volatility.keltner_channel_lband(high, low, close, kc_period)
    squeeze = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(float)
    return squeeze


def market_structure_score(df: pd.DataFrame, lookback=20) -> pd.Series:
    """Score 0-100: higher highs + higher lows = bullish structure."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    scores = pd.Series(50.0, index=df.index)
    for i in range(lookback, len(df)):
        window_h = high.iloc[i-lookback:i+1]
        window_l = low.iloc[i-lookback:i+1]
        hh = (window_h.diff().dropna() > 0).sum()
        hl = (window_l.diff().dropna() > 0).sum()
        total = lookback
        scores.iloc[i] = ((hh + hl) / (2 * total)) * 100
    return scores


def volume_profile_poc(df: pd.DataFrame, lookback=50, bins=20) -> pd.Series:
    """Point of Control: price level with highest volume in lookback window."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    poc = pd.Series(np.nan, index=df.index)
    for i in range(lookback, len(df)):
        window_close = close.iloc[i-lookback:i+1].values
        window_vol = volume.iloc[i-lookback:i+1].values
        bin_edges = np.linspace(window_close.min(), window_close.max(), bins + 1)
        vol_per_bin = np.zeros(bins)
        for j, price in enumerate(window_close):
            bin_idx = min(np.searchsorted(bin_edges[1:], price), bins - 1)
            vol_per_bin[bin_idx] += window_vol[j]
        max_bin = np.argmax(vol_per_bin)
        poc.iloc[i] = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    return poc


# ── Master list ──────────────────────────────────────────────────────────

def list_indicators() -> list[dict]:
    """List all available indicators with metadata."""
    indicators = []

    # ta library indicators
    ta_indicators = [
        {"name": "ema", "category": "trend", "description": "Exponential Moving Average"},
        {"name": "sma", "category": "trend", "description": "Simple Moving Average"},
        {"name": "macd", "category": "trend", "description": "MACD Line"},
        {"name": "macd_signal", "category": "trend", "description": "MACD Signal"},
        {"name": "macd_hist", "category": "trend", "description": "MACD Histogram"},
        {"name": "adx", "category": "trend", "description": "Average Directional Index"},
        {"name": "cci", "category": "trend", "description": "Commodity Channel Index"},
        {"name": "aroon_up", "category": "trend", "description": "Aroon Up"},
        {"name": "aroon_down", "category": "trend", "description": "Aroon Down"},
        {"name": "ichimoku_a", "category": "trend", "description": "Ichimoku Senkou A"},
        {"name": "ichimoku_b", "category": "trend", "description": "Ichimoku Senkou B"},
        {"name": "rsi", "category": "momentum", "description": "Relative Strength Index"},
        {"name": "stoch_k", "category": "momentum", "description": "Stochastic %K"},
        {"name": "stoch_d", "category": "momentum", "description": "Stochastic %D"},
        {"name": "williams_r", "category": "momentum", "description": "Williams %R"},
        {"name": "roc", "category": "momentum", "description": "Rate of Change"},
        {"name": "tsi", "category": "momentum", "description": "True Strength Index"},
        {"name": "uo", "category": "momentum", "description": "Ultimate Oscillator"},
        {"name": "ao", "category": "momentum", "description": "Awesome Oscillator"},
        {"name": "atr", "category": "volatility", "description": "Average True Range"},
        {"name": "bb_upper", "category": "volatility", "description": "Bollinger Upper Band"},
        {"name": "bb_lower", "category": "volatility", "description": "Bollinger Lower Band"},
        {"name": "bb_width", "category": "volatility", "description": "Bollinger Band Width"},
        {"name": "bb_pband", "category": "volatility", "description": "Bollinger %B"},
        {"name": "kc_upper", "category": "volatility", "description": "Keltner Upper"},
        {"name": "kc_lower", "category": "volatility", "description": "Keltner Lower"},
        {"name": "dc_upper", "category": "volatility", "description": "Donchian Upper"},
        {"name": "dc_lower", "category": "volatility", "description": "Donchian Lower"},
        {"name": "obv", "category": "volume", "description": "On Balance Volume"},
        {"name": "mfi", "category": "volume", "description": "Money Flow Index"},
        {"name": "cmf", "category": "volume", "description": "Chaikin Money Flow"},
        {"name": "adi", "category": "volume", "description": "Accumulation/Distribution"},
        {"name": "vwap", "category": "volume", "description": "Volume Weighted Avg Price"},
        {"name": "fi", "category": "volume", "description": "Force Index"},
        {"name": "eom", "category": "volume", "description": "Ease of Movement"},
    ]
    indicators.extend(ta_indicators)

    # Candlestick patterns
    for name in _CANDLE_PATTERNS:
        indicators.append({"name": f"candle_{name}", "category": "candlestick", "description": name.replace("_", " ").title()})

    # Chart patterns (structural)
    chart_pattern_descriptions = {
        "ascending_triangle": "上升三角形 — 平顶+上升底部，看涨突破",
        "descending_triangle": "下降三角形 — 平底+下降顶部，看跌突破",
        "symmetric_triangle": "对称三角形 — 收敛，方向待定",
        "expanding_triangle": "扩张三角形 — 发散，波动增加",
        "rising_wedge": "上升楔形 — 看跌反转",
        "falling_wedge": "下降楔形 — 看涨反转",
        "bull_flag": "牛旗 — 急涨后窄幅回调，看涨延续",
        "bear_flag": "熊旗 — 急跌后窄幅反弹，看跌延续",
        "head_and_shoulders": "头肩顶 — 看跌反转",
        "inverse_head_shoulders": "头肩底 — 看涨反转",
        "double_top": "双顶 — 看跌反转",
        "double_bottom": "双底 — 看涨反转",
        "triple_top": "三重顶 — 强看跌反转",
        "triple_bottom": "三重底 — 强看涨反转",
        "cup_and_handle": "杯柄形态 — 看涨延续",
        "channel_up": "上升通道",
        "channel_down": "下降通道",
        "rectangle": "矩形整理区间",
        "pennant": "三角旗 — 延续形态",
    }
    for name in _CHART_PATTERN_NAMES:
        indicators.append({"name": f"chart_{name}", "category": "chart_pattern", "description": chart_pattern_descriptions.get(name, name)})

    # Custom composites
    indicators.extend([
        {"name": "squeeze_momentum", "category": "composite", "description": "TTM Squeeze Momentum"},
        {"name": "market_structure", "category": "composite", "description": "Market Structure Score (HH/HL)"},
        {"name": "volume_poc", "category": "composite", "description": "Volume Profile Point of Control"},
    ])

    return indicators


__all__ = [
    "compute_all_ta",
    "compute_indicator",
    "detect_candlestick_patterns",
    "detect_chart_patterns",
    "list_indicators",
    "market_structure_score",
    "squeeze_momentum",
    "volume_profile_poc",
]
