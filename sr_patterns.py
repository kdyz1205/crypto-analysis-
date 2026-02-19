"""
Advanced Support/Resistance and Pattern Detection

Features:
- Horizontal support/resistance zones (clustered price levels)
- Tilted support/resistance lines (both directions)
- Triangle pattern detection (ascending, descending, symmetrical)
- Consolidation zone detection (flat rectangles with low volatility)
"""

# =============================================================================
# USER CONFIGURATION - Edit these values
# =============================================================================
# Data file path (e.g., "data/hypeusdt_4h.csv", "data/btcusdt_4h.csv")
# Run: python tools/download_full_okx.py HYPEUSDT 4h  first
DATA_PATH = "data/hypeusdt_4h.csv"

# Start date filter (set to None to use all data)
START_DATE = "2026-01-18"

# REPLAY MODE: Set to a datetime string to replay historical moment
# Format: "YYYY-MM-DD HH:MM" (24-hour format)
# Set to None for normal mode (use all data as current)
REPLAY_TIME = None  # e.g., "2026-01-20 14:00"

# Display timeframe (resample to this interval)
# Use Polars duration strings: "5m", "15m", "1h", "4h", "1d"
# Set to None to use the dataset's native timeframe
DISPLAY_TIMEFRAME = "15m" # e.g., "4h"

# Display toggles
SHOW_EMA = True              # Show EMA line
SHOW_TRENDLINES = True       # Show support/resistance trendlines
SHOW_SR_ZONES = False        # Show horizontal S/R zone lines (yellow)
SHOW_TRIANGLES = False       # Show triangle pattern overlays (blue)
SHOW_CONSOLIDATION = True    # Show consolidation zone rectangles (orange)
SHOW_LOCAL_EXTREMA = True    # Show local high/low markers
SHOW_TREND_SEGMENTS = True   # Show trend segment bar at x-axis (green/yellow/red)
# =============================================================================

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import os
import re


def extract_token_and_interval(filepath: str) -> tuple[str, str]:
    """
    Extract token and interval from filename.

    Examples:
        "riverusdt_1h.csv" -> ("RIVER", "1H")
        "data/btc_4h.csv" -> ("BTC", "4H")
        "ENSO_15m.csv" -> ("ENSO", "15M")
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]  # Remove .csv

    # Match pattern: {token}[usdt|usd|usdc]_{interval}
    match = re.match(r"^(.+?)(?:usdt|usd|usdc|busd)?_(\d+[mhd])$", name, re.IGNORECASE)
    if match:
        token = match.group(1).upper()
        interval = match.group(2).upper()
        return token, interval

    # Fallback: split by underscore
    parts = name.split("_")
    if len(parts) >= 2:
        token = parts[0].upper()
        for suffix in ["USDT", "USD", "USDC", "BUSD"]:
            if token.endswith(suffix):
                token = token[:-len(suffix)]
                break
        interval = parts[-1].upper()
        return token, interval

    return name.upper(), ""


# Map human-friendly interval strings to timedelta for comparison
_INTERVAL_TO_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080,
}


def _interval_str_to_minutes(interval: str) -> int:
    """Convert interval string like '1h', '4h', '15m', '1d' to minutes."""
    key = interval.lower().strip()
    if key in _INTERVAL_TO_MINUTES:
        return _INTERVAL_TO_MINUTES[key]
    # Try parsing: number + unit
    m = re.match(r"^(\d+)\s*(m|h|d|w)$", key)
    if m:
        val, unit = int(m.group(1)), m.group(2)
        mult = {"m": 1, "h": 60, "d": 1440, "w": 10080}[unit]
        return val * mult
    raise ValueError(f"Cannot parse interval: {interval}")


def _interval_to_polars_duration(interval: str) -> str:
    """Convert interval string to Polars duration format (e.g. '4h' -> '4h')."""
    key = interval.lower().strip()
    m = re.match(r"^(\d+)\s*(m|h|d|w)$", key)
    if m:
        val, unit = m.group(1), m.group(2)
        return f"{val}{unit}"
    raise ValueError(f"Cannot parse interval for Polars: {interval}")


def detect_data_interval(df: pl.DataFrame) -> tuple[int, str]:
    """
    Detect the native interval of the dataset from the first two timestamps.
    Returns (minutes, human_label) e.g. (60, "1H").
    """
    times = df["open_time"]
    if len(times) < 2:
        return 0, "?"
    delta = times[1] - times[0]
    total_minutes = int(delta.total_seconds() / 60)

    # Find best human label
    for label, mins in sorted(_INTERVAL_TO_MINUTES.items(), key=lambda x: x[1]):
        if mins == total_minutes:
            return total_minutes, label.upper()
    # Fallback
    if total_minutes >= 1440:
        return total_minutes, f"{total_minutes // 1440}D"
    elif total_minutes >= 60:
        return total_minutes, f"{total_minutes // 60}H"
    else:
        return total_minutes, f"{total_minutes}M"


def resample_ohlcv(df: pl.DataFrame, target_interval: str, native_minutes: int) -> tuple[pl.DataFrame, bool, str]:
    """
    Resample OHLCV data to a larger timeframe using Polars group_by_dynamic.

    Returns (resampled_df, was_resampled, message).
    - If target is smaller than native, returns original with a warning.
    - If target produces fewer than 3 bars, reduces to a workable interval.
    """
    target_minutes = _interval_str_to_minutes(target_interval)

    if target_minutes < native_minutes:
        _, native_label = detect_data_interval(df)
        msg = (f"Warning: requested timeframe {target_interval} is smaller than "
               f"dataset granularity ({native_label}). Showing native {native_label}.")
        return df, False, msg

    if target_minutes == native_minutes:
        return df, False, ""

    # Resample
    duration = _interval_to_polars_duration(target_interval)
    resampled = (
        df.sort("open_time")
        .group_by_dynamic("open_time", every=duration)
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum() if "volume" in df.columns else pl.lit(0).alias("volume"),
        ])
    )

    # Check if we have enough bars
    if len(resampled) < 3:
        # Try halving the interval until we get >= 3 bars
        attempt_minutes = target_minutes // 2
        while attempt_minutes > native_minutes and attempt_minutes > 0:
            # Build a label for the attempt
            if attempt_minutes >= 1440:
                attempt_label = f"{attempt_minutes // 1440}d"
            elif attempt_minutes >= 60:
                attempt_label = f"{attempt_minutes // 60}h"
            else:
                attempt_label = f"{attempt_minutes}m"

            attempt_duration = _interval_to_polars_duration(attempt_label)
            resampled = (
                df.sort("open_time")
                .group_by_dynamic("open_time", every=attempt_duration)
                .agg([
                    pl.col("open").first(),
                    pl.col("high").max(),
                    pl.col("low").min(),
                    pl.col("close").last(),
                    pl.col("volume").sum() if "volume" in df.columns else pl.lit(0).alias("volume"),
                ])
            )
            if len(resampled) >= 3:
                msg = (f"Warning: {target_interval} produces <3 bars. "
                       f"Reduced to {attempt_label.upper()} ({len(resampled)} bars).")
                return resampled, True, msg
            attempt_minutes //= 2

        # If nothing works, return native
        _, native_label = detect_data_interval(df)
        msg = (f"Warning: {target_interval} produces <3 bars even after reduction. "
               f"Showing native {native_label}.")
        return df, False, msg

    return resampled, True, f"Resampled to {target_interval.upper()} ({len(resampled)} bars)."


@dataclass
class SRParams:
    """
    Parameters for pattern detection.

    Epsilon (eps) values — there are two, both price-adaptive:
      1. eps (trendline eps) = ATR(atr_period) * atr_multiplier
         Computed at runtime from recent volatility. Used for:
         - Trendline touch detection (is price "near" the line?)
         - Trendline break detection (has price crossed through?)
         - Collinear point merging (are 3+ extrema on the same line?)
      2. sr_eps = current_price * sr_eps_pct
         Percentage of price. Used for horizontal S/R zone clustering.
    """
    # --- Local extrema detection ---
    window_left: int = 1       # Left lookback for local high/low detection (bars)
    window_right: int = 5      # Right lookback for local high/low detection (bars)
    window_short: int = 2      # Shortened window used on the trend side (asymmetric detection)
    prominence_multiplier: float = 1.0  # Filter extrema by prominence (height relative to ATR)

    # --- Trend detection ---
    ema_span: int = 50         # EMA period; lower = more reactive, higher = smoother
    trend_threshold: float = 0.5  # Combined score threshold to classify up/down/sideways (%)
    use_fast_trend: bool = True   # True = price-vs-EMA + momentum; False = EMA slope only

    # --- Epsilon / tolerance (ATR-based, for trendlines) ---
    atr_period: int = 14          # ATR lookback period
    atr_multiplier: float = 0.5   # eps = ATR * this. Controls how close a touch must be to a line
    line_break_factor: float = 0.3  # A line is "broken" if price exceeds eps * this beyond the line
    
    # --- Dynamic tolerance for strict pivot validation ---
    tolerance_percent: float = 0.005  # Base tolerance percentage (0.5% default, more lenient)
    k_atr: float = 0.5  # ATR multiplier for dynamic tolerance: max(fixed_pct * price, k_atr * ATR)
    fixed_pct: float = 0.005  # Fixed percentage for dynamic tolerance (0.5% default, more lenient)
    high_volatility_threshold: float = 0.02  # ATR/price ratio above which asset is considered high volatility
    high_vol_tolerance_pct: float = 0.007  # Tolerance for high volatility assets (0.7% for high volatility)
    # Strict non-crossing: line must sit on top of wicks (resistance) or below (support)
    ceiling_tolerance: float = 0.001  # Resistance: High[i] <= LineValue[i] * (1 + ceiling_tolerance). Default 0.001 = 0.1%

    # --- Epsilon / tolerance (%-based, for horizontal S/R zones) ---
    sr_eps_pct: float = 0.01     # sr_eps = price * this (0.01 = 1%). Cluster width for S/R zones

    # --- General line / zone parameters ---
    lookback: Optional[int] = 300  # Only consider bars within this window (None = all data)
    min_touches: int = 0           # Minimum touch count for a valid horizontal zone or trendline
    show_horizontal_zones: bool = False  # Toggle yellow horizontal S/R zone lines on the plot
    show_triangles: bool = False          # Toggle blue triangle pattern overlays on the plot
    trend_filter_lines: bool = True       # Hide counter-trend lines (dashed green in downtrend, dashed red in uptrend)
    zone_merge_factor: float = 1.5  # Merge horizontal zones within sr_eps * this of each other
    slope_tolerance: float = 0.0001 # Slopes with abs < this are treated as flat (per bar)
    max_lines: int = None           # Cap on trendlines per type (None = unlimited)

    # --- Consolidation zone detection ---
    consol_window: int = 10         # Rolling window size for consolidation check (bars)
    consol_range_pct: float = 0.1   # Max (high-low)/low within window to count as tight (0.10 = 10%)
    consol_slope_threshold: float = 0.003  # Max abs normalized slope to count as flat
    consol_min_duration: int = 10   # Minimum consecutive bars to form a consolidation zone
    consol_efficiency_max: float = 0.35  # Max efficiency ratio (net move / total move); low = choppy/consolidating


# =============================================================================
# PATTERN DETECTION PARAMETERS - Edit these values to tune detection
# =============================================================================
PARAMS = SRParams(
    # Local extrema detection
    window_left=1,          # bars to the left for peak/trough detection
    window_right=5,         # bars to the right for peak/trough detection
    window_short=2,         # shortened window on the trend-favored side
    # Trend detection
    ema_span=50,            # EMA period for trend classification
    trend_threshold=0.5,    # score threshold for up/down/sideways
    use_fast_trend=True,    # True = price-vs-EMA + momentum; False = EMA slope
    # Trendline epsilon (ATR-based): eps = ATR(atr_period) * atr_multiplier
    atr_period=14,          # ATR lookback
    atr_multiplier=0.5,     # touch distance = ATR * this
    line_break_factor=0.3,  # line broken when price exceeds eps * this past it
    # Horizontal S/R epsilon (%-based): sr_eps = price * sr_eps_pct
    sr_eps_pct=0.01,        # 0.01 = 1% of price
    # General
    lookback=None,          # bar window (None = all data)
    min_touches=2,          # min touches for zones and trendlines
    zone_merge_factor=1.5,  # merge zones within sr_eps * this
    slope_tolerance=0.0001, # slopes below this are "flat"
    max_lines=15,           # max trendlines per type
)


@dataclass
class HorizontalZone:
    """Horizontal support/resistance zone."""
    price_low: float
    price_high: float
    price_center: float
    start_idx: int
    end_idx: int
    touches: int
    zone_type: str  # 'support' or 'resistance'
    strength: float  # Based on touches and recency


@dataclass
class TrendLine:
    """Tilted support or resistance line."""
    x1: int
    y1: float
    x2: int
    y2: float
    slope: float
    line_type: str  # 'support' or 'resistance'
    touches: int
    strength: float
    tolerance: float = 0.0  # Tolerance used for validation (for zone display)


@dataclass
class TrianglePattern:
    """Triangle/wedge pattern."""
    support_line: TrendLine
    resistance_line: TrendLine
    apex_x: int
    apex_price: float
    pattern_type: str  # 'ascending', 'descending', 'symmetrical', 'rising_wedge', 'falling_wedge'
    breakout_bias: str  # 'bullish', 'bearish', 'neutral'
    completion_pct: float  # How close price is to apex (0-100%)


@dataclass
class ConsolidationZone:
    """Consolidation/accumulation zone (flat rectangle)."""
    start_idx: int
    end_idx: int
    price_low: float   # Bottom of the rectangle
    price_high: float  # Top of the rectangle
    price_center: float
    channel_width_pct: float  # (high - low) / low as percentage
    avg_slope: float   # Average normalized slope (near zero = flat)
    duration: int      # Number of bars in the zone


@dataclass
class TrendSegment:
    """A contiguous segment of bars sharing the same trend classification."""
    start_idx: int
    end_idx: int
    trend: int       # 1=uptrend, -1=downtrend, 0=sideways
    avg_score: float  # Average trend score over the segment


def segment_trends(
    df: pl.DataFrame,
    local_high_indices: np.ndarray,
    local_high_prices: np.ndarray,
    local_low_indices: np.ndarray,
    local_low_prices: np.ndarray,
    min_segment_bars: int = 3,
    min_swing_pct: float = 0.03,
) -> list[TrendSegment]:
    """
    Classify bars into uptrend / downtrend / sideways using market structure
    (local highs and lows) instead of EMA-based scoring.

    Logic (Dow Theory / swing structure):
    - Uptrend:  higher highs AND higher lows
    - Downtrend: lower highs AND lower lows
    - Mixed signals (e.g. higher high + lower low): keep current state

    Transition placement:
    - UP → DOWN: at the bar of the last swing high (the peak)
    - DOWN → UP: at the bar of the last swing low (the trough)

    Only "significant" swings (reversals >= min_swing_pct of price) are used,
    to avoid noisy flip-flopping from small oscillations.

    Short segments (< min_segment_bars) are absorbed into their neighbour.
    """
    n = len(df)
    if n == 0:
        return []

    # Build chronological swing list
    raw_swings = []
    for i in range(len(local_high_indices)):
        raw_swings.append(('H', int(local_high_indices[i]), float(local_high_prices[i])))
    for i in range(len(local_low_indices)):
        raw_swings.append(('L', int(local_low_indices[i]), float(local_low_prices[i])))
    raw_swings.sort(key=lambda x: x[1])  # sort by bar index

    if len(raw_swings) < 4:
        return [TrendSegment(0, n - 1, 0, 0.0)]

    # Filter to significant swings: alternating H-L-H-L where each reversal
    # represents at least min_swing_pct move from the previous swing.
    # If same type repeats, keep the more extreme one (highest H, lowest L).
    swings: list[tuple[str, int, float]] = [raw_swings[0]]
    for swing_type, bar_idx, price in raw_swings[1:]:
        last_type, last_idx, last_price = swings[-1]
        if swing_type == last_type:
            # Same type: keep the more extreme
            if (swing_type == 'H' and price > last_price) or \
               (swing_type == 'L' and price < last_price):
                swings[-1] = (swing_type, bar_idx, price)
        else:
            # Different type: only accept if move is significant
            move_pct = abs(price - last_price) / last_price
            if move_pct >= min_swing_pct:
                swings.append((swing_type, bar_idx, price))
            # else: ignore this minor swing, keep looking

    if len(swings) < 2:
        return [TrendSegment(0, n - 1, 0, 0.0)]

    # Each significant swing marks a trend transition:
    #   Swing HIGH → price just peaked → DOWN segment starts here
    #   Swing LOW  → price just bottomed → UP segment starts here
    transitions: list[tuple[int, int]] = []
    for swing_type, bar_idx, price in swings:
        new_state = -1 if swing_type == 'H' else 1
        transitions.append((bar_idx, new_state))

    # Infer the segment before the first swing from its type:
    # if first swing is a HIGH, price was rising → UP before it
    # if first swing is a LOW, price was falling → DOWN before it
    first_state = 1 if swings[0][0] == 'H' else -1
    if transitions[0][0] > 0:
        transitions.insert(0, (0, first_state))

    # --- build segments from transitions ---
    if not transitions:
        return [TrendSegment(0, n - 1, 0, 0.0)]

    segments: list[TrendSegment] = []
    # Before first transition: sideways
    if transitions[0][0] > 0:
        segments.append(TrendSegment(0, transitions[0][0] - 1, 0, 0.0))

    for i, (t_bar, trend) in enumerate(transitions):
        end_bar = transitions[i + 1][0] - 1 if i + 1 < len(transitions) else n - 1
        if end_bar >= t_bar:
            segments.append(TrendSegment(t_bar, end_bar, trend, 0.0))

    if not segments:
        return [TrendSegment(0, n - 1, 0, 0.0)]

    # --- merge short segments into the longer neighbour ---
    while True:
        # Find the shortest segment below the threshold
        shortest_idx = None
        shortest_dur = float('inf')
        for i, seg in enumerate(segments):
            dur = seg.end_idx - seg.start_idx + 1
            if dur < min_segment_bars and dur < shortest_dur:
                shortest_dur = dur
                shortest_idx = i

        if shortest_idx is None:
            break  # all segments meet the minimum

        if len(segments) <= 1:
            break  # nothing to merge into

        seg = segments[shortest_idx]
        if shortest_idx == 0:
            # First segment: merge into next
            nxt = segments[1]
            segments[1] = TrendSegment(seg.start_idx, nxt.end_idx, nxt.trend, 0.0)
            segments.pop(0)
        elif shortest_idx == len(segments) - 1:
            # Last segment: merge into previous
            prev = segments[-2]
            segments[-2] = TrendSegment(prev.start_idx, seg.end_idx, prev.trend, 0.0)
            segments.pop(-1)
        else:
            # Middle: merge into the longer neighbour
            prev = segments[shortest_idx - 1]
            nxt = segments[shortest_idx + 1]
            prev_dur = prev.end_idx - prev.start_idx + 1
            nxt_dur = nxt.end_idx - nxt.start_idx + 1
            if prev_dur >= nxt_dur:
                segments[shortest_idx - 1] = TrendSegment(
                    prev.start_idx, seg.end_idx, prev.trend, 0.0
                )
            else:
                segments[shortest_idx + 1] = TrendSegment(
                    seg.start_idx, nxt.end_idx, nxt.trend, 0.0
                )
            segments.pop(shortest_idx)

    # Merge consecutive segments with the same trend
    merged: list[TrendSegment] = [segments[0]]
    for seg in segments[1:]:
        if seg.trend == merged[-1].trend:
            merged[-1] = TrendSegment(
                merged[-1].start_idx, seg.end_idx, seg.trend, 0.0
            )
        else:
            merged.append(seg)

    return merged


@dataclass
class PatternResult:
    """Container for all detected patterns."""
    df: pl.DataFrame
    horizontal_zones: list[HorizontalZone] = field(default_factory=list)
    support_lines: list[TrendLine] = field(default_factory=list)
    resistance_lines: list[TrendLine] = field(default_factory=list)
    triangles: list[TrianglePattern] = field(default_factory=list)
    consolidation_zones: list[ConsolidationZone] = field(default_factory=list)
    current_trend: int = 0
    trend_slope: float = 0.0
    eps: float = 0.0  # ATR-based eps (for trendlines)
    sr_eps: float = 0.0  # Percentage-based eps (for S/R zones)
    replay_idx: Optional[int] = None  # Index of replay cutoff (None = no replay)
    ema: Optional[np.ndarray] = None  # EMA array for plotting
    trend_segments: list[TrendSegment] = field(default_factory=list)


def calculate_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Calculate Average True Range."""
    return df.with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs()
        ).alias("tr")
    ]).with_columns([
        pl.col("tr").rolling_mean(window_size=period).alias("atr")
    ])


def calculate_ema(closes: np.ndarray, span: int) -> np.ndarray:
    """Calculate EMA array."""
    n = len(closes)
    alpha = 2 / (span + 1)
    ema = np.zeros(n)
    ema[0] = closes[0]
    for i in range(1, n):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
    return ema


def calculate_trend(
    df: pl.DataFrame,
    ema_span: int = 50,
    threshold: float = 0.5,
    use_fast: bool = True
) -> tuple[int, float, np.ndarray]:
    """
    Calculate current trend using EMA.

    Two modes:
    1. Classic (use_fast=False): EMA slope over ema_span bars
       - More stable but slower to react
    2. Fast (use_fast=True): Combines price position vs EMA + short-term momentum
       - Price above EMA + rising over last 5 bars = uptrend
       - More reactive to recent price action

    Returns:
        (trend_direction, trend_slope_pct, ema_array)
        trend_direction: 1 = uptrend, -1 = downtrend, 0 = sideways
        trend_slope_pct: percentage change metric
        ema_array: the calculated EMA for plotting
    """
    closes = df["close"].to_numpy()
    n = len(closes)

    if n < ema_span:
        return 0, 0.0, np.full(n, np.nan)

    # Calculate EMA
    ema = calculate_ema(closes, ema_span)

    if use_fast:
        # Fast trend detection:
        # 1. Price position relative to EMA (current)
        # 2. Short-term momentum (last 5-10 bars)
        current_price = closes[-1]
        ema_now = ema[-1]

        # Price vs EMA: how far price is from EMA as % of EMA
        price_vs_ema_pct = (current_price - ema_now) / ema_now * 100

        # Short-term momentum: price change over last 5 bars
        lookback = min(5, n - 1)
        momentum_pct = (closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback] * 100

        # Combined score
        trend_score = price_vs_ema_pct * 0.6 + momentum_pct * 0.4

        # Classification with threshold
        if trend_score > threshold:
            trend_direction = 1
        elif trend_score < -threshold:
            trend_direction = -1
        else:
            trend_direction = 0

        return trend_direction, trend_score, ema

    else:
        # Classic: EMA slope over period
        ema_now = ema[-1]
        ema_prev = ema[-ema_span] if n > ema_span else ema[0]

        trend_slope = ((ema_now - ema_prev) / ema_prev * 100) if ema_prev else 0
        trend_direction = 1 if trend_slope > threshold else (-1 if trend_slope < -threshold else 0)

        return trend_direction, trend_slope, ema


def detect_local_extrema(
    df: pl.DataFrame,
    window_left: int,
    window_right: int,
    window_short: int = None,
    trend: int = 0,
    consolidation_zones: list[ConsolidationZone] = None
) -> pl.DataFrame:
    """
    Detect local highs and lows with trend-aware asymmetric windows.

    In an uptrend:
      - Local min (support): left=short, right=normal
        (we expect lower prices on left, so shorter lookback is fine;
         need normal right window to confirm the bounce)
      - Local max (resistance): left=normal, right=short
        (need normal left to confirm it's a peak;
         shorter right because price may continue up in trend)

    In a downtrend (mirror):
      - Local max: left=short, right=normal
      - Local min: left=normal, right=short

    In sideways OR inside consolidation zones: symmetric windows for both.

    Args:
        df: DataFrame with high/low columns
        window_left: Base left window size
        window_right: Base right window size
        window_short: Shortened window for trend direction (defaults to window_left)
        trend: 1=uptrend, -1=downtrend, 0=sideways
        consolidation_zones: List of consolidation zones - use symmetric windows inside
    """
    n = len(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    if window_short is None:
        window_short = window_left

    # Pre-compute which bars are in consolidation zones for O(1) lookup
    in_consolidation = np.zeros(n, dtype=bool)
    if consolidation_zones:
        for cz in consolidation_zones:
            in_consolidation[cz.start_idx:cz.end_idx + 1] = True

    is_local_high = np.zeros(n, dtype=bool)
    is_local_low = np.zeros(n, dtype=bool)

    for i in range(n):
        # Use symmetric windows inside consolidation zones (trend is unreliable there)
        local_trend = 0 if in_consolidation[i] else trend

        # Determine windows based on trend
        if local_trend == 1:  # Uptrend
            # Local max: normal left, short right (quicker to mark resistance)
            high_left, high_right = window_left, window_short
            # Local min: short left, normal right (easier to catch support)
            low_left, low_right = window_short, window_right
        elif local_trend == -1:  # Downtrend
            # Local max: short left, normal right (easier to catch resistance)
            high_left, high_right = window_short, window_right
            # Local min: normal left, short right (quicker to mark support)
            low_left, low_right = window_left, window_short
        else:  # Sideways or consolidation - symmetric
            high_left = low_left = window_left
            high_right = low_right = window_right

        # Check local high with its windows
        if i >= high_left:
            left_bound = max(0, i - high_left)
            right_bound = min(n - 1, i + high_right)
            window_highs = highs[left_bound:right_bound + 1]
            if highs[i] >= np.max(window_highs) - 1e-10:
                is_local_high[i] = True

        # Check local low with its windows
        if i >= low_left:
            left_bound = max(0, i - low_left)
            right_bound = min(n - 1, i + low_right)
            window_lows = lows[left_bound:right_bound + 1]
            if lows[i] <= np.min(window_lows) + 1e-10:
                is_local_low[i] = True

    df = df.with_columns([
        pl.Series("is_local_high", is_local_high),
        pl.Series("is_local_low", is_local_low),
    ])

    return df


def find_horizontal_zones(
    local_prices: np.ndarray,
    local_indices: np.ndarray,
    eps: float,
    current_idx: int,
    min_touches: int,
    merge_factor: float,
    zone_type: str
) -> list[HorizontalZone]:
    """Cluster price levels into horizontal zones."""
    if len(local_prices) < min_touches:
        return []

    # Sort by price
    sorted_idx = np.argsort(local_prices)
    sorted_prices = local_prices[sorted_idx]
    sorted_bar_indices = local_indices[sorted_idx]

    zones = []
    merge_dist = eps * merge_factor

    i = 0
    while i < len(sorted_prices):
        # Start a new cluster
        cluster_prices = [sorted_prices[i]]
        cluster_indices = [sorted_bar_indices[i]]

        j = i + 1
        while j < len(sorted_prices) and sorted_prices[j] - sorted_prices[i] <= merge_dist:
            cluster_prices.append(sorted_prices[j])
            cluster_indices.append(sorted_bar_indices[j])
            j += 1

        if len(cluster_prices) >= min_touches:
            price_low = min(cluster_prices)
            price_high = max(cluster_prices)
            price_center = np.mean(cluster_prices)

            # Strength based on touches and recency
            recency_weight = np.mean([(idx / current_idx) for idx in cluster_indices])
            strength = len(cluster_prices) * (0.5 + 0.5 * recency_weight)

            zones.append(HorizontalZone(
                price_low=price_center - eps * 0.5,
                price_high=price_center + eps * 0.5,
                price_center=price_center,
                start_idx=int(min(cluster_indices)),
                end_idx=int(max(cluster_indices)),
                touches=len(cluster_prices),
                zone_type=zone_type,
                strength=strength
            ))

        i = j

    return zones


def is_valid_line(
    prices: np.ndarray,
    x1: int, y1: float,
    slope: float,
    current_idx: int,
    eps: float,
    line_type: str,
    x2: int = None,
    break_factor: float = 0.3
) -> tuple[bool, int]:
    """
    Check if line is valid and count touches.
    Returns (is_valid, touch_count).
    
    This is the original validation function - more lenient than strict validation.
    """
    touches = 0
    break_tolerance = eps * break_factor
    end_idx = x2 if x2 is not None else current_idx

    for offset in range(end_idx - x1 + 1):
        bar_idx = x1 + offset
        if bar_idx >= len(prices):
            break

        line_y = y1 + slope * offset
        price = prices[bar_idx]

        if line_type == 'support':
            if price < line_y - break_tolerance:
                return False, 0
            if price >= line_y - break_tolerance and price <= line_y + eps:
                touches += 1
        else:  # resistance
            if price > line_y + break_tolerance:
                return False, 0
            if price <= line_y + break_tolerance and price >= line_y - eps:
                touches += 1

    return True, touches


def is_valid_trendline_ceiling(
    highs: np.ndarray,
    lows: np.ndarray,
    x1: int, y1: float,
    slope: float,
    end_idx: int,
    line_type: str,
    tolerance: float = 0.001
) -> tuple[bool, int]:
    """
    Strict non-crossing validation: line acts as ceiling (resistance) or floor (support).
    Resistance: for every candle i in [x1, end_idx], High[i] <= LineValue[i] * (1 + tolerance).
    Support: for every candle i in [x1, end_idx], Low[i] >= LineValue[i] * (1 - tolerance).
    If any bar pierces the line beyond tolerance, the trendline is invalid.
    Returns (is_valid, touch_count).
    """
    touches = 0
    for bar_idx in range(x1, min(end_idx + 1, len(highs))):
        line_val = y1 + slope * (bar_idx - x1)
        if line_val <= 0:
            continue
        if line_type == 'resistance':
            h = highs[bar_idx]
            if h > line_val * (1 + tolerance):
                return False, 0
            if abs(h - line_val) <= line_val * tolerance:
                touches += 1
        else:
            l = lows[bar_idx]
            if l < line_val * (1 - tolerance):
                return False, 0
            if abs(l - line_val) <= line_val * tolerance:
                touches += 1
    return True, touches


def find_extension_stop(
    highs: np.ndarray,
    lows: np.ndarray,
    x1: int, y1: float,
    slope: float,
    start_idx: int,
    end_idx: int,
    line_type: str,
    tolerance: float = 0.001
) -> int:
    """
    Find the last bar index where the line is still valid when extended.
    Returns the index to stop the line (inclusive). If no violation, returns end_idx.
    """
    stop = start_idx - 1
    for bar_idx in range(start_idx, min(end_idx + 1, len(highs))):
        line_val = y1 + slope * (bar_idx - x1)
        if line_val <= 0:
            stop = bar_idx
            continue
        if line_type == 'resistance':
            if highs[bar_idx] > line_val * (1 + tolerance):
                return bar_idx - 1
        else:
            if lows[bar_idx] < line_val * (1 - tolerance):
                return bar_idx - 1
        stop = bar_idx
    return stop


def find_left_anchor(
    highs: np.ndarray,
    lows: np.ndarray,
    x1: int, y1: float,
    slope: float,
    line_type: str,
    tolerance: float = 0.001
) -> tuple[int, float]:
    """
    For resistance: extend anchor left so the line starts as far left as possible
    without any bar piercing (no 毛刺). So the line "starts at the top" and goes right.
    Returns (new_anchor_x, new_anchor_y).
    """
    new_anchor_x = x1
    new_anchor_y = y1
    for candidate in range(x1 - 1, -1, -1):
        valid = True
        for bar_idx in range(candidate, min(x1 + 1, len(highs))):
            line_val = y1 + slope * (bar_idx - x1)
            if line_val <= 0:
                continue
            if line_type == 'resistance':
                if highs[bar_idx] > line_val * (1 + tolerance):
                    valid = False
                    break
            else:
                if lows[bar_idx] < line_val * (1 - tolerance):
                    valid = False
                    break
        if valid:
            new_anchor_x = candidate
            new_anchor_y = y1 + slope * (candidate - x1)
        else:
            break
    return new_anchor_x, new_anchor_y


def find_horizontal_trendlines(
    local_prices: np.ndarray,
    local_indices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    current_idx: int,
    current_price: float,
    line_type: str,
    price_tolerance_pct: float = 0.008,
    ceiling_tolerance: float = 0.001,
    min_bars_span: int = 2,
) -> list[TrendLine]:
    """
    Connect same-level highs (resistance) or lows (support) with horizontal lines.
    E.g. 15 Feb 09:00 high and 15 Feb 14:00 high at similar price → one horizontal resistance line.
    Clusters extrema by price (within price_tolerance_pct), then for each cluster with >= 2
    points spanning at least min_bars_span bars, emits one horizontal line and validates ceiling/floor.
    """
    if len(local_prices) < 2:
        return []

    sort_order = np.argsort(local_indices)
    pts_x = local_indices[sort_order].astype(float)
    pts_y = local_prices[sort_order]
    n_pts = len(pts_y)
    avg_price = float(np.mean(pts_y))
    if avg_price <= 0:
        return []
    merge_dist = avg_price * price_tolerance_pct

    # Cluster by price: points within merge_dist of the cluster's range (so 09:00 and 14:00 highs connect)
    clusters: list[list[int]] = []
    used = np.zeros(n_pts, dtype=bool)

    for i in range(n_pts):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        cluster_lo = pts_y[i] - merge_dist
        cluster_hi = pts_y[i] + merge_dist
        changed = True
        while changed:
            changed = False
            for j in range(n_pts):
                if used[j]:
                    continue
                if cluster_lo <= pts_y[j] <= cluster_hi:
                    cluster.append(j)
                    used[j] = True
                    cluster_lo = min(cluster_lo, pts_y[j] - merge_dist)
                    cluster_hi = max(cluster_hi, pts_y[j] + merge_dist)
                    changed = True
        if len(cluster) >= 2:
            clusters.append(cluster)

    lines = []
    for cluster in clusters:
        cluster_sorted = sorted(cluster, key=lambda k: pts_x[k])
        gx = pts_x[cluster_sorted]
        gy = pts_y[cluster_sorted]
        if np.max(gx) - np.min(gx) < min_bars_span:
            continue
        anchor_x = int(np.min(gx))
        last_anchor = int(np.max(gx))
        if line_type == 'resistance':
            anchor_y = float(np.max(gy))
        else:
            anchor_y = float(np.min(gy))
        slope = 0.0

        ceiling_valid, touches = is_valid_trendline_ceiling(
            highs, lows, anchor_x, anchor_y, slope, last_anchor,
            line_type, tolerance=ceiling_tolerance
        )
        if not ceiling_valid or touches < 2:
            continue

        anchor_x, anchor_y = find_left_anchor(
            highs, lows, anchor_x, anchor_y, slope, line_type, tolerance=ceiling_tolerance
        )
        end_x = min(current_idx + 30, len(highs) - 1)
        ext_stop = find_extension_stop(
            highs, lows, anchor_x, anchor_y, slope,
            last_anchor + 1, end_x, line_type, tolerance=ceiling_tolerance
        )
        end_x = ext_stop
        end_y = anchor_y

        span = last_anchor - anchor_x
        span_score = span / current_idx if current_idx > 0 else 0
        recency = last_anchor / current_idx if current_idx > 0 else 0
        distance_pct = abs(anchor_y - current_price) / current_price if current_price > 0 else 0
        proximity = 1.0 / (1.0 + distance_pct * 10)
        strength = touches * (0.3 * span_score + 0.3 * recency + 0.4 * proximity)

        lines.append(TrendLine(
            x1=int(anchor_x), y1=anchor_y,
            x2=end_x, y2=end_y,
            slope=0.0,
            line_type=line_type,
            touches=touches,
            strength=strength,
            tolerance=ceiling_tolerance * anchor_y
        ))

    return lines


def is_valid_line_strict(
    highs: np.ndarray,
    lows: np.ndarray,
    x1: int, y1: float,
    slope: float,
    current_idx: int,
    tolerance_pct: float,
    line_type: str,
    x2: int = None,
    atr_array: np.ndarray = None,
    k_atr: float = 0.5,
    fixed_pct: float = 0.003
) -> tuple[bool, int]:
    """
    Strict Pivot Validation: Check that NO intermediate candle pierces the line.
    
    For resistance lines: ALL candles must have High <= Trendline Value + Tolerance
    For support lines: ALL candles must have Low >= Trendline Value - Tolerance
    
    Uses dynamic tolerance: max(fixed_pct * price, k_atr * ATR)
    
    Args:
        highs: Array of high prices for all bars
        lows: Array of low prices for all bars
        x1, y1: Starting point of the line
        slope: Slope of the line (price change per bar)
        current_idx: Current bar index (validation goes from x1 to current_idx)
        tolerance_pct: Base tolerance percentage (e.g., 0.003 = 0.3%)
        line_type: 'support' or 'resistance'
        x2: Optional end point for validation (defaults to current_idx)
        atr_array: Optional ATR array for dynamic tolerance
        k_atr: ATR multiplier for dynamic tolerance
        fixed_pct: Fixed percentage for dynamic tolerance (default 0.3%)
    
    Returns:
        (is_valid, touch_count)
    """
    touches = 0
    end_idx = x2 if x2 is not None else current_idx
    
    # Validate all bars between x1 and end_idx (and slightly after for safety)
    check_end = min(end_idx + 3, len(highs), current_idx + 1)
    
    for bar_idx in range(x1, check_end):
        if bar_idx >= len(highs):
            break
        
        # Calculate line value at this bar
        line_y = y1 + slope * (bar_idx - x1)
        
        # Dynamic tolerance: max(fixed_pct * price, k_atr * ATR)
        # Use the line value at this bar for tolerance calculation (more accurate)
        price_at_bar = abs(line_y)  # Use abs of line value for tolerance calculation
        fixed_tolerance = fixed_pct * price_at_bar
        
        if atr_array is not None and bar_idx < len(atr_array) and not np.isnan(atr_array[bar_idx]):
            atr_tolerance = k_atr * atr_array[bar_idx]
            tolerance = max(fixed_tolerance, atr_tolerance)
        else:
            tolerance = fixed_tolerance
        
        # Ensure minimum tolerance to avoid rejecting all lines (at least 0.1% of price)
        min_tolerance = price_at_bar * 0.001
        tolerance = max(tolerance, min_tolerance)
        
        if line_type == 'support':
            # For support: Low must be >= line - tolerance (strict ceiling)
            low_at_bar = lows[bar_idx]
            if low_at_bar < line_y - tolerance:
                return False, 0  # Line pierced below - invalid
            
            # Count touch if low is near the line (within tolerance)
            if low_at_bar >= line_y - tolerance and low_at_bar <= line_y + tolerance:
                touches += 1
        else:  # resistance
            # For resistance: High must be <= line + tolerance (strict ceiling)
            high_at_bar = highs[bar_idx]
            if high_at_bar > line_y + tolerance:
                return False, 0  # Line pierced above - invalid
            
            # Count touch if high is near the line (within tolerance)
            if high_at_bar >= line_y - tolerance and high_at_bar <= line_y + tolerance:
                touches += 1
    
    return True, touches


def find_trendlines(
    local_prices: np.ndarray,
    local_indices: np.ndarray,
    all_prices: np.ndarray,  # lows for support, highs for resistance
    eps: float,
    current_idx: int,
    current_price: float,
    line_type: str,
    max_lines: int,
    slope_tolerance: float,
    required_slope_sign: Optional[int] = None,  # 1 for positive, -1 for negative, None for any
    break_factor: float = 0.3,
    highs: np.ndarray = None,  # Full high array for strict validation
    lows: np.ndarray = None,   # Full low array for strict validation
    atr_array: np.ndarray = None,  # ATR array for dynamic tolerance
    tolerance_percent: float = 0.003,  # Base tolerance percentage
    k_atr: float = 0.5,  # ATR multiplier
    fixed_pct: float = 0.003,  # Fixed percentage
    use_strict_validation: bool = True,  # Use strict pivot validation
    ceiling_tolerance: float = 0.001  # Strict non-crossing: High <= line*(1+tol) for resistance
) -> list[TrendLine]:
    """
    Find valid trend lines with slope constraints.

    For uptrend context: support lines should have slope > 0 (required_slope_sign=1)
    For downtrend context: resistance lines should have slope < 0 (required_slope_sign=-1)
    """
    if len(local_prices) < 2:
        return []

    n_pts = len(local_prices)

    # ================================================================
    # Step 1: For each pair (i, j), find all collinear points.
    #   A point k is collinear with line(i, j) if its distance to the
    #   line is < eps. Group collinear points, then for each group emit
    #   ONE line from the earliest point with a best-fit slope.
    # ================================================================

    # Build sorted arrays (by bar index) for consistent ordering
    sort_order = np.argsort(local_indices)
    pts_x = local_indices[sort_order].astype(float)
    pts_y = local_prices[sort_order]

    # Track which groups we've already emitted (as frozensets of indices)
    seen_groups = set()
    lines = []

    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = pts_x[i], pts_y[i]
            x2, y2 = pts_x[j], pts_y[j]

            if x1 >= x2:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Check slope constraint
            if required_slope_sign is not None:
                if required_slope_sign == 1 and slope <= 0:
                    continue
                if required_slope_sign == -1 and slope >= 0:
                    continue

            # Find collinear points: within eps of the line AND at or after x1.
            # Only extend the group forward (>= x1) to avoid pulling in
            # earlier points that happen to be near the line by coincidence
            # but belong to a different trajectory.
            group = [i, j]
            for k in range(n_pts):
                if k == i or k == j:
                    continue
                xk, yk = pts_x[k], pts_y[k]
                if xk < x1:
                    continue  # don't extend backwards before the anchor
                line_y_at_k = y1 + slope * (xk - x1)
                if abs(yk - line_y_at_k) < eps:
                    group.append(k)

            # Deduplicate: skip if we've seen this exact set of points
            group_key = frozenset(group)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)

            # Build the line from the earliest point.
            # If 3+ collinear points, use least-squares for the slope.
            group_sorted = sorted(group)
            gx = pts_x[group_sorted]
            gy = pts_y[group_sorted]

            anchor_x = gx[0]
            anchor_y = gy[0]

            intercept = None
            if len(group_sorted) >= 3:
                # Least-squares fit through all collinear points
                slope, intercept = np.polyfit(gx, gy, 1)
                anchor_y = intercept + slope * anchor_x
            # Resistance: anchor at the highest point in the group so line "starts at the top"
            if line_type == 'resistance':
                top_idx = int(np.argmax(gy))
                anchor_x = gx[top_idx]
                anchor_y = (intercept + slope * anchor_x) if intercept is not None else gy[top_idx]

            # Primary gate: lenient is_valid_line (ensures lines render)
            valid, touches = is_valid_line(
                all_prices, int(anchor_x), anchor_y, slope, current_idx, eps,
                line_type, x2=int(gx[-1]), break_factor=break_factor
            )
            if not valid or touches < 2:
                continue

            last_anchor = int(gx[-1])
            end_x = current_idx + 30
            end_y = anchor_y + slope * (end_x - anchor_x)

            # Ceiling/floor rule: no 毛刺 (candles piercing the line).
            # For resistance: require ceiling validation and trim extension so line never extends past a pierce.
            if use_strict_validation and highs is not None and lows is not None:
                ceiling_valid, ceiling_touches = is_valid_trendline_ceiling(
                    highs, lows, int(anchor_x), anchor_y, slope, last_anchor,
                    line_type, tolerance=ceiling_tolerance
                )
                # Resistance: only keep line if it passes ceiling (no candles sticking out above)
                if line_type == 'resistance' and (not ceiling_valid or ceiling_touches < 2):
                    continue
                if line_type == 'resistance' and ceiling_valid and ceiling_touches >= 2:
                    touches = ceiling_touches
                    # Extend anchor left so line "starts at the top" without left-side 毛刺
                    anchor_x, anchor_y = find_left_anchor(
                        highs, lows, int(anchor_x), anchor_y, slope,
                        line_type, tolerance=ceiling_tolerance
                    )
                # Always trim right extension at first pierce (so no 毛刺 on the right)
                ext_stop = find_extension_stop(
                    highs, lows, int(anchor_x), anchor_y, slope,
                    last_anchor + 1, current_idx + 30, line_type, tolerance=ceiling_tolerance
                )
                if ext_stop < end_x:
                    end_x = ext_stop
                    end_y = anchor_y + slope * (end_x - anchor_x)

            # Line price at current bar
            line_at_current = anchor_y + slope * (current_idx - anchor_x)

            # ============================================================
            # STRENGTH CALCULATION
            # ============================================================
            # Components (each normalized to ~0-1 range):
            #   1. touches: how many bars are near the line (>= 2)
            #   2. span_score: (last_anchor - first_anchor) / current_idx
            #   3. recency: last_anchor / current_idx
            #   4. proximity: inverse distance of line to current price
            #
            # Formula: strength = touches * (0.2*span + 0.2*recency + 0.6*proximity)
            # ============================================================
            span = gx[-1] - gx[0]
            span_score = span / current_idx
            recency = gx[-1] / current_idx

            distance_pct = abs(line_at_current - current_price) / current_price
            proximity = 1.0 / (1.0 + distance_pct * 10)

            strength = touches * (0.2 * span_score + 0.2 * recency + 0.6 * proximity)

            # Calculate average tolerance for this line (for zone display)
            avg_tolerance = tolerance_percent * current_price
            if atr_array is not None:
                # Use average ATR in the line's range
                start_atr = int(anchor_x)
                end_atr = min(int(gx[-1]) + 1, len(atr_array))
                if end_atr > start_atr:
                    line_atr = atr_array[start_atr:end_atr]
                    line_atr = line_atr[~np.isnan(line_atr)]
                    if len(line_atr) > 0:
                        avg_atr = np.mean(line_atr)
                        avg_tolerance = max(fixed_pct * current_price, k_atr * avg_atr)
            
            lines.append(TrendLine(
                x1=int(anchor_x), y1=anchor_y,
                x2=end_x, y2=end_y,
                slope=slope,
                line_type=line_type,
                touches=touches,
                strength=strength,
                tolerance=avg_tolerance
            ))

    # ================================================================
    # Step 2: Merge collinear lines.
    #   Two lines are collinear if their y-values at two reference
    #   points (current_idx and midpoint) are both within eps.
    #   Keep the one with higher strength.
    # ================================================================
    lines.sort(key=lambda x: x.strength, reverse=True)

    merged = []
    for line in lines:
        is_collinear = False
        for kept in merged:
            # Compare y-values at current bar and at the midpoint of overlap
            mid_x = (max(line.x1, kept.x1) + current_idx) // 2
            y_line_at_cur = line.y1 + line.slope * (current_idx - line.x1)
            y_kept_at_cur = kept.y1 + kept.slope * (current_idx - kept.x1)
            y_line_at_mid = line.y1 + line.slope * (mid_x - line.x1)
            y_kept_at_mid = kept.y1 + kept.slope * (mid_x - kept.x1)

            if abs(y_line_at_cur - y_kept_at_cur) < eps and abs(y_line_at_mid - y_kept_at_mid) < eps:
                is_collinear = True
                # If the new line has more touches, update the kept line's touches/strength
                if line.touches > kept.touches:
                    kept.touches = line.touches
                    kept.strength = max(kept.strength, line.strength)
                break
        if not is_collinear:
            merged.append(line)

    return merged[:max_lines]


def find_triangles(
    support_lines: list[TrendLine],
    resistance_lines: list[TrendLine],
    current_idx: int,
    current_price: float,
    eps: float,
    slope_tolerance: float
) -> list[TrianglePattern]:
    """Detect triangle patterns from converging lines."""
    triangles = []

    for sup in support_lines:
        for res in resistance_lines:
            # Lines must overlap in time
            if sup.x1 > res.x2 or res.x1 > sup.x2:
                continue

            # Check if lines converge (will intersect in the future)
            slope_diff = res.slope - sup.slope

            if abs(slope_diff) < 1e-10:
                continue  # Parallel lines

            # Find intersection point
            # sup: y = sup.y1 + sup.slope * (x - sup.x1)
            # res: y = res.y1 + res.slope * (x - res.x1)
            # Solve for x:
            # sup.y1 + sup.slope * (x - sup.x1) = res.y1 + res.slope * (x - res.x1)
            x_intersect = (res.y1 - sup.y1 + sup.slope * sup.x1 - res.slope * res.x1) / slope_diff
            y_intersect = sup.y1 + sup.slope * (x_intersect - sup.x1)

            # Apex should be in the future but not too far
            if x_intersect <= current_idx or x_intersect > current_idx + 100:
                continue

            # Support should be below resistance at current price level
            sup_at_current = sup.y1 + sup.slope * (current_idx - sup.x1)
            res_at_current = res.y1 + res.slope * (current_idx - res.x1)

            if sup_at_current >= res_at_current:
                continue  # Invalid - support above resistance

            # Classify pattern
            sup_is_flat = abs(sup.slope) < slope_tolerance
            res_is_flat = abs(res.slope) < slope_tolerance
            sup_rising = sup.slope > slope_tolerance
            sup_falling = sup.slope < -slope_tolerance
            res_rising = res.slope > slope_tolerance
            res_falling = res.slope < -slope_tolerance

            if sup_rising and res_is_flat:
                pattern_type = "ascending"
                bias = "bullish"
            elif sup_is_flat and res_falling:
                pattern_type = "descending"
                bias = "bearish"
            elif sup_rising and res_falling:
                pattern_type = "symmetrical"
                bias = "neutral"
            elif sup_rising and res_rising and res.slope > sup.slope:
                pattern_type = "rising_wedge"
                bias = "bearish"
            elif sup_falling and res_falling and sup.slope < res.slope:
                pattern_type = "falling_wedge"
                bias = "bullish"
            else:
                continue  # Not a recognized pattern

            # Calculate completion percentage
            pattern_start = max(sup.x1, res.x1)
            total_span = x_intersect - pattern_start
            current_progress = current_idx - pattern_start
            completion = min(100, max(0, (current_progress / total_span) * 100))

            triangles.append(TrianglePattern(
                support_line=sup,
                resistance_line=res,
                apex_x=int(x_intersect),
                apex_price=y_intersect,
                pattern_type=pattern_type,
                breakout_bias=bias,
                completion_pct=completion
            ))

    # Sort by completion (most complete first)
    triangles.sort(key=lambda x: x.completion_pct, reverse=True)
    return triangles[:5]  # Limit to top 5


def detect_consolidation_zones(
    df: pl.DataFrame,
    window: int,
    range_pct_threshold: float,  # Max channel width as percentage (e.g., 0.05 = 5%)
    slope_threshold: float,
    min_duration: int,
    efficiency_max: float = 0.35,
) -> list[ConsolidationZone]:
    """
    Detect consolidation zones based on channel tightness, trend flatness,
    and low directional efficiency.

    Per-window conditions (all must hold):
    1. Tightness: (rolling_max - rolling_min) / rolling_min <= range_pct_threshold
    2. Flatness:  |normalized slope| <= slope_threshold
    3. Choppiness: efficiency ratio <= efficiency_max
       ER = |net move| / total absolute bar-to-bar movement.
       ER ≈ 0 → price goes nowhere (consolidation).
       ER ≈ 1 → straight-line trend.

    Rolling computations are vectorized (no Python loops per bar).

    After detecting a consolidation zone, extends it backwards/forwards to find
    where price first entered the zone's price range, then re-validates the
    full zone's range, slope, and efficiency.
    """
    n = len(df)
    if n < window:
        return []

    highs = df["high"].to_numpy().astype(np.float64)
    lows = df["low"].to_numpy().astype(np.float64)
    closes = df["close"].to_numpy().astype(np.float64)

    w = window

    # ------------------------------------------------------------------
    # 1. Rolling max(high) and min(low) — vectorized via sliding_window_view
    # ------------------------------------------------------------------
    high_windows = np.lib.stride_tricks.sliding_window_view(highs, w)  # shape (n-w+1, w)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows, w)

    rolling_max = np.full(n, np.nan)
    rolling_min = np.full(n, np.nan)
    rolling_max[w - 1:] = high_windows.max(axis=1)
    rolling_min[w - 1:] = low_windows.min(axis=1)

    channel_width_pct = (rolling_max - rolling_min) / rolling_min

    # ------------------------------------------------------------------
    # 2. Rolling normalized slope — closed-form linear regression
    #    slope_raw = (w * Sxy - Sx * Sy) / denom
    #    slope_norm = slope_raw / y[0]   (first close in window)
    #
    #    Sx, Sx2, denom are constants (x = 0..w-1).
    #    Sy and Sxy use cumulative sums for O(n) computation.
    # ------------------------------------------------------------------
    Sx = w * (w - 1) / 2.0
    Sx2 = w * (w - 1) * (2 * w - 1) / 6.0
    denom = w * Sx2 - Sx * Sx  # constant

    # Cumulative sums (prepend 0 for easy range queries)
    cum_c = np.concatenate(([0.0], np.cumsum(closes)))           # Σ closes
    indices = np.arange(n, dtype=np.float64)
    cum_ic = np.concatenate(([0.0], np.cumsum(indices * closes)))  # Σ (k * closes[k])

    # For window ending at bar i (i >= w-1):
    #   window spans [i-w+1 .. i]
    #   Sy  = cum_c[i+1] - cum_c[i-w+1]
    #   Sic = cum_ic[i+1] - cum_ic[i-w+1]  (sum of k*closes[k])
    #   Sxy = Sic - (i-w+1) * Sy            (rebase x to start at 0)
    valid = np.arange(w - 1, n)
    Sy = cum_c[valid + 1] - cum_c[valid - w + 1]
    Sic = cum_ic[valid + 1] - cum_ic[valid - w + 1]
    window_start = valid - w + 1
    Sxy = Sic - window_start.astype(np.float64) * Sy

    slope_raw = (w * Sxy - Sx * Sy) / denom

    # Normalize by the first close in each window
    first_close = closes[window_start]
    safe_first = np.where(first_close != 0, first_close, 1.0)
    slope_norm_arr = slope_raw / safe_first

    slopes = np.full(n, np.nan)
    slopes[w - 1:] = slope_norm_arr

    # ------------------------------------------------------------------
    # 3. Rolling efficiency ratio — vectorized
    #    ER = |close[end] - close[start]| / Σ|close[i] - close[i-1]|
    # ------------------------------------------------------------------
    abs_diffs = np.abs(np.diff(closes))  # length n-1
    cum_abs = np.concatenate(([0.0], np.cumsum(abs_diffs)))  # length n

    # For window [i-w+1 .. i]: net = |closes[i] - closes[i-w+1]|
    #   total_path = cum_abs[i] - cum_abs[i-w+1]
    #   (cum_abs[k] = sum of abs_diffs[0..k-1] = path from bar 0 to bar k)
    net_move = np.abs(closes[valid] - closes[window_start])
    total_path = cum_abs[valid] - cum_abs[window_start]
    safe_path = np.where(total_path > 0, total_path, 1.0)
    er_arr = net_move / safe_path

    efficiency = np.full(n, np.nan)
    efficiency[w - 1:] = er_arr

    # ------------------------------------------------------------------
    # 4. Mark consolidation bars (all three conditions)
    # ------------------------------------------------------------------
    is_consolidation = (
        (channel_width_pct <= range_pct_threshold) &
        (np.abs(slopes) <= slope_threshold) &
        (efficiency <= efficiency_max) &
        ~np.isnan(channel_width_pct)
    )

    # ------------------------------------------------------------------
    # 5. Merge consecutive bars into zones, extend, and validate
    # ------------------------------------------------------------------
    zones = []
    i = 0
    while i < n:
        if is_consolidation[i]:
            start_idx = i
            while i < n and is_consolidation[i]:
                i += 1
            end_idx = i - 1
            duration = end_idx - start_idx + 1

            if duration >= min_duration:
                zone_high = np.max(highs[start_idx:end_idx + 1])
                zone_low = np.min(lows[start_idx:end_idx + 1])

                # Extend backwards
                extended_start = start_idx
                for j in range(start_idx - 1, -1, -1):
                    if lows[j] <= zone_high and highs[j] >= zone_low:
                        extended_start = j
                    else:
                        break

                # Extend forwards
                extended_end = end_idx
                for j in range(end_idx + 1, n):
                    if lows[j] <= zone_high and highs[j] >= zone_low:
                        extended_end = j
                    else:
                        break
                end_idx = extended_end
                i = end_idx + 1

                # Recalculate boundaries with extended range
                zone_high = np.max(highs[extended_start:end_idx + 1])
                zone_low = np.min(lows[extended_start:end_idx + 1])
                zone_center = (zone_high + zone_low) / 2
                width_pct = (zone_high - zone_low) / zone_low if zone_low > 0 else 0
                duration = end_idx - extended_start + 1

                # --- Post-extension validation ---
                # 1. Range check
                if width_pct > range_pct_threshold:
                    continue
                # 2. Overall slope
                zone_closes = closes[extended_start:end_idx + 1]
                zn = len(zone_closes)
                if zn >= 2:
                    zx = np.arange(zn, dtype=np.float64)
                    zy = zone_closes / zone_closes[0] if zone_closes[0] != 0 else zone_closes
                    z_Sx = zn * (zn - 1) / 2.0
                    z_Sx2 = zn * (zn - 1) * (2 * zn - 1) / 6.0
                    z_denom = zn * z_Sx2 - z_Sx * z_Sx
                    z_Sy = zy.sum()
                    z_Sxy = (zx * zy).sum()
                    overall_slope = (zn * z_Sxy - z_Sx * z_Sy) / z_denom if z_denom != 0 else 0
                    if abs(overall_slope) > slope_threshold:
                        continue
                # 3. Overall efficiency ratio
                if zn >= 2:
                    zone_net = abs(zone_closes[-1] - zone_closes[0])
                    zone_path = np.sum(np.abs(np.diff(zone_closes)))
                    zone_er = zone_net / zone_path if zone_path > 0 else 0
                    if zone_er > efficiency_max:
                        continue

                # Average slope (from per-window values)
                valid_slopes = slopes[start_idx:end_idx + 1]
                valid_slopes = valid_slopes[~np.isnan(valid_slopes)]
                avg_slope = np.mean(valid_slopes) if len(valid_slopes) > 0 else 0

                zones.append(ConsolidationZone(
                    start_idx=extended_start,
                    end_idx=end_idx,
                    price_low=zone_low,
                    price_high=zone_high,
                    price_center=zone_center,
                    channel_width_pct=width_pct,
                    avg_slope=avg_slope,
                    duration=duration
                ))
        else:
            i += 1

    return zones


def detect_patterns(
    df: pl.DataFrame,
    params: SRParams = None,
    replay_idx: Optional[int] = None
) -> PatternResult:
    """
    Main function to detect all patterns.

    If replay_idx is provided, patterns are detected as if that bar is the "current" bar.
    The full dataframe is returned for plotting, but pattern detection only uses data up to replay_idx.
    """
    if params is None:
        params = SRParams()

    # Add bar index to full dataframe
    df = df.with_row_index("bar_index")

    # For pattern detection, use only data up to replay_idx (or all data if None)
    if replay_idx is not None:
        df_analysis = df.filter(pl.col("bar_index") <= replay_idx)
    else:
        df_analysis = df

    # Calculate ATR-based eps (for trendlines)
    df_analysis = calculate_atr(df_analysis, params.atr_period)
    current_eps = df_analysis["atr"][-1] * params.atr_multiplier
    if np.isnan(current_eps) or current_eps <= 0:
        atr_dropped = df_analysis["atr"].drop_nulls()
        current_eps = float(atr_dropped[-1] * params.atr_multiplier) if len(atr_dropped) > 0 else None
    if current_eps is None or np.isnan(current_eps) or current_eps <= 0:
        current_eps = float(df_analysis["close"][-1]) * 0.005  # 0.5% fallback so trendlines can still be found

    # Calculate percentage-based eps for S/R zones
    current_price = df_analysis["close"][-1]
    sr_eps = current_price * params.sr_eps_pct

    # Calculate trend (needed for trend-aware extrema detection)
    current_trend, trend_slope, ema = calculate_trend(
        df_analysis,
        ema_span=params.ema_span,
        threshold=params.trend_threshold,
        use_fast=params.use_fast_trend
    )

    # Get arrays
    current_idx = len(df_analysis) - 1
    min_bar = 0 if params.lookback is None else max(0, current_idx - params.lookback)

    highs = df_analysis["high"].to_numpy()
    lows = df_analysis["low"].to_numpy()

    # Detect consolidation zones FIRST (needed for extrema detection)
    consolidation_zones = detect_consolidation_zones(
        df_analysis,
        window=params.consol_window,
        range_pct_threshold=params.consol_range_pct,
        slope_threshold=params.consol_slope_threshold,
        min_duration=params.consol_min_duration,
        efficiency_max=params.consol_efficiency_max,
    )

    # Detect local extrema with trend-aware asymmetric windows
    # Pass consolidation zones to use symmetric windows inside them
    df_analysis = detect_local_extrema(
        df_analysis,
        params.window_left,
        params.window_right,
        window_short=params.window_short,
        trend=current_trend,
        consolidation_zones=consolidation_zones
    )

    # Get local highs/lows within lookback
    local_highs_df = df_analysis.filter(
        (pl.col("is_local_high")) & (pl.col("bar_index") >= min_bar)
    )
    local_lows_df = df_analysis.filter(
        (pl.col("is_local_low")) & (pl.col("bar_index") >= min_bar)
    )

    local_high_prices = local_highs_df["high"].to_numpy()
    local_high_indices = local_highs_df["bar_index"].to_numpy()
    local_low_prices = local_lows_df["low"].to_numpy()
    local_low_indices = local_lows_df["bar_index"].to_numpy()
    
    # Filter by prominence if ATR is available (optional, less aggressive)
    # Only apply if we have enough extrema to filter
    if (params.prominence_multiplier > 0 and 
        "atr" in df_analysis.columns and 
        len(local_high_prices) > 5 and len(local_low_prices) > 5):
        try:
            atr_values = df_analysis["atr"].to_numpy()
            current_atr = atr_values[-1] if len(atr_values) > 0 else None
            
            if current_atr is not None and not np.isnan(current_atr):
                prominence_threshold = params.prominence_multiplier * current_atr
                
                # Filter highs by prominence (height relative to neighboring lows)
                filtered_highs = []
                filtered_high_indices = []
                for i, (price, idx) in enumerate(zip(local_high_prices, local_high_indices)):
                    # Find nearest low before and after
                    before_lows = local_low_prices[local_low_indices < idx]
                    after_lows = local_low_prices[local_low_indices > idx]
                    
                    if len(before_lows) > 0 and len(after_lows) > 0:
                        min_before = np.min(before_lows)
                        min_after = np.min(after_lows)
                        prominence = price - max(min_before, min_after)
                        
                        if prominence >= prominence_threshold:
                            filtered_highs.append(price)
                            filtered_high_indices.append(idx)
                    else:
                        # Keep if we can't calculate prominence (edge cases)
                        filtered_highs.append(price)
                        filtered_high_indices.append(idx)
                
                # Filter lows by prominence
                filtered_lows = []
                filtered_low_indices = []
                for i, (price, idx) in enumerate(zip(local_low_prices, local_low_indices)):
                    # Find nearest high before and after
                    before_highs = local_high_prices[local_high_indices < idx]
                    after_highs = local_high_prices[local_high_indices > idx]
                    
                    if len(before_highs) > 0 and len(after_highs) > 0:
                        max_before = np.max(before_highs)
                        max_after = np.max(after_highs)
                        prominence = min(max_before, max_after) - price
                        
                        if prominence >= prominence_threshold:
                            filtered_lows.append(price)
                            filtered_low_indices.append(idx)
                    else:
                        # Keep if we can't calculate prominence (edge cases)
                        filtered_lows.append(price)
                        filtered_low_indices.append(idx)
                
                # Only apply filtering if we still have enough extrema
                if len(filtered_highs) >= 2:
                    local_high_prices = np.array(filtered_highs)
                    local_high_indices = np.array(filtered_high_indices)
                if len(filtered_lows) >= 2:
                    local_low_prices = np.array(filtered_lows)
                    local_low_indices = np.array(filtered_low_indices)
        except Exception:
            # If prominence filtering fails, keep original extrema
            pass

    # Filter extrema in consolidation zones: cluster by price within sr_eps
    # For extrema outside consolidation zones, keep all of them
    def filter_extrema_for_consolidation(prices, indices, is_high=True):
        """
        Within consolidation zones, cluster extrema by price (within sr_eps).
        For each cluster, keep only the most extreme value to avoid duplicate S/R lines.
        """
        filtered_prices = []
        filtered_indices = []
        used_in_consol = set()

        # Collect extrema per consolidation zone
        consol_extrema = {}  # zone_idx -> list of (price, bar_idx)
        for i, (price, idx) in enumerate(zip(prices, indices)):
            for z_idx, cz in enumerate(consolidation_zones):
                if cz.start_idx <= idx <= cz.end_idx:
                    used_in_consol.add(i)
                    if z_idx not in consol_extrema:
                        consol_extrema[z_idx] = []
                    consol_extrema[z_idx].append((price, idx))
                    break

        # Add extrema outside consolidation zones (unchanged)
        for i, (price, idx) in enumerate(zip(prices, indices)):
            if i not in used_in_consol:
                filtered_prices.append(price)
                filtered_indices.append(idx)

        # For each consolidation zone, cluster extrema by price and keep one per cluster
        for z_idx, extrema_list in consol_extrema.items():
            if not extrema_list:
                continue

            # Sort by price
            sorted_extrema = sorted(extrema_list, key=lambda x: x[0])

            # Cluster by sr_eps
            clusters = []
            current_cluster = [sorted_extrema[0]]

            for price, idx in sorted_extrema[1:]:
                # Check if within sr_eps of any price in current cluster
                if price - current_cluster[0][0] <= sr_eps:
                    current_cluster.append((price, idx))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(price, idx)]
            clusters.append(current_cluster)

            # For each cluster, keep only the most extreme value
            for cluster in clusters:
                if is_high:
                    # Keep the highest high
                    best = max(cluster, key=lambda x: x[0])
                else:
                    # Keep the lowest low
                    best = min(cluster, key=lambda x: x[0])
                filtered_prices.append(best[0])
                filtered_indices.append(best[1])

        return np.array(filtered_prices), np.array(filtered_indices)

    # Apply filtering for S/R zone detection
    sr_high_prices, sr_high_indices = filter_extrema_for_consolidation(
        local_high_prices, local_high_indices, is_high=True
    )
    sr_low_prices, sr_low_indices = filter_extrema_for_consolidation(
        local_low_prices, local_low_indices, is_high=False
    )

    # Find horizontal zones (using percentage-based eps) with filtered extrema
    support_zones = find_horizontal_zones(
        sr_low_prices, sr_low_indices, sr_eps, current_idx,
        params.min_touches, params.zone_merge_factor, "support"
    )
    resistance_zones = find_horizontal_zones(
        sr_high_prices, sr_high_indices, sr_eps, current_idx,
        params.min_touches, params.zone_merge_factor, "resistance"
    )

    # Find trend lines - no slope restriction so both support and resistance appear
    support_slope_sign = None
    resistance_slope_sign = None
    support_max = params.max_lines
    resistance_max = params.max_lines

    # Detect volatility for tolerance adjustment
    current_atr = None
    atr_array = None
    if "atr" in df_analysis.columns:
        atr_values = df_analysis["atr"].to_numpy()
        atr_array = atr_values
        # Get last valid ATR
        valid_atrs = atr_values[~np.isnan(atr_values)]
        if len(valid_atrs) > 0:
            current_atr = valid_atrs[-1]
    
    if current_atr is not None and current_price > 0:
        atr_ratio = current_atr / current_price
        is_high_vol = atr_ratio > params.high_volatility_threshold
        tolerance_pct = params.high_vol_tolerance_pct if is_high_vol else params.tolerance_percent
    else:
        tolerance_pct = params.tolerance_percent
    
    ceiling_tol = getattr(params, 'ceiling_tolerance', 0.001)
    support_lines = find_trendlines(
        local_low_prices, local_low_indices, lows, current_eps,
        current_idx, current_price, "support", support_max, params.slope_tolerance,
        required_slope_sign=support_slope_sign,
        break_factor=params.line_break_factor,
        highs=highs, lows=lows, atr_array=atr_array,
        tolerance_percent=tolerance_pct, k_atr=params.k_atr, fixed_pct=params.fixed_pct,
        use_strict_validation=True, ceiling_tolerance=ceiling_tol
    )
    resistance_lines = find_trendlines(
        local_high_prices, local_high_indices, highs, current_eps,
        current_idx, current_price, "resistance", resistance_max, params.slope_tolerance,
        required_slope_sign=resistance_slope_sign,
        break_factor=params.line_break_factor,
        highs=highs, lows=lows, atr_array=atr_array,
        tolerance_percent=tolerance_pct, k_atr=params.k_atr, fixed_pct=params.fixed_pct,
        use_strict_validation=True, ceiling_tolerance=ceiling_tol
    )
    # Horizontal lines: connect same-day / same-level highs (e.g. 15 Feb 09:00 and 14:00) into one line
    horizontal_res = find_horizontal_trendlines(
        local_high_prices, local_high_indices, highs, lows, current_idx, current_price,
        "resistance", price_tolerance_pct=0.008, ceiling_tolerance=ceiling_tol, min_bars_span=2
    )
    resistance_lines = (horizontal_res + resistance_lines)
    resistance_lines.sort(key=lambda x: x.strength, reverse=True)
    resistance_lines = resistance_lines[:resistance_max]

    horizontal_sup = find_horizontal_trendlines(
        local_low_prices, local_low_indices, highs, lows, current_idx, current_price,
        "support", price_tolerance_pct=0.008, ceiling_tolerance=ceiling_tol, min_bars_span=2
    )
    support_lines = (horizontal_sup + support_lines)
    support_lines.sort(key=lambda x: x.strength, reverse=True)
    support_lines = support_lines[:support_max]

    # Find triangles
    triangles = find_triangles(
        support_lines, resistance_lines, current_idx,
        df_analysis["close"][-1], current_eps, params.slope_tolerance
    )

    # consolidation_zones already detected earlier (before extrema detection)

    # Add local extrema columns to full dataframe for plotting
    # (we need to recalculate for the full df since df_analysis is a subset)
    df_full = calculate_atr(df, params.atr_period)
    df_full = detect_local_extrema(
        df_full,
        params.window_left,
        params.window_right,
        window_short=params.window_short,
        trend=current_trend,
        consolidation_zones=consolidation_zones
    )

    # Calculate EMA for full dataframe for plotting
    ema_full = calculate_ema(df["close"].to_numpy(), params.ema_span)

    # Segment trends using local highs/lows (market structure)
    trend_segments = segment_trends(
        df_analysis,
        local_high_indices=local_high_indices,
        local_high_prices=local_high_prices,
        local_low_indices=local_low_indices,
        local_low_prices=local_low_prices,
    )

    return PatternResult(
        df=df_full,
        horizontal_zones=support_zones + resistance_zones,
        support_lines=support_lines,
        resistance_lines=resistance_lines,
        triangles=triangles,
        consolidation_zones=consolidation_zones,
        current_trend=current_trend,
        trend_slope=trend_slope,
        eps=current_eps,
        sr_eps=sr_eps,
        replay_idx=replay_idx,
        ema=ema_full,
        trend_segments=trend_segments,
    )


def plot_patterns(
    result: PatternResult,
    title: str = "Support & Resistance Patterns",
    params: SRParams = None,
    show_ema: bool = True,
    show_trendlines: bool = True,
    show_sr_zones: bool = False,
    show_triangles: bool = False,
    show_consolidation: bool = True,
    show_local_extrema: bool = True,
    show_trend_segments: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot all detected patterns.

    If result.replay_idx is set, candles after that index are shown in light gray:
    - Hollow (outline only) for bullish candles (close >= open)
    - Filled for bearish candles (close < open)
    """
    df = result.df
    replay_idx = result.replay_idx

    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot candlesticks
    times = df["open_time"].to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    width = 0.02
    for i in range(len(df)):
        is_future = replay_idx is not None and i > replay_idx
        is_bullish = closes[i] >= opens[i]

        if is_future:
            # Future candles: light gray color, hollow for bullish, filled for bearish
            wick_color = "#B0B0B0"  # Light gray
            if is_bullish:
                # Hollow candle: just outline
                face_color = "none"
                edge_color = "#B0B0B0"
            else:
                # Filled candle
                face_color = "#B0B0B0"
                edge_color = "#B0B0B0"
        else:
            # Normal candles: green for bullish, red for bearish
            if is_bullish:
                wick_color = "#26a69a"
                face_color = "#26a69a"
                edge_color = "#26a69a"
            else:
                wick_color = "#ef5350"
                face_color = "#ef5350"
                edge_color = "#ef5350"

        ax.plot([times[i], times[i]], [lows[i], highs[i]], color=wick_color, linewidth=0.8)
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        rect = Rectangle(
            (mdates.date2num(times[i]) - width / 2, body_bottom),
            width, body_height,
            facecolor=face_color, edgecolor=edge_color, linewidth=1.2
        )
        ax.add_patch(rect)

    # Plot EMA (semi-transparent blue line)
    if show_ema and result.ema is not None:
        # In replay mode, only plot EMA up to replay_idx
        if replay_idx is not None:
            ema_times = times[:replay_idx + 1]
            ema_values = result.ema[:replay_idx + 1]
        else:
            ema_times = times
            ema_values = result.ema
        ax.plot(ema_times, ema_values, color="#2196F3", linewidth=2, alpha=0.6,
                label=f"EMA", zorder=4)

    # Determine the cutoff index for plotting markers/patterns
    # Only show local highs/lows up to replay_idx
    if replay_idx is not None:
        df_visible = df.filter(pl.col("bar_index") <= replay_idx)
    else:
        df_visible = df

    # Plot local highs/lows (only up to replay time)
    if show_local_extrema:
        local_highs = df_visible.filter(pl.col("is_local_high"))
        local_lows = df_visible.filter(pl.col("is_local_low"))

        if local_highs.height > 0:
            ax.scatter(local_highs["open_time"].to_numpy(), local_highs["high"].to_numpy(),
                       marker="v", color="red", s=30, alpha=0.7, zorder=5, label=f"Local Highs ({local_highs.height})")
        if local_lows.height > 0:
            ax.scatter(local_lows["open_time"].to_numpy(), local_lows["low"].to_numpy(),
                       marker="^", color="lime", s=30, alpha=0.7, zorder=5, label=f"Local Lows ({local_lows.height})")

    # Calculate right edge for S/R zone lines (extend past last candle for labels)
    time_delta = times[-1] - times[-2] if len(times) >= 2 else np.timedelta64(1, 'h')
    x_right_edge = times[-1] + time_delta * 5
    if replay_idx is not None:
        x_right_edge = times[min(replay_idx, len(times) - 1)] + time_delta * 5

    # Plot horizontal zones as lines (light yellow) extending to right edge with price labels
    for zone in result.horizontal_zones if show_sr_zones else []:
        alpha = min(0.9, 0.5 + zone.strength * 0.1)
        linewidth = min(2.5, 1 + zone.touches * 0.3)

        x_start = times[zone.start_idx]

        ax.hlines(y=zone.price_center, xmin=x_start, xmax=x_right_edge,
                  color="#FFEB3B", linewidth=linewidth, alpha=alpha, linestyle="-")

        ax.text(x_right_edge, zone.price_center, f" {zone.price_center:.4f}",
                fontsize=7, color="black", va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFEB3B", edgecolor="none", alpha=0.9),
                clip_on=True)

    # Helper to convert bar index to time
    def idx_to_time(idx):
        if idx < len(times):
            return times[idx]
        else:
            time_delta = times[-1] - times[-2]
            return times[-1] + time_delta * (idx - len(times) + 1)

    # For trendlines in replay mode, limit x2 to not extend past replay_idx too much
    def get_line_end_idx(line):
        if replay_idx is not None:
            return min(line.x2, replay_idx + 30)
        return line.x2

    # Compute visible price range for y-range culling
    current_bar = replay_idx if replay_idx is not None else len(times) - 1
    if replay_idx is not None:
        visible_lows_arr = lows[:replay_idx + 1]
        visible_highs_arr = highs[:replay_idx + 1]
    else:
        visible_lows_arr = lows
        visible_highs_arr = highs
    price_min = np.min(visible_lows_arr)
    price_max = np.max(visible_highs_arr)

    # ================================================================
    # TRENDLINE DISPLAY RULES (when trend_filter_lines=True):
    #
    # "Solid" = slope agrees with line type (rising support, falling resistance)
    # "Dashed" = slope disagrees (falling support, rising resistance)
    #
    # Downtrend: show ALL green (support) lines, only SOLID red (resistance)
    # Uptrend:   show ALL red (resistance) lines, only SOLID green (support)
    # Sideways:  show ALL lines
    # ================================================================
    if show_trendlines:
        trend = result.current_trend
        filter_lines = params.trend_filter_lines if params else False

        def line_in_y_range(line):
            """Check if line's y-value at current bar is within visible price range."""
            y_at_current = line.y1 + line.slope * (current_bar - line.x1)
            return price_min <= y_at_current <= price_max

        # Plot support lines (green, solid=rising, dashed=falling)
        for i, line in enumerate(result.support_lines):
            if not line_in_y_range(line):
                continue
            style = "-" if line.slope >= 0 else "--"
            if filter_lines and trend == 1 and line.slope < 0:
                continue
            alpha = min(0.7, 0.3 + line.strength * 0.1)
            end_idx = get_line_end_idx(line)
            end_y = line.y1 + line.slope * (end_idx - line.x1)
            ax.plot([idx_to_time(line.x1), idx_to_time(end_idx)], [line.y1, end_y],
                    color="green", linewidth=1, linestyle=style, alpha=alpha)

        # Plot resistance lines (red, solid=falling, dashed=rising)
        for i, line in enumerate(result.resistance_lines):
            if not line_in_y_range(line):
                continue
            style = "-" if line.slope <= 0 else "--"
            if filter_lines and trend == -1 and line.slope > 0:
                continue
            alpha = min(0.7, 0.3 + line.strength * 0.1)
            end_idx = get_line_end_idx(line)
            end_y = line.y1 + line.slope * (end_idx - line.x1)
            ax.plot([idx_to_time(line.x1), idx_to_time(end_idx)], [line.y1, end_y],
                    color="red", linewidth=1, linestyle=style, alpha=alpha)

    # Highlight triangles (blue lines + star apex)
    if show_triangles:
        for tri in result.triangles:
            sup, res = tri.support_line, tri.resistance_line

            apex_x_display = tri.apex_x
            if replay_idx is not None:
                apex_x_display = min(apex_x_display, replay_idx + 50)
            apex_y_display = sup.y1 + sup.slope * (apex_x_display - sup.x1)

            ax.plot([idx_to_time(sup.x1), idx_to_time(apex_x_display)],
                    [sup.y1, apex_y_display],
                    color="blue", linewidth=2, alpha=0.8)

            apex_y_res = res.y1 + res.slope * (apex_x_display - res.x1)
            ax.plot([idx_to_time(res.x1), idx_to_time(apex_x_display)],
                    [res.y1, apex_y_res],
                    color="blue", linewidth=2, alpha=0.8)

            ax.scatter([idx_to_time(tri.apex_x)], [tri.apex_price],
                       marker="*", color="yellow", s=200, zorder=10, edgecolor="black")

            bias_color = "green" if tri.breakout_bias == "bullish" else ("red" if tri.breakout_bias == "bearish" else "gray")
            ax.annotate(f"{tri.pattern_type}\n{tri.completion_pct:.0f}% complete\n{tri.breakout_bias}",
                        xy=(idx_to_time(tri.apex_x), tri.apex_price),
                        xytext=(10, 10), textcoords="offset points",
                        fontsize=9, color=bias_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Plot consolidation zones (orange rectangles) - only up to replay_idx
    for cz in result.consolidation_zones if show_consolidation else []:
        # Skip zones that are entirely after replay time
        if replay_idx is not None and cz.start_idx > replay_idx:
            continue

        x_start = times[cz.start_idx]
        end_idx = cz.end_idx
        if replay_idx is not None:
            end_idx = min(end_idx, replay_idx)
        x_end = times[min(end_idx, len(times) - 1)]

        rect = Rectangle(
            (mdates.date2num(x_start), cz.price_low),
            mdates.date2num(x_end) - mdates.date2num(x_start),
            cz.price_high - cz.price_low,
            facecolor="orange", alpha=0.25, edgecolor="orange",
            linewidth=1.5, linestyle="--", zorder=2
        )
        ax.add_patch(rect)

        # Add label for longer consolidation zones
        if cz.duration >= 15:
            mid_idx = (cz.start_idx + min(cz.end_idx, end_idx)) // 2
            mid_time = times[mid_idx]
            ax.annotate(f"Consol\n{cz.duration} bars\n{cz.channel_width_pct*100:.1f}%",
                        xy=(mid_time, cz.price_high),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=8, color="darkorange", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # Add replay time indicator line
    if replay_idx is not None:
        replay_time = times[replay_idx]
        replay_time_str = str(replay_time)[:16]  # Convert numpy datetime64 to string
        ax.axvline(x=replay_time, color="red", linewidth=2, linestyle="--", alpha=0.8, zorder=15)
        ax.text(replay_time, ax.get_ylim()[1], f" REPLAY: {replay_time_str} ",
                fontsize=10, color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", edgecolor="none", alpha=0.9))

    # Trend indicator box
    if result.current_trend == 1:
        trend_color, trend_text = "green", f"UP\n{result.trend_slope:+.2f}%"
    elif result.current_trend == -1:
        trend_color, trend_text = "red", f"DOWN\n{result.trend_slope:+.2f}%"
    else:
        trend_color, trend_text = "gray", f"SIDE\n{result.trend_slope:+.2f}%"

    ax.text(0.98, 0.98, trend_text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color="white",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=trend_color, alpha=0.8))

    # Stats box
    stats_text = (f"S/R Zones: {len(result.horizontal_zones)}\n"
                  f"Support: {len(result.support_lines)}\n"
                  f"Resistance: {len(result.resistance_lines)}\n"
                  f"Triangles: {len(result.triangles)}\n"
                  f"Consolidation: {len(result.consolidation_zones)}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Set y-axis limits based on actual price data (reuse price_min/price_max from above)
    price_range = price_max - price_min
    padding = price_range * 0.05  # 5% padding above and below
    # Reserve extra space at the bottom for the trend segment bar
    bottom_pad = price_range * 0.08 if show_trend_segments else padding
    ax.set_ylim(price_min - bottom_pad, price_max + padding)

    # Plot trend segment colored bar at the bottom of the chart
    if show_trend_segments and result.trend_segments:
        trend_colors = {1: "#26a69a", 0: "#FFC107", -1: "#ef5350"}  # green, amber, red
        bar_y = price_min - bottom_pad  # Bottom of visible area
        bar_height = bottom_pad * 0.35  # Thick bar

        for seg in result.trend_segments:
            # Clamp to replay_idx if in replay mode
            s_start = seg.start_idx
            s_end = seg.end_idx
            if replay_idx is not None:
                if s_start > replay_idx:
                    continue
                s_end = min(s_end, replay_idx)

            x_start = mdates.date2num(times[min(s_start, len(times) - 1)])
            x_end = mdates.date2num(times[min(s_end, len(times) - 1)])
            color = trend_colors.get(seg.trend, "#FFC107")

            rect = Rectangle(
                (x_start, bar_y),
                x_end - x_start, bar_height,
                facecolor=color, alpha=0.5, edgecolor="none", zorder=20,
                clip_on=True,
            )
            ax.add_patch(rect)

    # Format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=45)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.savefig("sr_patterns_plot.png", dpi=150)
        print("Plot saved to sr_patterns_plot.png")
    if not save_path:
        plt.show()

    return fig, ax


def parse_replay_time(time_str: str) -> datetime:
    """
    Parse replay time string. Supports multiple formats:
    - "YYYY-MM-DD HH:MM"
    - "YYYY-MM-DD HH-MM"
    - "YYYY-MM-DD"
    """
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H-%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse time: {time_str}. Use format like 'YYYY-MM-DD HH:MM'")


def process_file(
    data_path: str,
    params: SRParams = None,
    display_timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    replay_time: Optional[str] = None,
    save_path: Optional[str] = None,
    show_ema: bool = True,
    show_trendlines: bool = True,
    show_sr_zones: bool = False,
    show_triangles: bool = False,
    show_consolidation: bool = True,
    show_local_extrema: bool = True,
    show_trend_segments: bool = True,
):
    """Process a single CSV file: load data, detect patterns, plot and save."""
    if params is None:
        params = PARAMS

    token, interval = extract_token_and_interval(data_path)
    print(f"\nProcessing: {data_path} ({token} {interval})")

    df = pl.read_csv(data_path)
    df = df.with_columns([
        pl.col("open_time").str.to_datetime("%Y-%m-%d %H:%M:%S")
    ])

    if start_date:
        dt = datetime.strptime(start_date, "%Y-%m-%d")
        df = df.filter(pl.col("open_time") >= dt)

    native_minutes, native_label = detect_data_interval(df)
    display_interval = native_label

    if display_timeframe is not None:
        df, was_resampled, resample_msg = resample_ohlcv(df, display_timeframe, native_minutes)
        if resample_msg:
            print(f"  {resample_msg}")
        if was_resampled:
            display_interval = display_timeframe.upper()

    print(f"  Data shape: {df.shape}")
    print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")

    replay_idx = None
    if replay_time is not None:
        replay_dt = parse_replay_time(replay_time)
        df_temp = df.with_row_index("_idx")
        matching = df_temp.filter(pl.col("open_time") <= replay_dt)
        if matching.height == 0:
            replay_idx = 0
        else:
            replay_idx = int(matching["_idx"][-1])

    result = detect_patterns(df, params, replay_idx=replay_idx)

    print(f"  Support: {len(result.support_lines)}, Resistance: {len(result.resistance_lines)}, "
          f"Consolidation: {len(result.consolidation_zones)}, Triangles: {len(result.triangles)}")

    if replay_idx is not None:
        replay_time_str = str(df['open_time'][replay_idx])[:16]
        title = f"{token} {display_interval} - S/R Patterns (Replay: {replay_time_str})"
    else:
        title = f"{token} {display_interval} - Support/Resistance & Patterns"

    plot_patterns(
        result, title, params=params,
        show_ema=show_ema,
        show_trendlines=show_trendlines,
        show_sr_zones=show_sr_zones,
        show_triangles=show_triangles,
        show_consolidation=show_consolidation,
        show_local_extrema=show_local_extrema,
        show_trend_segments=show_trend_segments,
        save_path=save_path,
    )


def _process_one(args: tuple) -> tuple[str, Optional[str]]:
    """
    Worker function for parallel processing.
    Returns (filepath, error_message) — error_message is None on success.
    """
    import matplotlib
    matplotlib.use("Agg")

    filepath, save_path, kwargs = args
    try:
        process_file(data_path=filepath, save_path=save_path, **kwargs)
        plt.close("all")
        return filepath, None
    except Exception as e:
        plt.close("all")
        return filepath, str(e)


def main(
    data_dir: str = "data",
    output_dir: str = "plots",
    timeframe: Optional[str] = "1h",
    start_date: Optional[str] = None,
    replay_time: Optional[str] = None,
    max_workers: Optional[int] = None,
):
    """
    Process all CSV files in data_dir in parallel and save plots to output_dir.

    Args:
        data_dir: Directory containing CSV data files.
        output_dir: Directory to save plots (created if it doesn't exist).
        timeframe: Display timeframe to resample to (e.g. "1h", "4h", "1d").
                   Set to None to use each file's native timeframe.
        start_date: Optional start date filter ("YYYY-MM-DD").
        replay_time: Optional replay time ("YYYY-MM-DD HH:MM").
        max_workers: Max parallel processes (defaults to CPU count).
    """
    import glob
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {data_dir}/")
        return

    print(f"Found {len(csv_files)} CSV files in {data_dir}/")
    print(f"Timeframe: {timeframe or 'native'}")
    print(f"Output dir: {output_dir}/")

    tf_label = timeframe if timeframe else "native"
    shared_kwargs = dict(
        display_timeframe=timeframe,
        start_date=start_date,
        replay_time=replay_time,
        show_ema=SHOW_EMA,
        show_trendlines=SHOW_TRENDLINES,
        show_sr_zones=SHOW_SR_ZONES,
        show_triangles=SHOW_TRIANGLES,
        show_consolidation=SHOW_CONSOLIDATION,
        show_local_extrema=SHOW_LOCAL_EXTREMA,
        show_trend_segments=SHOW_TREND_SEGMENTS,
    )

    tasks = []
    for filepath in csv_files:
        token, _ = extract_token_and_interval(filepath)
        filename = f"{token.lower()}_{tf_label}.png"
        save_path = os.path.join(output_dir, filename)
        tasks.append((filepath, save_path, shared_kwargs))

    succeeded = []
    failed = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, task): task[0] for task in tasks}
        for future in as_completed(futures):
            filepath, error = future.result()
            basename = os.path.basename(filepath)
            if error is None:
                succeeded.append(basename)
            else:
                failed.append((basename, error))
                print(f"  FAILED: {basename} — {error}")

    print(f"\n{'='*50}")
    print(f"Results: {len(succeeded)}/{len(csv_files)} succeeded")
    if succeeded:
        print(f"  OK: {', '.join(sorted(succeeded))}")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for name, err in sorted(failed):
            print(f"    - {name}: {err}")


if __name__ == "__main__":
    main()
