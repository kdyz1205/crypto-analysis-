"""
Advanced Support/Resistance and Pattern Detection

Features:
- Horizontal support/resistance zones (clustered price levels)
- Tilted support/resistance lines (both directions)
- Triangle pattern detection (ascending, descending, symmetrical)
- Consolidation zone detection (flat rectangles with low volatility)
- Trend-aware asymmetric window extrema detection
- Fast trend detection (price vs EMA + momentum)
"""

import polars as pl
import numpy as np
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SRParams:
    """Parameters for pattern detection."""
    window_left: int = 1
    window_right: int = 5
    window_short: int = 2  # Shortened window for trend-aware extrema detection
    # Trend detection parameters
    ema_span: int = 50  # EMA period for trend detection
    trend_threshold: float = 0.5  # % slope threshold for trend classification
    use_fast_trend: bool = True  # Use faster trend detection (price vs EMA + momentum)
    atr_period: int = 14
    atr_multiplier: float = 0.5  # For trendline eps (ATR-based)
    sr_eps_pct: float = 0.01  # S/R zone eps as percentage of price (0.01 = 1%)
    lookback: int = 100
    min_touches: int = 2  # Minimum touches for horizontal zones
    zone_merge_factor: float = 1.5  # Merge zones within this * eps
    slope_tolerance: float = 0.0001  # Slopes below this are "flat"
    max_lines: int = 10  # Limit lines to avoid clutter
    line_break_factor: float = 0.3  # Break tolerance as fraction of eps (ATR-based). Higher = more lenient
    # Consolidation zone parameters
    consol_window: int = 10  # Rolling window for consolidation detection
    consol_range_pct: float = 0.15  # Max channel width as percentage (0.05 = 5%)
    consol_slope_threshold: float = 0.008  # Max normalized slope (0.002 = 0.2% per bar)
    consol_min_duration: int = 10  # Minimum bars to form a consolidation zone


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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_token_and_interval(filepath: str) -> tuple[str, str]:
    """
    Extract token and interval from filename.

    Examples:
        "riverusdt_1h.csv" -> ("RIVER", "1H")
        "data/btc_4h.csv" -> ("BTC", "4H")
        "ENSO_15m.csv" -> ("ENSO", "15M")
    """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]

    match = re.match(r"^(.+?)(?:usdt|usd|usdc|busd)?_(\d+[mhd])$", name, re.IGNORECASE)
    if match:
        token = match.group(1).upper()
        interval = match.group(2).upper()
        return token, interval

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


# =============================================================================
# CORE CALCULATIONS
# =============================================================================

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

    ema = calculate_ema(closes, ema_span)

    if use_fast:
        current_price = closes[-1]
        ema_now = ema[-1]

        price_vs_ema_pct = (current_price - ema_now) / ema_now * 100

        lookback = min(5, n - 1)
        momentum_pct = (closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback] * 100

        trend_score = price_vs_ema_pct * 0.6 + momentum_pct * 0.4

        if trend_score > threshold:
            trend_direction = 1
        elif trend_score < -threshold:
            trend_direction = -1
        else:
            trend_direction = 0

        return trend_direction, trend_score, ema

    else:
        ema_now = ema[-1]
        ema_prev = ema[-ema_span] if n > ema_span else ema[0]

        trend_slope = ((ema_now - ema_prev) / ema_prev * 100) if ema_prev else 0
        trend_direction = 1 if trend_slope > threshold else (-1 if trend_slope < -threshold else 0)

        return trend_direction, trend_slope, ema


# =============================================================================
# EXTREMA DETECTION
# =============================================================================

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
      - Local max (resistance): left=normal, right=short

    In a downtrend (mirror):
      - Local max: left=short, right=normal
      - Local min: left=normal, right=short

    In sideways OR inside consolidation zones: symmetric windows for both.
    """
    n = len(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    if window_short is None:
        window_short = window_left

    in_consolidation = np.zeros(n, dtype=bool)
    if consolidation_zones:
        for cz in consolidation_zones:
            in_consolidation[cz.start_idx:cz.end_idx + 1] = True

    is_local_high = np.zeros(n, dtype=bool)
    is_local_low = np.zeros(n, dtype=bool)

    for i in range(n):
        local_trend = 0 if in_consolidation[i] else trend

        if local_trend == 1:
            high_left, high_right = window_left, window_short
            low_left, low_right = window_short, window_right
        elif local_trend == -1:
            high_left, high_right = window_short, window_right
            low_left, low_right = window_left, window_short
        else:
            high_left = low_left = window_left
            high_right = low_right = window_right

        if i >= high_left:
            left_bound = max(0, i - high_left)
            right_bound = min(n - 1, i + high_right)
            window_highs = highs[left_bound:right_bound + 1]
            if highs[i] >= np.max(window_highs) - 1e-10:
                is_local_high[i] = True

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


# =============================================================================
# HORIZONTAL ZONES
# =============================================================================

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

    sorted_idx = np.argsort(local_prices)
    sorted_prices = local_prices[sorted_idx]
    sorted_bar_indices = local_indices[sorted_idx]

    zones = []
    merge_dist = eps * merge_factor

    i = 0
    while i < len(sorted_prices):
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


# =============================================================================
# TRENDLINES
# =============================================================================

def is_valid_line(
    prices: np.ndarray,
    x1: int, y1: float,
    slope: float,
    current_idx: int,
    eps: float,
    line_type: str,
    break_factor: float = 0.3
) -> tuple[bool, int]:
    """
    Check if line is valid and count touches.
    Returns (is_valid, touch_count).
    """
    touches = 0
    break_tolerance = eps * break_factor

    for offset in range(current_idx - x1 + 1):
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
        else:
            if price > line_y + break_tolerance:
                return False, 0
            if price <= line_y + break_tolerance and price >= line_y - eps:
                touches += 1

    return True, touches


def find_trendlines(
    local_prices: np.ndarray,
    local_indices: np.ndarray,
    all_prices: np.ndarray,
    eps: float,
    current_idx: int,
    current_price: float,
    line_type: str,
    max_lines: int,
    slope_tolerance: float,
    required_slope_sign: Optional[int] = None,
    break_factor: float = 0.3
) -> list[TrendLine]:
    """
    Find valid trend lines with slope constraints.

    For uptrend context: support lines should have slope > 0 (required_slope_sign=1)
    For downtrend context: resistance lines should have slope < 0 (required_slope_sign=-1)
    """
    if len(local_prices) < 2:
        return []

    lines = []

    for i in range(len(local_prices)):
        for j in range(i + 1, len(local_prices)):
            x1, y1 = int(local_indices[i]), local_prices[i]
            x2, y2 = int(local_indices[j]), local_prices[j]

            if x1 >= x2:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if required_slope_sign is not None:
                if required_slope_sign == 1 and slope <= 0:
                    continue
                if required_slope_sign == -1 and slope >= 0:
                    continue

            valid, touches = is_valid_line(
                all_prices, x1, y1, slope, current_idx, eps, line_type,
                break_factor=break_factor
            )

            if valid and touches >= 2:
                end_x = current_idx + 30
                end_y = y1 + slope * (end_x - x1)
                line_at_current = y1 + slope * (current_idx - x1)

                span = x2 - x1
                span_score = span / current_idx
                recency = x2 / current_idx

                distance_pct = abs(line_at_current - current_price) / current_price
                proximity = 1.0 / (1.0 + distance_pct * 10)

                strength = touches * (0.2 * span_score + 0.2 * recency + 0.6 * proximity)

                lines.append(TrendLine(
                    x1=x1, y1=y1,
                    x2=end_x, y2=end_y,
                    slope=slope,
                    line_type=line_type,
                    touches=touches,
                    strength=strength
                ))

    lines.sort(key=lambda x: x.strength, reverse=True)
    return lines[:max_lines]


# =============================================================================
# TRIANGLES
# =============================================================================

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
            if sup.x1 > res.x2 or res.x1 > sup.x2:
                continue

            slope_diff = res.slope - sup.slope

            if abs(slope_diff) < 1e-10:
                continue

            x_intersect = (res.y1 - sup.y1 + sup.slope * sup.x1 - res.slope * res.x1) / slope_diff
            y_intersect = sup.y1 + sup.slope * (x_intersect - sup.x1)

            if x_intersect <= current_idx or x_intersect > current_idx + 100:
                continue

            sup_at_current = sup.y1 + sup.slope * (current_idx - sup.x1)
            res_at_current = res.y1 + res.slope * (current_idx - res.x1)

            if sup_at_current >= res_at_current:
                continue

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
                continue

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

    triangles.sort(key=lambda x: x.completion_pct, reverse=True)
    return triangles[:5]


# =============================================================================
# CONSOLIDATION ZONES
# =============================================================================

def detect_consolidation_zones(
    df: pl.DataFrame,
    window: int,
    range_pct_threshold: float,
    slope_threshold: float,
    min_duration: int
) -> list[ConsolidationZone]:
    """
    Detect consolidation zones based on channel tightness and trend flatness.

    After detecting a consolidation zone, extends it backwards to find where
    price first entered the zone's price range.
    """
    n = len(df)
    if n < window:
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    rolling_max = np.empty(n)
    rolling_min = np.empty(n)
    rolling_max[:] = np.nan
    rolling_min[:] = np.nan

    for i in range(window - 1, n):
        rolling_max[i] = np.max(highs[i - window + 1:i + 1])
        rolling_min[i] = np.min(lows[i - window + 1:i + 1])

    channel_width_pct = (rolling_max - rolling_min) / rolling_min

    slopes = np.empty(n)
    slopes[:] = np.nan
    x = np.arange(window)

    for i in range(window - 1, n):
        y = closes[i - window + 1:i + 1]
        y_norm = y / y[0] if y[0] != 0 else y
        slope, _ = np.polyfit(x, y_norm, 1)
        slopes[i] = slope

    is_consolidation = (
        (channel_width_pct <= range_pct_threshold) &
        (np.abs(slopes) <= slope_threshold) &
        ~np.isnan(channel_width_pct)
    )

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

                extended_start = start_idx
                for j in range(start_idx - 1, -1, -1):
                    bar_in_range = (lows[j] <= zone_high and highs[j] >= zone_low)
                    if bar_in_range:
                        extended_start = j
                    else:
                        break

                zone_high = np.max(highs[extended_start:end_idx + 1])
                zone_low = np.min(lows[extended_start:end_idx + 1])
                zone_center = (zone_high + zone_low) / 2
                width_pct = (zone_high - zone_low) / zone_low if zone_low > 0 else 0
                duration = end_idx - extended_start + 1

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


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

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

    df = df.with_row_index("bar_index")

    if replay_idx is not None:
        df_analysis = df.filter(pl.col("bar_index") <= replay_idx)
    else:
        df_analysis = df

    # Calculate ATR-based eps (for trendlines)
    df_analysis = calculate_atr(df_analysis, params.atr_period)
    current_eps = df_analysis["atr"][-1] * params.atr_multiplier
    if np.isnan(current_eps):
        current_eps = df_analysis["atr"].drop_nulls()[-1] * params.atr_multiplier

    # Calculate percentage-based eps for S/R zones
    current_price = df_analysis["close"][-1]
    sr_eps = current_price * params.sr_eps_pct

    # Calculate trend
    current_trend, trend_slope, ema = calculate_trend(
        df_analysis,
        ema_span=params.ema_span,
        threshold=params.trend_threshold,
        use_fast=params.use_fast_trend
    )

    current_idx = len(df_analysis) - 1
    min_bar = max(0, current_idx - params.lookback)

    highs = df_analysis["high"].to_numpy()
    lows = df_analysis["low"].to_numpy()

    # Detect consolidation zones FIRST (needed for extrema detection)
    consolidation_zones = detect_consolidation_zones(
        df_analysis,
        window=params.consol_window,
        range_pct_threshold=params.consol_range_pct,
        slope_threshold=params.consol_slope_threshold,
        min_duration=params.consol_min_duration
    )

    # Detect local extrema with trend-aware asymmetric windows
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

    # Filter extrema in consolidation zones
    def filter_extrema_for_consolidation(prices, indices, is_high=True):
        filtered_prices = []
        filtered_indices = []
        used_in_consol = set()

        consol_extrema = {}
        for i, (price, idx) in enumerate(zip(prices, indices)):
            for z_idx, cz in enumerate(consolidation_zones):
                if cz.start_idx <= idx <= cz.end_idx:
                    used_in_consol.add(i)
                    if z_idx not in consol_extrema:
                        consol_extrema[z_idx] = []
                    consol_extrema[z_idx].append((price, idx))
                    break

        for i, (price, idx) in enumerate(zip(prices, indices)):
            if i not in used_in_consol:
                filtered_prices.append(price)
                filtered_indices.append(idx)

        for z_idx, extrema_list in consol_extrema.items():
            if not extrema_list:
                continue

            sorted_extrema = sorted(extrema_list, key=lambda x: x[0])
            clusters = []
            current_cluster = [sorted_extrema[0]]

            for price, idx in sorted_extrema[1:]:
                if price - current_cluster[0][0] <= sr_eps:
                    current_cluster.append((price, idx))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(price, idx)]
            clusters.append(current_cluster)

            for cluster in clusters:
                if is_high:
                    best = max(cluster, key=lambda x: x[0])
                else:
                    best = min(cluster, key=lambda x: x[0])
                filtered_prices.append(best[0])
                filtered_indices.append(best[1])

        return np.array(filtered_prices), np.array(filtered_indices)

    sr_high_prices, sr_high_indices = filter_extrema_for_consolidation(
        local_high_prices, local_high_indices, is_high=True
    )
    sr_low_prices, sr_low_indices = filter_extrema_for_consolidation(
        local_low_prices, local_low_indices, is_high=False
    )

    # Find horizontal zones
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

    support_lines = find_trendlines(
        local_low_prices, local_low_indices, lows, current_eps,
        current_idx, current_price, "support", support_max, params.slope_tolerance,
        required_slope_sign=support_slope_sign,
        break_factor=params.line_break_factor
    )
    resistance_lines = find_trendlines(
        local_high_prices, local_high_indices, highs, current_eps,
        current_idx, current_price, "resistance", resistance_max, params.slope_tolerance,
        required_slope_sign=resistance_slope_sign,
        break_factor=params.line_break_factor
    )

    # Find triangles
    triangles = find_triangles(
        support_lines, resistance_lines, current_idx,
        df_analysis["close"][-1], current_eps, params.slope_tolerance
    )

    # Add local extrema columns to full dataframe for plotting
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
        ema=ema_full
    )
