"""
Support and Resistance Lines Detection

Converted from TradingView Pine Script to Python/Polars.
Detects local highs/lows and draws trend-following support/resistance lines.
"""

import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class SRParams:
    """Parameters for Support and Resistance detection."""
    window_left: int = 3  # Left window for local high/low detection
    window_right: int = 3  # Right window for local high/low detection
    atr_period: int = 9  # ATR Period
    atr_multiplier: float = 0.2  # ATR Multiplier for eps
    start_time: Optional[datetime] = None  # Start date filter
    lookback: int = 150  # Lookback window for line detection
    ma_window: int = 50  # MA window for trend detection
    trend_threshold: float = 0.5  # Trend threshold percentage
    slope_lookback: int = 50  # Bars to look back for slope calculation


@dataclass
class TrendLine:
    """Represents a support or resistance trend line."""
    x1: int  # Start bar index
    y1: float  # Start price
    x2: int  # End bar index
    y2: float  # End price
    slope: float
    line_type: str  # 'support' or 'resistance'


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


def calculate_ema(df: pl.DataFrame, column: str, period: int) -> pl.DataFrame:
    """Calculate Exponential Moving Average."""
    return df.with_columns([
        pl.col(column).ewm_mean(span=period, adjust=False).alias(f"ema_{period}")
    ])


def detect_local_extrema(df: pl.DataFrame, window_left: int, window_right: int, eps: pl.Series) -> pl.DataFrame:
    """
    Detect local highs and lows with asymmetrical window.
    A local high is confirmed when it's the highest in [i - window_left, i + window_right].
    """
    # Get eps as a column if it's a series
    if isinstance(eps, pl.Series):
        df = df.with_columns([eps.alias("eps")])
    else:
        df = df.with_columns([pl.lit(eps).alias("eps")])

    # For asymmetric windows, we need to compute max/min manually
    # Rolling with shift: compute max over window_left+window_right+1 bars, then shift
    full_window = window_left + window_right + 1

    # Rolling max/min looking back full_window bars, then shift forward by window_right
    # This gives us max/min over [i - window_left, i + window_right]
    df = df.with_columns([
        pl.col("high").rolling_max(window_size=full_window).shift(-window_right).alias("rolling_high"),
        pl.col("low").rolling_min(window_size=full_window).shift(-window_right).alias("rolling_low"),
    ])

    # Detect local highs and lows with eps tolerance
    df = df.with_columns([
        (pl.col("high") >= pl.col("rolling_high") - 0.1 * pl.col("eps")).alias("is_local_high"),
        (pl.col("low") <= pl.col("rolling_low") + 0.1 * pl.col("eps")).alias("is_local_low"),
    ])

    return df


def calculate_trend(df: pl.DataFrame, ma_window: int, slope_lookback: int, threshold: float) -> pl.DataFrame:
    """
    Calculate trend based on EMA slope.
    Returns df with trend column: 1 for uptrend, -1 for downtrend, 0 for sideways.
    """
    df = calculate_ema(df, "close", ma_window)

    df = df.with_columns([
        pl.col(f"ema_{ma_window}").shift(slope_lookback).alias("ema_shifted")
    ])

    df = df.with_columns([
        ((pl.col(f"ema_{ma_window}") - pl.col("ema_shifted")) / pl.col("ema_shifted") * 100)
        .fill_null(0)
        .alias("perc_slope")
    ])

    df = df.with_columns([
        pl.when(pl.col("perc_slope") > threshold)
        .then(pl.lit(1))
        .when(pl.col("perc_slope") < -threshold)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("trend")
    ])

    return df


def is_valid_support(
    lows: np.ndarray,
    x1: int,
    y1: float,
    slope: float,
    current_idx: int,
    eps: float
) -> bool:
    """Check if support line is valid (no candle low crosses below line from x1 onward)."""
    for offset in range(current_idx - x1 + 1):
        bar_idx = x1 + offset
        if bar_idx >= len(lows):
            break
        line_y = y1 + slope * offset
        if lows[bar_idx] < line_y - eps:
            return False
    return True


def is_valid_resistance(
    highs: np.ndarray,
    x1: int,
    y1: float,
    slope: float,
    current_idx: int,
    eps: float
) -> bool:
    """Check if resistance line is valid (no candle high crosses above line from x1 onward)."""
    for offset in range(current_idx - x1 + 1):
        bar_idx = x1 + offset
        if bar_idx >= len(highs):
            break
        line_y = y1 + slope * offset
        if highs[bar_idx] > line_y + eps:
            return False
    return True


def find_support_resistance_lines(
    df: pl.DataFrame,
    params: SRParams = SRParams(),
    time_col: str = "open_time"
) -> tuple[pl.DataFrame, list[TrendLine]]:
    """
    Find support and resistance lines in OHLCV data.

    Args:
        df: Polars DataFrame with OHLCV data
        params: SRParams configuration
        time_col: Name of the time column

    Returns:
        Tuple of (annotated DataFrame, list of TrendLine objects)
    """
    # Ensure we have a bar_index column
    df = df.with_row_index("bar_index")

    # Filter by start time if specified
    if params.start_time is not None:
        df = df.with_columns([
            (pl.col(time_col) >= params.start_time).alias("is_after_start")
        ])
    else:
        df = df.with_columns([pl.lit(True).alias("is_after_start")])

    # Calculate ATR and eps
    df = calculate_atr(df, params.atr_period)
    df = df.with_columns([
        (pl.col("atr") * params.atr_multiplier).alias("eps")
    ])

    # Calculate trend
    df = calculate_trend(df, params.ma_window, params.slope_lookback, params.trend_threshold)

    # Detect local extrema
    eps_series = df["eps"]
    df = detect_local_extrema(df, params.window_left, params.window_right, eps_series)

    # Apply start time filter to local extrema
    df = df.with_columns([
        (pl.col("is_local_high") & pl.col("is_after_start")).alias("is_local_high"),
        (pl.col("is_local_low") & pl.col("is_after_start")).alias("is_local_low"),
    ])

    # Extract numpy arrays for efficient line validation
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    eps_arr = df["eps"].fill_null(0).to_numpy()

    # Get local high/low points
    local_highs = df.filter(pl.col("is_local_high")).select(["bar_index", "high"]).to_numpy()
    local_lows = df.filter(pl.col("is_local_low")).select(["bar_index", "low"]).to_numpy()

    # Get current bar info (last bar)
    current_idx = len(df) - 1
    current_trend = df["trend"][-1]
    current_eps = eps_arr[-1] if not np.isnan(eps_arr[-1]) else eps_arr[~np.isnan(eps_arr)][-1]

    # Filter points within lookback window
    min_bar = current_idx - params.lookback

    filtered_highs = local_highs[local_highs[:, 0] >= min_bar] if len(local_highs) > 0 else np.array([])
    filtered_lows = local_lows[local_lows[:, 0] >= min_bar] if len(local_lows) > 0 else np.array([])

    trend_lines = []

    # Find support lines in uptrend
    if current_trend == 1 and len(filtered_lows) >= 2:
        for i in range(len(filtered_lows) - 1):
            for j in range(i + 1, len(filtered_lows)):
                x1, y1 = int(filtered_lows[i, 0]), filtered_lows[i, 1]
                x2, y2 = int(filtered_lows[j, 0]), filtered_lows[j, 1]

                if x1 < x2:
                    slope = (y2 - y1) / (x2 - x1)

                    # Support lines should have positive slope in uptrend
                    if slope > 0 and is_valid_support(lows, x1, y1, slope, current_idx, current_eps):
                        # Extend line 50 bars into the future
                        end_x = current_idx + 50
                        end_y = y1 + slope * (end_x - x1)

                        trend_lines.append(TrendLine(
                            x1=x1, y1=y1,
                            x2=end_x, y2=end_y,
                            slope=slope,
                            line_type='support'
                        ))

    # Find resistance lines in downtrend
    if current_trend == -1 and len(filtered_highs) >= 2:
        for i in range(len(filtered_highs) - 1):
            for j in range(i + 1, len(filtered_highs)):
                x1, y1 = int(filtered_highs[i, 0]), filtered_highs[i, 1]
                x2, y2 = int(filtered_highs[j, 0]), filtered_highs[j, 1]

                if x1 < x2:
                    slope = (y2 - y1) / (x2 - x1)

                    # Resistance lines should have negative slope in downtrend
                    if slope < 0 and is_valid_resistance(highs, x1, y1, slope, current_idx, current_eps):
                        # Extend line 50 bars into the future
                        end_x = current_idx + 50
                        end_y = y1 + slope * (end_x - x1)

                        trend_lines.append(TrendLine(
                            x1=x1, y1=y1,
                            x2=end_x, y2=end_y,
                            slope=slope,
                            line_type='resistance'
                        ))

    return df, trend_lines


def get_trend_label(df: pl.DataFrame) -> str:
    """Get human-readable trend label for the current bar."""
    trend = df["trend"][-1]
    slope = df["perc_slope"][-1]

    if trend == 1:
        return f"Uptrend: {slope:.2f}%"
    elif trend == -1:
        return f"Downtrend: {slope:.2f}%"
    else:
        return f"Sideways: {slope:.2f}%"


def lines_to_dataframe(lines: list[TrendLine]) -> pl.DataFrame:
    """Convert list of TrendLines to a Polars DataFrame for easier analysis."""
    if not lines:
        return pl.DataFrame({
            "x1": [], "y1": [], "x2": [], "y2": [],
            "slope": [], "line_type": []
        })

    return pl.DataFrame({
        "x1": [l.x1 for l in lines],
        "y1": [l.y1 for l in lines],
        "x2": [l.x2 for l in lines],
        "y2": [l.y2 for l in lines],
        "slope": [l.slope for l in lines],
        "line_type": [l.line_type for l in lines],
    })


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n = 300

    # Generate sample OHLCV data with a trend
    base_price = 100
    trend = np.cumsum(np.random.randn(n) * 0.5)

    opens = base_price + trend
    highs = opens + np.abs(np.random.randn(n) * 0.5)
    lows = opens - np.abs(np.random.randn(n) * 0.5)
    closes = opens + np.random.randn(n) * 0.3
    volumes = np.random.randint(1000, 10000, n)
    times = pl.datetime_range(
        datetime(2025, 1, 1),
        datetime(2025, 1, 1) + pl.duration(hours=n-1),
        interval="1h",
        eager=True
    )

    df = pl.DataFrame({
        "open_time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # Find support and resistance lines
    params = SRParams(
        window_left=3,
        window_right=3,
        atr_period=9,
        atr_multiplier=0.2,
        lookback=150,
        ma_window=50,
        trend_threshold=0.5
    )

    result_df, lines = find_support_resistance_lines(df, params)

    print(f"Trend: {get_trend_label(result_df)}")
    print(f"Local Highs: {result_df.filter(pl.col('is_local_high')).height}")
    print(f"Local Lows: {result_df.filter(pl.col('is_local_low')).height}")
    print(f"Support Lines: {len([l for l in lines if l.line_type == 'support'])}")
    print(f"Resistance Lines: {len([l for l in lines if l.line_type == 'resistance'])}")

    # Convert lines to DataFrame
    lines_df = lines_to_dataframe(lines)
    print("\nTrend Lines:")
    print(lines_df)
