"""
Plot helper functions for crypto technical analysis.

Provides reusable charting with:
- Candlestick chart with Bollinger Bands overlay
- Quote asset volume (USDT) subplot
- Taker volume ratio subplot
"""

import os

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

PLOTS_DIR = "plots"


def calculate_bollinger_bands(
    closes: np.ndarray, period: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Returns (middle, upper, lower) as numpy arrays.
    Values before `period` are NaN.
    """
    n = len(closes)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        sma = window.mean()
        std = window.std(ddof=0)
        middle[i] = sma
        upper[i] = sma + num_std * std
        lower[i] = sma - num_std * std

    return middle, upper, lower


def plot_candlestick_volume(
    df: pl.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    title: str = "Price Chart",
    figsize: tuple = (18, 14),
    save_path: str | None = None,
    source_file: str | None = None,
):
    """
    Plot candlestick chart with Bollinger Bands, quote volume, and taker volume ratio.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: open_time, open, high, low, close.
        If quote_asset_volume and taker_buy_quote_asset_volume are present,
        volume and taker ratio subplots are shown. Otherwise only the
        candlestick chart with Bollinger Bands is plotted.
    bb_period : int
        Bollinger Bands lookback period (default 20).
    bb_std : float
        Number of standard deviations for bands (default 2.0).
    title : str
        Chart title.
    figsize : tuple
        Figure size.
    save_path : str | None
        If provided, save the figure to this path. Otherwise auto-saves to
        the ``plots/`` directory using the source filename with a .png extension.
    source_file : str | None
        Original data filename (e.g. "riverusdt_1h.csv"). Used to derive
        the default save path when ``save_path`` is not given.

    Returns
    -------
    fig, axes : tuple
        Matplotlib figure and array of axes [candlestick, volume, ratio].
    """
    times = df["open_time"].to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    has_volume = (
        "quote_asset_volume" in df.columns
        and "taker_buy_quote_asset_volume" in df.columns
    )

    if has_volume:
        quote_vol = df["quote_asset_volume"].cast(pl.Float64).to_numpy()
        taker_buy_vol = df["taker_buy_quote_asset_volume"].cast(pl.Float64).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            taker_ratio = np.where(quote_vol > 0, taker_buy_vol / quote_vol, np.nan)

    # Bollinger Bands
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(closes, bb_period, bb_std)

    # --- Layout ---
    if has_volume:
        fig, (ax_price, ax_vol, ax_ratio) = plt.subplots(
            3, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
    else:
        fig, ax_price = plt.subplots(figsize=figsize)
        ax_vol = None
        ax_ratio = None

    # ---- Candlestick chart ----
    width = 0.02
    bull_color = "#26a69a"
    bear_color = "#ef5350"

    for i in range(len(df)):
        is_bullish = closes[i] >= opens[i]
        color = bull_color if is_bullish else bear_color

        # Wick
        ax_price.plot(
            [times[i], times[i]], [lows[i], highs[i]],
            color=color, linewidth=0.8,
        )
        # Body
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        rect = Rectangle(
            (mdates.date2num(times[i]) - width / 2, body_bottom),
            width, body_height,
            facecolor=color, edgecolor=color, linewidth=1.2,
        )
        ax_price.add_patch(rect)

    # Bollinger Bands overlay
    valid = ~np.isnan(bb_mid)
    ax_price.plot(times[valid], bb_mid[valid], color="#2196F3", linewidth=1, alpha=0.8, label=f"BB Mid ({bb_period})")
    ax_price.plot(times[valid], bb_upper[valid], color="#90CAF9", linewidth=0.8, alpha=0.7, label=f"BB Upper ({bb_std}σ)")
    ax_price.plot(times[valid], bb_lower[valid], color="#90CAF9", linewidth=0.8, alpha=0.7, label=f"BB Lower ({bb_std}σ)")
    ax_price.fill_between(times[valid], bb_upper[valid], bb_lower[valid], color="#2196F3", alpha=0.08)

    ax_price.set_title(title, fontsize=14)
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)

    if has_volume:
        # ---- Volume subplot ----
        vol_colors = [bull_color if closes[i] >= opens[i] else bear_color for i in range(len(df))]
        ax_vol.bar(times, quote_vol, width=width, color=vol_colors, alpha=0.7)
        ax_vol.set_ylabel("Volume (USDT)")
        ax_vol.grid(True, alpha=0.3)

        # ---- Taker ratio subplot ----
        ax_ratio.plot(times, taker_ratio, color="#AB47BC", linewidth=1, alpha=0.9)
        ax_ratio.axhline(y=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_ratio.set_ylabel("Taker Volume Ratio")
        ax_ratio.set_xlabel("Time")
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.set_ylim(0, 1)

    # Format x-axis
    bottom_ax = ax_ratio if has_volume else ax_price
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    bottom_ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Determine save path
    if save_path is None and source_file is not None:
        basename = os.path.splitext(os.path.basename(source_file))[0] + ".png"
        save_path = os.path.join(PLOTS_DIR, basename)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    # plt.show()
    return fig, (ax_price, ax_vol, ax_ratio)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "riverusdt_1h.csv"
    df = pl.read_csv(path)
    df = df.with_columns(pl.col("open_time").str.to_datetime("%Y-%m-%d %H:%M:%S"))

    from sr_patterns import extract_token_and_interval
    token, interval = extract_token_and_interval(path)

    plot_candlestick_volume(
        df,
        title=f"{token} {interval} - Candlestick / Volume / Taker Ratio",
        source_file=path,
    )
