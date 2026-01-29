import polars as pl
import httpx
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYMBOLS_FILE = PROJECT_ROOT / "binance_futures_usdt.txt"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"

CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]

# Which base interval to download for each requested interval
RESAMPLE_MAP = {
    "5m": ("5m", None),
    "15m": ("5m", "15m"),
    "1h": ("1h", None),
    "4h": ("1h", "4h"),
    "1d": ("1h", "1d"),
}

# How many days of base data to download per interval
DOWNLOAD_DAYS = {
    "5m": 7,
    "1h": 30,
}


def load_symbols() -> list[str]:
    if not SYMBOLS_FILE.exists():
        return []
    lines = SYMBOLS_FILE.read_text().strip().splitlines()
    return sorted([l.strip() for l in lines if l.strip()])


def _find_csv(symbol: str, interval: str) -> Path | None:
    """Find CSV file in data/ directory."""
    filename = f"{symbol.lower()}_{interval}.csv"
    path = DATA_DIR / filename
    if path.exists():
        return path
    # Also check project root for backward compat
    root_path = PROJECT_ROOT / filename
    if root_path.exists():
        return root_path
    return None


def _load_csv(path: Path) -> pl.DataFrame:
    """Load a CSV and parse open_time as datetime."""
    df = pl.read_csv(path)
    # Parse open_time string to datetime
    if df["open_time"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("open_time").str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        if col in df.columns and df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
    return df.sort("open_time")


def resample_ohlcv(df: pl.DataFrame, target_interval: str) -> pl.DataFrame:
    """Resample OHLCV data to a coarser timeframe."""
    return (
        df.sort("open_time")
        .group_by_dynamic("open_time", every=target_interval)
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
    )


async def download_ohlcv(symbol: str, interval: str, days: int = 30) -> pl.DataFrame:
    """Download OHLCV data from Binance Futures API with pagination."""
    DATA_DIR.mkdir(exist_ok=True)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    all_data = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        while start_ms < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1500,
            }
            resp = await client.get(BINANCE_FUTURES_URL, params=params)
            data = resp.json()

            if isinstance(data, dict) and "code" in data:
                raise ValueError(f"Binance API error: {data}")
            if not data:
                break

            all_data.extend(data)
            start_ms = data[-1][0] + 1
            if len(data) < 1500:
                break

    if not all_data:
        raise ValueError(f"No data returned for {symbol} {interval}")

    import pandas as pd

    pdf = pd.DataFrame(all_data, columns=CSV_COLUMNS)
    pdf["open_time"] = pd.to_datetime(pdf["open_time"], unit="ms")
    pdf["close_time"] = pd.to_datetime(pdf["close_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    pdf[numeric_cols] = pdf[numeric_cols].apply(pd.to_numeric, axis=1)

    filename = f"{symbol.lower()}_{interval}.csv"
    save_path = DATA_DIR / filename
    pdf.to_csv(save_path, index=False)

    return _load_csv(save_path)


async def get_ohlcv(
    symbol: str,
    interval: str,
    end_time: str | None = None,
    days: int = 30,
) -> dict:
    """
    Get OHLCV data for a symbol/interval. Downloads if missing.
    Returns dict with 'candles' and 'volume' arrays for lightweight-charts.
    """
    base_interval, resample_to = RESAMPLE_MAP.get(interval, (interval, None))

    # Try to load existing CSV for the base interval
    csv_path = _find_csv(symbol, base_interval)

    if csv_path is None:
        # Try to find any usable base data we can resample from
        # Priority: check 5m (finest available) then 1h
        for fallback_interval in ["5m", "1h"]:
            alt_path = _find_csv(symbol, fallback_interval)
            if alt_path is not None:
                csv_path = alt_path
                # We'll resample from whatever we found to the target interval
                resample_to = interval if interval != fallback_interval else None
                break

    if csv_path is None:
        # Download from Binance — prefer 5m for finest granularity
        download_interval = "5m" if interval in ("5m", "15m") else "1h"
        download_days = DOWNLOAD_DAYS.get(download_interval, 30)
        await download_ohlcv(symbol, download_interval, days=download_days)
        csv_path = _find_csv(symbol, download_interval)
        if csv_path is None:
            raise ValueError(f"Failed to download data for {symbol}")
        resample_to = interval if interval != download_interval else None

    df = _load_csv(csv_path)

    # Resample if needed
    if resample_to:
        df = resample_ohlcv(df, resample_to)

    # Filter by end_time for replay mode
    if end_time:
        end_dt = pl.Series([end_time]).str.to_datetime("%Y-%m-%dT%H:%M")[0]
        df = df.filter(pl.col("open_time") <= end_dt)

    # Convert to lightweight-charts format
    candles = []
    volume = []
    for row in df.iter_rows(named=True):
        ts = int(row["open_time"].timestamp())
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        v = float(row["volume"])
        candles.append({"time": ts, "open": o, "high": h, "low": l, "close": c})
        color = "#26a69a80" if c >= o else "#ef535080"
        volume.append({"time": ts, "value": v, "color": color})

    return {"candles": candles, "volume": volume}
