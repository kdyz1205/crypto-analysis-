import polars as pl
import httpx
import time
from datetime import timezone
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
    "1h": 60,
}

# Interval durations in milliseconds (for staleness / gap detection)
INTERVAL_MS = {
    "5m": 5 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}

# When no 5m CSV exists, fetch this many days
FRESH_DOWNLOAD_5M_DAYS = 10


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


# ── Incremental update helpers ──

def _get_last_timestamp_ms(df: pl.DataFrame) -> int:
    """Get the last open_time as epoch milliseconds (UTC)."""
    last_dt = df["open_time"].max()
    # Binance data is UTC; naive datetimes must be tagged as UTC for correct epoch
    return int(last_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


async def _fetch_candles_since(symbol: str, interval: str, start_ms: int) -> pl.DataFrame:
    """Fetch candles from Binance from start_ms to now. Returns polars DataFrame."""
    end_ms = int(time.time() * 1000)
    all_data = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
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
            cursor = data[-1][0] + 1
            if len(data) < 1500:
                break

    if not all_data:
        return pl.DataFrame()

    import pandas as pd
    pdf = pd.DataFrame(all_data, columns=CSV_COLUMNS)
    pdf["open_time"] = pd.to_datetime(pdf["open_time"], unit="ms")
    pdf["close_time"] = pd.to_datetime(pdf["close_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        pdf[col] = pd.to_numeric(pdf[col])

    result = pl.from_pandas(pdf)
    # Match schema with CSVs loaded by _load_csv
    casts = []
    # open_time: pandas Datetime(ns) → Datetime(us) to match polars CSV loads
    if result["open_time"].dtype == pl.Datetime("ns"):
        casts.append(pl.col("open_time").cast(pl.Datetime("us")))
    # close_time: convert to String to match how _load_csv leaves it
    if "close_time" in result.columns and result["close_time"].dtype != pl.Utf8:
        casts.append(pl.col("close_time").cast(pl.Utf8))
    # Float64 columns to match _load_csv's casting
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume",
                "taker_buy_quote_asset_volume"]:
        if col in result.columns and result[col].dtype != pl.Float64:
            casts.append(pl.col(col).cast(pl.Float64))
    if casts:
        result = result.with_columns(casts)
    return result.sort("open_time")


def _merge_and_save(existing_df: pl.DataFrame, new_df: pl.DataFrame, symbol: str, interval: str) -> pl.DataFrame:
    """Deduplicate, merge, sort, and save to CSV. Returns merged DataFrame."""
    # Align new_df schema to match existing_df
    casts = []
    for col_name, col_type in existing_df.schema.items():
        if col_name in new_df.columns and new_df[col_name].dtype != col_type:
            casts.append(pl.col(col_name).cast(col_type))
    if casts:
        new_df = new_df.with_columns(casts)
    combined = pl.concat([existing_df, new_df])
    combined = combined.unique(subset=["open_time"], keep="last").sort("open_time")

    # Save with open_time formatted as string for CSV consistency
    DATA_DIR.mkdir(exist_ok=True)
    save_path = DATA_DIR / f"{symbol.lower()}_{interval}.csv"
    df_out = combined.with_columns(
        pl.col("open_time").dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    if "close_time" in df_out.columns and df_out["close_time"].dtype != pl.Utf8:
        df_out = df_out.with_columns(pl.col("close_time").cast(pl.Utf8))
    df_out.write_csv(save_path)

    return combined


async def _incremental_update(symbol: str, base_interval: str) -> pl.DataFrame:
    """
    Load existing CSV, fetch missing candles, merge, save, return full DataFrame.
    Fresh-downloads if no CSV exists or if a gap is detected.
    """
    csv_path = _find_csv(symbol, base_interval)

    if csv_path is None:
        # No existing data — fresh download
        days = FRESH_DOWNLOAD_5M_DAYS if base_interval == "5m" else DOWNLOAD_DAYS.get(base_interval, 60)
        return await download_ohlcv(symbol, base_interval, days=days)

    try:
        existing_df = _load_csv(csv_path)
    except Exception:
        # Corrupt CSV — re-download
        csv_path.unlink(missing_ok=True)
        days = FRESH_DOWNLOAD_5M_DAYS if base_interval == "5m" else DOWNLOAD_DAYS.get(base_interval, 60)
        return await download_ohlcv(symbol, base_interval, days=days)

    if existing_df.is_empty():
        days = FRESH_DOWNLOAD_5M_DAYS if base_interval == "5m" else DOWNLOAD_DAYS.get(base_interval, 60)
        return await download_ohlcv(symbol, base_interval, days=days)

    last_ms = _get_last_timestamp_ms(existing_df)
    now_ms = int(time.time() * 1000)
    interval_ms = INTERVAL_MS[base_interval]

    # Already current — skip API call
    if (now_ms - last_ms) < interval_ms:
        return existing_df

    # Fetch from last known timestamp (overlap by 1 candle for dedup/validation)
    new_df = await _fetch_candles_since(symbol, base_interval, start_ms=last_ms)

    if new_df.is_empty():
        return existing_df

    # Gap detection: first new candle should be at most 1 interval after last existing
    first_new_ms = int(new_df["open_time"].min().replace(tzinfo=timezone.utc).timestamp() * 1000)
    expected_next_ms = last_ms + interval_ms
    if first_new_ms > expected_next_ms:
        # Gap detected — full re-download
        days = FRESH_DOWNLOAD_5M_DAYS if base_interval == "5m" else DOWNLOAD_DAYS.get(base_interval, 60)
        return await download_ohlcv(symbol, base_interval, days=days)

    return _merge_and_save(existing_df, new_df, symbol, base_interval)


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

    # Incrementally update base data (fetch missing candles, merge, save)
    df = await _incremental_update(symbol, base_interval)

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
