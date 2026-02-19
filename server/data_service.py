import polars as pl
import httpx
import time
from datetime import timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYMBOLS_FILE = PROJECT_ROOT / "binance_futures_usdt.txt"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"

# Data source configuration
# - exchange: 'binance' or 'okx'
# - offline_only: when True, never call any exchange API and use only local CSVs
# - api_only: when True (and not offline_only), never read from CSV; always fetch latest from API for fullest data
EXCHANGE = "okx"
OFFLINE_ONLY = False
API_ONLY = True

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
    "1d": ("1d", None),
}

# How many days of base data to download per interval
DOWNLOAD_DAYS = {
    "5m": 7,
    "1h": 60,
    "1d": 365,
}

# Interval durations in milliseconds (for staleness / gap detection)
INTERVAL_MS = {
    "5m": 5 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

# When no 5m CSV exists, fetch this many days
FRESH_DOWNLOAD_5M_DAYS = 10

# When chart requests this many days or more (no local CSV), pull max history from OKX (paginated)
FULL_HISTORY_DAYS = 365 * 6  # ~6 years (2020–2026)
OKX_CANDLES_PAGE_LIMIT = 300  # OKX max per request
OKX_START_TS_MS = 1609459200000  # 2021-01-01 00:00:00 UTC


# Cache for OKX swap symbols with precision
_okx_swap_cache: dict[str, dict] | None = None


async def load_okx_swap_symbols() -> dict[str, dict]:
    """
    Fetch OKX perpetual swap (SWAP) instruments and return a dict mapping
    symbol (e.g., 'BTCUSDT') to metadata including price precision (tickSz).
    
    Returns: {symbol: {'instId': 'BTC-USDT-SWAP', 'tickSz': '0.1', ...}, ...}
    """
    global _okx_swap_cache
    
    if _okx_swap_cache is not None:
        return _okx_swap_cache
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(OKX_INSTRUMENTS_URL, params={"instType": "SWAP"})
            data = resp.json()
        
        if not isinstance(data, dict) or data.get("code") != "0":
            # If API fails, return empty dict instead of raising
            print(f"Warning: OKX instruments API error: {data}")
            _okx_swap_cache = {}
            return {}
        
        rows = data.get("data") or []
        result = {}
        
        for row in rows:
            inst_id = row.get("instId", "")
            if not inst_id.endswith("-USDT-SWAP"):
                continue
            
            # Extract base symbol: BTC-USDT-SWAP -> BTCUSDT
            base = inst_id.replace("-USDT-SWAP", "")
            symbol = f"{base}USDT"
            
            result[symbol] = {
                "instId": inst_id,
                "tickSz": row.get("tickSz", "0.1"),  # Price precision
                "lotSz": row.get("lotSz", "1"),      # Size precision
                "ctVal": row.get("ctVal", "0.01"),   # Contract value
                "state": row.get("state", "live"),   # live, suspend, preopen
            }
        
        _okx_swap_cache = result
        return result
    except Exception as e:
        # If API call fails, return empty dict and log error
        print(f"Warning: Failed to load OKX swap symbols: {e}")
        _okx_swap_cache = {}
        return {}


def load_symbols() -> list[str]:
    """
    Load available symbols based on EXCHANGE setting.
    
    For OKX, this returns an empty list (use load_okx_swap_symbols() instead).
    For Binance, reads from the symbols file.
    """
    if EXCHANGE.lower() == "okx":
        # OKX symbols are loaded dynamically via API
        return []
    
    if not SYMBOLS_FILE.exists():
        return []

    text = SYMBOLS_FILE.read_text(encoding="utf-8", errors="ignore")
    lines = text.strip().splitlines()
    return sorted([l.strip() for l in lines if l.strip()])


def _scan_okx_csv_files() -> dict[tuple[str, str], Path]:
    """Scan data/ for OKX_*.csv filenames and return (symbol_lower, interval) -> path."""
    import re
    result = {}
    if not DATA_DIR.exists():
        return result
    for p in DATA_DIR.iterdir():
        if not p.suffix.lower() == ".csv" or not p.name.upper().startswith("OKX_"):
            continue
        # e.g. "OKX_BTCUSDT.P, 1D (4).csv" or "OKX_TRUMPUSDT.P, 1D.csv"
        name = p.stem  # "OKX_BTCUSDT.P, 1D (4)" or "OKX_TRUMPUSDT.P, 1D"
        m = re.match(r"OKX_([A-Z0-9]+)\.P\s*,\s*(\d+[mhdD])", name, re.IGNORECASE)
        if not m:
            continue
        sym = m.group(1).upper()
        if not sym.endswith("USDT"):
            sym = sym + "USDT"
        interval_raw = m.group(2).upper()
        if interval_raw.endswith("D"):
            interval = "1d"
        elif interval_raw.endswith("H"):
            interval = interval_raw[:-1] + "h" if interval_raw[:-1] else "1h"
        elif interval_raw.endswith("M"):
            interval = interval_raw[:-1] + "m" if interval_raw[:-1] else "1m"
        else:
            continue
        result[(sym.lower(), interval)] = p
    return result


def _find_csv(symbol: str, interval: str) -> Path | None:
    """Find CSV file in data/ directory. Supports standard name and OKX_*.csv names."""
    symbol_lower = symbol.lower()
    filename = f"{symbol_lower}_{interval}.csv"
    path = DATA_DIR / filename
    if path.exists():
        return path
    root_path = PROJECT_ROOT / filename
    if root_path.exists():
        return root_path
    okx_map = _scan_okx_csv_files()
    return okx_map.get((symbol_lower, interval))


def _load_csv(path: Path) -> pl.DataFrame:
    """Load a CSV and parse open_time as datetime. Handles OKX export (time=unix, Volume)."""
    df = pl.read_csv(path)

    # OKX export: "time" (unix sec), "Volume" (capital V), optional extra columns
    if "time" in df.columns and "open_time" not in df.columns:
        df = df.with_columns(
            pl.from_epoch(pl.col("time"), time_unit="s").alias("open_time")
        )
    if "Volume" in df.columns and "volume" not in df.columns:
        df = df.with_columns(pl.col("Volume").alias("volume"))

    # Keep only OHLCV + open_time for consistency
    want = ["open_time", "open", "high", "low", "close", "volume"]
    have = [c for c in want if c in df.columns]
    if len(have) < 6:
        raise ValueError(f"CSV missing required columns; need {want}, got {list(df.columns)}")
    df = df.select(have)

    if df["open_time"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("open_time").str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
    for col in ["open", "high", "low", "close", "volume"]:
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
    """
    Download OHLCV data from the configured exchange and save to CSV.

    For EXCHANGE == 'okx', this uses the public OKX candles endpoint.
    For EXCHANGE == 'binance', it uses the Binance Futures klines endpoint
    (original behaviour).
    """
    if EXCHANGE.lower() == "okx":
        return await _download_ohlcv_okx(symbol, interval, days)
    else:
        return await _download_ohlcv_binance(symbol, interval, days)


async def _download_ohlcv_binance(symbol: str, interval: str, days: int = 30) -> pl.DataFrame:
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

    if API_ONLY:
        result = pl.from_pandas(pdf).select(["open_time", "open", "high", "low", "close", "volume"])
        if result["open_time"].dtype == pl.Datetime("ns"):
            result = result.with_columns(pl.col("open_time").cast(pl.Datetime("us")))
        return result.sort("open_time")

    DATA_DIR.mkdir(exist_ok=True)
    filename = f"{symbol.lower()}_{interval}.csv"
    save_path = DATA_DIR / filename
    pdf.to_csv(save_path, index=False)
    return _load_csv(save_path)


def _okx_rows_to_records(rows: list) -> list:
    """Convert OKX candle rows [ts, o, h, l, c, vol, ...] to CSV_COLUMNS records."""
    records = []
    for r in rows:
        ts_ms = int(r[0])
        o = float(r[1])
        h = float(r[2])
        l = float(r[3])
        c = float(r[4])
        vol = float(r[5])
        quote_vol = float(r[7]) if len(r) > 7 and r[7] is not None else 0.0
        records.append([
            ts_ms, o, h, l, c, vol, ts_ms, quote_vol, 0, 0.0, 0.0, 0,
        ])
    return records


async def _download_ohlcv_okx(symbol: str, interval: str, days: int = 30) -> pl.DataFrame:
    """
    Download OHLCV data from OKX perpetual swap (SWAP) candles API.

    - Maps HYPEUSDT → HYPE-USDT-SWAP.
    - When days >= 365, uses pagination (after + limit 300) to pull up to ~6 years (2021–2026).
    - OKX returns newest-first; we collect then reverse to oldest-first.
    """
    import math
    import pandas as pd
    import asyncio

    DATA_DIR.mkdir(exist_ok=True)

    swap_symbols = await load_okx_swap_symbols()
    symbol_upper = symbol.upper()
    if symbol_upper not in swap_symbols:
        base = symbol_upper.replace("USDT", "")
        inst_id = f"{base}-USDT-SWAP"
    else:
        inst_id = swap_symbols[symbol_upper]["instId"]

    interval_map = {"5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}
    bar = interval_map.get(interval)
    if bar is None:
        raise ValueError(f"Unsupported interval for OKX: {interval}")

    all_records = []
    minutes_per_bar = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}[interval]
    approx_candles = days * 24 * 60 / minutes_per_bar
    use_pagination = approx_candles > OKX_CANDLES_PAGE_LIMIT
    target_oldest_ms = int(time.time() * 1000) - days * 24 * 3600 * 1000  # stop when we have this much history

    if use_pagination:
        # Paginate: request 300 at a time, stop when we have enough days or hit OKX_START_TS_MS
        after_ts = None  # first request gets latest
        while True:
            params = {"instId": inst_id, "bar": bar, "limit": str(OKX_CANDLES_PAGE_LIMIT)}
            if after_ts is not None:
                params["after"] = str(after_ts)
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(OKX_CANDLES_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()
            except httpx.TimeoutException:
                raise ValueError(f"OKX API timeout for {symbol} {interval}. Please try again.")
            except httpx.RequestError as e:
                raise ValueError(f"OKX API request failed for {symbol} {interval}: {e}")

            if not isinstance(data, dict) or data.get("code") != "0":
                error_msg = data.get("msg", "Unknown error")
                raise ValueError(f"OKX API error for {symbol} {interval}: {error_msg} (code: {data.get('code')})")

            rows = data.get("data") or []
            if not rows:
                break
            # OKX returns newest first; we want oldest first at the end, so we reverse and append
            rows_oldest_first = list(reversed(rows))
            all_records.extend(_okx_rows_to_records(rows_oldest_first))
            oldest_ts = int(rows_oldest_first[0][0])
            if oldest_ts <= OKX_START_TS_MS or oldest_ts <= target_oldest_ms:
                break
            after_ts = oldest_ts  # next page: records earlier than this ts
            await asyncio.sleep(0.06)  # ~40/2s rate limit → ~50ms between requests

        if not all_records:
            raise ValueError(f"No data returned from OKX for {symbol} {interval}. The symbol may not exist or have no trading history.")
    else:
        limit = int(min(OKX_CANDLES_PAGE_LIMIT, max(100, math.ceil(approx_candles))))

        params = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(OKX_CANDLES_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            raise ValueError(f"OKX API timeout for {symbol} {interval}. Please try again.")
        except httpx.RequestError as e:
            raise ValueError(f"OKX API request failed for {symbol} {interval}: {e}")

        if not isinstance(data, dict) or data.get("code") != "0":
            error_msg = data.get("msg", "Unknown error")
            raise ValueError(f"OKX API error for {symbol} {interval}: {error_msg} (code: {data.get('code')})")

        rows = data.get("data") or []
        if not rows:
            raise ValueError(f"No data returned from OKX for {symbol} {interval}. The symbol may not exist or have no trading history.")
        rows = list(reversed(rows))
        all_records = _okx_rows_to_records(rows)

    pdf = pd.DataFrame(all_records, columns=CSV_COLUMNS)
    pdf["open_time"] = pd.to_datetime(pdf["open_time"], unit="ms")
    pdf["close_time"] = pd.to_datetime(pdf["close_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    pdf[numeric_cols] = pdf[numeric_cols].apply(pd.to_numeric, axis=1)
    pdf = pdf.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")

    if API_ONLY:
        result = pl.from_pandas(pdf).select(["open_time", "open", "high", "low", "close", "volume"])
        if result["open_time"].dtype == pl.Datetime("ns"):
            result = result.with_columns(pl.col("open_time").cast(pl.Datetime("us")))
        return result.sort("open_time")

    DATA_DIR.mkdir(exist_ok=True)
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


async def _fetch_candles_since_okx(symbol: str, interval: str, start_ms: int) -> pl.DataFrame:
    """Fetch OKX swap candles from start_ms to now for incremental updates."""
    import pandas as pd
    
    swap_symbols = await load_okx_swap_symbols()
    symbol_upper = symbol.upper()
    
    # If symbol not found, try constructing instId directly
    if symbol_upper not in swap_symbols:
        base = symbol_upper.replace("USDT", "")
        inst_id = f"{base}-USDT-SWAP"
    else:
        inst_id = swap_symbols[symbol_upper]["instId"]
    
    # Map interval to OKX bar
    interval_map = {
        "5m": "5m",
        "15m": "15m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }
    bar = interval_map.get(interval)
    if bar is None:
        return pl.DataFrame()
    
    # OKX accepts after parameter (timestamp in ISO format or ms)
    # We'll use limit to get recent candles and filter
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": "200",  # Get last 200 candles, filter by timestamp
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(OKX_CANDLES_URL, params=params)
        data = resp.json()
    
    if not isinstance(data, dict) or data.get("code") != "0":
        return pl.DataFrame()
    
    rows = data.get("data") or []
    if not rows:
        return pl.DataFrame()
    
    # OKX returns newest-first; reverse to oldest-first
    rows = list(reversed(rows))
    
    # Filter rows where timestamp >= start_ms
    filtered_rows = []
    for r in rows:
        ts_ms = int(r[0])
        if ts_ms >= start_ms:
            filtered_rows.append(r)
    
    if not filtered_rows:
        return pl.DataFrame()
    
    # Convert to DataFrame
    records = []
    for r in filtered_rows:
        ts_ms = int(r[0])
        o = float(r[1])
        h = float(r[2])
        l = float(r[3])
        c = float(r[4])
        vol = float(r[5])
        quote_vol = float(r[7]) if len(r) > 7 and r[7] is not None else 0.0
        records.append([
            ts_ms, o, h, l, c, vol, ts_ms, quote_vol, 0, 0.0, 0.0, 0,
        ])
    
    pdf = pd.DataFrame(records, columns=CSV_COLUMNS)
    pdf["open_time"] = pd.to_datetime(pdf["open_time"], unit="ms")
    pdf["close_time"] = pd.to_datetime(pdf["close_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    pdf[numeric_cols] = pdf[numeric_cols].apply(pd.to_numeric, axis=1)
    
    result = pl.from_pandas(pdf)
    # Match schema
    casts = []
    if result["open_time"].dtype == pl.Datetime("ns"):
        casts.append(pl.col("open_time").cast(pl.Datetime("us")))
    if "close_time" in result.columns and result["close_time"].dtype != pl.Utf8:
        casts.append(pl.col("close_time").cast(pl.Utf8))
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume",
                "taker_buy_quote_asset_volume"]:
        if col in result.columns and result[col].dtype != pl.Float64:
            casts.append(pl.col(col).cast(pl.Float64))
    if casts:
        result = result.with_columns(casts)
    return result.sort("open_time")


async def _fetch_candles_since(symbol: str, interval: str, start_ms: int) -> pl.DataFrame:
    """Fetch candles from the configured exchange from start_ms to now."""
    if EXCHANGE.lower() == "okx":
        return await _fetch_candles_since_okx(symbol, interval, start_ms)
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


async def _append_okx_live(df: pl.DataFrame, symbol: str, base_interval: str) -> pl.DataFrame:
    """
    Fetch latest candles from OKX API since last bar in df, merge in memory only (no save).
    Use this for OKX export CSVs so we show CSV history + real-time tail without overwriting the file.
    """
    if df.is_empty():
        return df
    last_ms = _get_last_timestamp_ms(df)
    now_ms = int(time.time() * 1000)
    interval_ms = INTERVAL_MS.get(base_interval, 60 * 60 * 1000)
    if (now_ms - last_ms) < interval_ms:
        return df
    new_df = await _fetch_candles_since(symbol, base_interval, start_ms=last_ms)
    if new_df.is_empty():
        return df
    # Merge in memory: concat + dedupe by open_time (keep last), sort
    casts = []
    for col_name, col_type in df.schema.items():
        if col_name in new_df.columns and new_df[col_name].dtype != col_type:
            casts.append(pl.col(col_name).cast(col_type))
    if casts:
        new_df = new_df.with_columns(casts)
    combined = pl.concat([df, new_df])
    return combined.unique(subset=["open_time"], keep="last").sort("open_time")


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
    Get OHLCV data for a symbol/interval.

    In OFFLINE_ONLY mode, this will **only** use existing CSV files and will
    never hit the Binance API, so you won't see IP/region restriction errors.
    """
    base_interval, resample_to = RESAMPLE_MAP.get(interval, (interval, None))
    loaded_from_csv = False

    if OFFLINE_ONLY:
        # Pure offline path: look for an exact match CSV first.
        csv_path = _find_csv(symbol, base_interval)

        # If not found, try a couple of sensible fallbacks (5m/1h) and resample.
        if csv_path is None:
            for fallback in ["5m", "1h"]:
                alt = _find_csv(symbol, fallback)
                if alt is not None:
                    csv_path = alt
                    # Resample from fallback to requested interval if needed
                    resample_to = interval if interval != fallback else None
                    break

        if csv_path is None:
            raise ValueError(
                f"No local data found for {symbol} {interval}. "
                f"Place a CSV like '{symbol.lower()}_{base_interval}.csv' in the data/ folder."
            )

        df = _load_csv(csv_path)
        loaded_from_csv = True
    else:
        # Online mode: API_ONLY = always fetch from API. Use requested days so chart loads in reasonable time.
        if API_ONLY:
            download_days = days
            try:
                df = await download_ohlcv(symbol, base_interval, days=download_days)
            except ValueError as e:
                raise
            except Exception as e:
                raise ValueError(f"Failed to download data for {symbol} {base_interval}: {str(e)}\n\nPossible causes:\n- Symbol does not exist on {EXCHANGE.upper()}\n- Network connection issue\n- API rate limit or restriction")
        else:
            csv_path = _find_csv(symbol, base_interval)
            if csv_path is not None:
                try:
                    df = _load_csv(csv_path)
                    loaded_from_csv = True
                    is_okx_file = csv_path.name.upper().startswith("OKX_")
                    if not is_okx_file:
                        last_ms = _get_last_timestamp_ms(df)
                        now_ms = int(time.time() * 1000)
                        interval_ms = INTERVAL_MS.get(base_interval, 60 * 60 * 1000)
                        if (now_ms - last_ms) >= interval_ms * 2:
                            try:
                                df = await _incremental_update(symbol, base_interval)
                            except Exception as e:
                                print(f"Warning: Incremental update failed for {symbol}, using existing CSV: {e}")
                    else:
                        if EXCHANGE.lower() == "okx":
                            try:
                                df = await _append_okx_live(df, symbol, base_interval)
                            except Exception as e:
                                print(f"Warning: OKX live append failed for {symbol}, using CSV only: {e}")
                except Exception:
                    download_days = max(days, DOWNLOAD_DAYS.get(base_interval, 60))
                    df = await download_ohlcv(symbol, base_interval, days=download_days)
            else:
                download_days = max(days, DOWNLOAD_DAYS.get(base_interval, 60))
                try:
                    df = await download_ohlcv(symbol, base_interval, days=download_days)
                except ValueError as e:
                    raise
                except Exception as e:
                    raise ValueError(f"Failed to download data for {symbol} {base_interval}: {str(e)}\n\nPossible causes:\n- Symbol does not exist on {EXCHANGE.upper()}\n- Network connection issue\n- API rate limit or restriction")

    # Resample if needed
    if resample_to:
        df = resample_ohlcv(df, resample_to)

    # When data is from local CSV we return full history (no days limit). Downloaded data is already limited by DOWNLOAD_DAYS.

    # Filter by end_time for replay mode
    if end_time:
        end_dt = pl.Series([end_time]).str.to_datetime("%Y-%m-%dT%H:%M")[0]
        df = df.filter(pl.col("open_time") <= end_dt)

    # Get price precision for OKX symbols
    price_precision = None
    if EXCHANGE.lower() == "okx":
        try:
            swap_symbols = await load_okx_swap_symbols()
            symbol_upper = symbol.upper()
            if symbol_upper in swap_symbols:
                tick_sz = swap_symbols[symbol_upper]["tickSz"]
                # OKX-style: decimal places from tickSz ("0.1"->1, "0.01"->2, "0.0001"->4, "1"->0)
                if "." in str(tick_sz):
                    price_precision = len(str(tick_sz).split(".")[1])
                else:
                    price_precision = 0
        except Exception:
            pass
    
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

    result = {"candles": candles, "volume": volume}
    if price_precision is not None:
        result["pricePrecision"] = price_precision
    return result


async def get_ohlcv_with_df(
    symbol: str,
    interval: str,
    end_time: str | None = None,
    days: int = 30,
):
    """
    Same as get_ohlcv but returns (df, result_dict).
    Used so chart endpoint can run pattern detection on the same df as candles.
    """
    base_interval, resample_to = RESAMPLE_MAP.get(interval, (interval, None))

    if OFFLINE_ONLY:
        csv_path = _find_csv(symbol, base_interval)
        if csv_path is None:
            for fallback in ["5m", "1h"]:
                alt = _find_csv(symbol, fallback)
                if alt is not None:
                    csv_path = alt
                    resample_to = interval if interval != fallback else None
                    break
        if csv_path is None:
            raise ValueError(
                f"No local data found for {symbol} {interval}. "
                f"Place a CSV like '{symbol.lower()}_{base_interval}.csv' in the data/ folder."
            )
        df = _load_csv(csv_path)
        # From local CSV: full history is returned (no days limit)
    else:
        if API_ONLY:
            download_days = max(days, 365)
            try:
                df = await download_ohlcv(symbol, base_interval, days=download_days)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Failed to download data for {symbol} {base_interval}: {str(e)}")
        else:
            csv_path = _find_csv(symbol, base_interval)
            if csv_path is not None:
                try:
                    df = _load_csv(csv_path)
                    is_okx_file = csv_path.name.upper().startswith("OKX_")
                    if not is_okx_file:
                        last_ms = _get_last_timestamp_ms(df)
                        now_ms = int(time.time() * 1000)
                        interval_ms = INTERVAL_MS.get(base_interval, 60 * 60 * 1000)
                        if (now_ms - last_ms) >= interval_ms * 2:
                            try:
                                df = await _incremental_update(symbol, base_interval)
                            except Exception as e:
                                print(f"Warning: Incremental update failed for {symbol}, using existing CSV: {e}")
                    else:
                        if EXCHANGE.lower() == "okx":
                            try:
                                df = await _append_okx_live(df, symbol, base_interval)
                            except Exception as e:
                                print(f"Warning: OKX live append failed for {symbol}, using CSV only: {e}")
                except Exception:
                    download_days = max(days, DOWNLOAD_DAYS.get(base_interval, 60))
                    df = await download_ohlcv(symbol, base_interval, days=download_days)
            else:
                download_days = max(days, DOWNLOAD_DAYS.get(base_interval, 60))
                try:
                    df = await download_ohlcv(symbol, base_interval, days=download_days)
                except ValueError:
                    raise
                except Exception as e:
                    raise ValueError(f"Failed to download data for {symbol} {base_interval}: {str(e)}")

    if resample_to:
        df = resample_ohlcv(df, resample_to)

    if end_time:
        end_dt = pl.Series([end_time]).str.to_datetime("%Y-%m-%dT%H:%M")[0]
        df = df.filter(pl.col("open_time") <= end_dt)

    price_precision = None
    if EXCHANGE.lower() == "okx":
        try:
            swap_symbols = await load_okx_swap_symbols()
            symbol_upper = symbol.upper()
            if symbol_upper in swap_symbols:
                tick_sz = swap_symbols[symbol_upper]["tickSz"]
                if "." in str(tick_sz):
                    price_precision = len(str(tick_sz).split(".")[1])
                else:
                    price_precision = 0
        except Exception:
            pass

    candles = []
    volume = []
    for row in df.iter_rows(named=True):
        ts = int(row["open_time"].timestamp())
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        v = float(row["volume"])
        candles.append({"time": ts, "open": o, "high": h, "low": l, "close": c})
        color = "#26a69a80" if c >= o else "#ef535080"
        volume.append({"time": ts, "value": v, "color": color})

    result = {"candles": candles, "volume": volume}
    if price_precision is not None:
        result["pricePrecision"] = price_precision
    return df, result
