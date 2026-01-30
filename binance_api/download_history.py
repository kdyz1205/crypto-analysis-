"""
Download historical OHLCV data for all Binance Futures USDT pairs.
Supports any interval (1m+) and number of days, with parallel downloads and rate limiting.

Usage:
    python download_history.py --days 60 --interval 1h
    python download_history.py --days 7 --interval 1m
    python download_history.py --days 60 --interval 1h --workers 10
"""
import requests
import pandas as pd
import time
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TICKER_FILE = os.path.join(BASE_DIR, "binance_futures_usdt.txt")
TICKER_INFO_FILE = os.path.join(BASE_DIR, "binance_futures_ticker_info.csv")

URL = "https://fapi.binance.com/fapi/v1/klines"
MAX_WEIGHT_PER_MIN = 2400
SAFETY_FACTOR = 0.85  # use 85% of budget to avoid edge-case bans
MAX_LIMIT = 1500

# Interval durations in milliseconds
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000,
    "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
    "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
    "3d": 259_200_000, "1w": 604_800_000,
}


def get_weight(limit):
    """Return API weight cost for a given limit value."""
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit < 1000:
        return 5
    return 10


class RateLimiter:
    """Thread-safe sliding-window rate limiter tracking weight per minute."""

    def __init__(self, max_weight=MAX_WEIGHT_PER_MIN, safety=SAFETY_FACTOR):
        self.budget = int(max_weight * safety)
        self._lock = threading.Lock()
        self._window_start = time.monotonic()
        self._used = 0

    def acquire(self, weight):
        """Block until weight budget is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._window_start

                # Reset window every 60s
                if elapsed >= 60:
                    self._window_start = now
                    self._used = 0
                    elapsed = 0

                if self._used + weight <= self.budget:
                    self._used += weight
                    return  # acquired

                # Must wait for window reset
                wait = 60 - elapsed + 0.1

            # Sleep outside the lock so other threads aren't blocked
            time.sleep(wait)


def load_symbols():
    with open(TICKER_FILE) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("Total")]


def load_ticker_info():
    """Load ticker info CSV → {symbol: start_datetime}."""
    info = {}
    if not os.path.exists(TICKER_INFO_FILE):
        return info
    df = pd.read_csv(TICKER_INFO_FILE)
    for _, row in df.iterrows():
        ticker = row["ticker"]
        st = row.get("start_time")
        if pd.notna(st):
            info[ticker] = pd.to_datetime(st)
    return info


def load_progress(progress_file):
    """Return set of symbols already in the progress CSV."""
    if os.path.exists(progress_file):
        df = pd.read_csv(progress_file)
        return set(df["symbol"].tolist())
    return set()


def init_progress_file(progress_file):
    """Write header if file doesn't exist."""
    if not os.path.exists(progress_file):
        with open(progress_file, "w") as f:
            f.write("symbol,start_date,candles,status,downloaded_at\n")


def append_progress(progress_file, lock, symbol, start_date, candles, status):
    """Append one row to the progress CSV (thread-safe)."""
    with lock:
        with open(progress_file, "a") as f:
            f.write(f"{symbol},{start_date},{candles},{status},{datetime.now().isoformat()}\n")


def download_symbol(symbol, interval, days, rate_limiter, ticker_info):
    """
    Download all candles for a symbol, paginating if needed.
    Returns (candle_count, start_date_str, error_str_or_None).
    """
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    # Don't request before the coin existed
    if symbol in ticker_info:
        coin_start_ms = int(ticker_info[symbol].timestamp() * 1000)
        if coin_start_ms > start_ms:
            start_ms = coin_start_ms

    interval_ms = INTERVAL_MS.get(interval)
    if not interval_ms:
        return None, None, f"Unknown interval: {interval}"

    weight = get_weight(MAX_LIMIT)
    all_data = []
    cursor = start_ms

    while cursor < end_ms:
        rate_limiter.acquire(weight)

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }

        try:
            resp = requests.get(URL, params=params, timeout=30)

            # Handle rate-limit responses
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                time.sleep(retry_after)
                continue
            if resp.status_code == 418:
                time.sleep(120)
                continue

            data = resp.json()

            if isinstance(data, dict) and "code" in data:
                return None, None, f"API error: {data.get('msg')} (code {data.get('code')})"

            if not data:
                break

            all_data.extend(data)

            # Advance cursor past the last candle received
            cursor = data[-1][0] + interval_ms

            if len(data) < MAX_LIMIT:
                break  # no more data available

        except requests.exceptions.RequestException as e:
            return None, None, f"Network error: {e}"

    if not all_data:
        return 0, "", "No data"

    # Deduplicate by open_time (safety for pagination overlap)
    seen = set()
    unique = []
    for row in all_data:
        if row[0] not in seen:
            seen.add(row[0])
            unique.append(row)

    df = pd.DataFrame(unique, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)

    filename = os.path.join(DATA_DIR, f"{symbol.lower()}_{interval}.csv")
    df.to_csv(filename, index=False)

    start_date = df["open_time"].iloc[0].strftime("%Y-%m-%d %H:%M")
    return len(df), start_date, None


def main():
    parser = argparse.ArgumentParser(description="Download Binance Futures OHLCV data")
    parser.add_argument("--days", type=int, required=True, help="Number of days of history")
    parser.add_argument("--interval", type=str, required=True,
                        choices=list(INTERVAL_MS.keys()),
                        help="Candle interval (1m, 5m, 15m, 1h, 4h, 1d, ...)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel download threads (default: 10)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    progress_file = os.path.join(BASE_DIR, f"coins_progress_{args.interval}.csv")
    progress_lock = threading.Lock()

    symbols = load_symbols()
    ticker_info = load_ticker_info()
    done = load_progress(progress_file)
    remaining = [s for s in symbols if s not in done]

    init_progress_file(progress_file)

    # Estimate work
    candles_per_day = 86_400_000 // INTERVAL_MS[args.interval]
    total_candles_per_coin = args.days * candles_per_day
    pages_per_coin = max(1, -(-total_candles_per_coin // MAX_LIMIT))  # ceil div
    weight_per_coin = pages_per_coin * get_weight(MAX_LIMIT)
    total_weight = len(remaining) * weight_per_coin
    est_minutes = total_weight / (MAX_WEIGHT_PER_MIN * SAFETY_FACTOR)

    print(f"Interval: {args.interval} | Days: {args.days} | Workers: {args.workers}")
    print(f"Total: {len(symbols)} | Done: {len(done)} | Remaining: {len(remaining)}")
    print(f"~{pages_per_coin} pages/coin, ~{total_candles_per_coin} candles/coin max")
    print(f"Estimated: ~{est_minutes:.1f} min ({total_weight} total weight)")
    print()

    rate_limiter = RateLimiter()
    completed = [0]  # mutable for closure
    total = len(remaining)
    count_lock = threading.Lock()

    def process(symbol):
        result = download_symbol(symbol, args.interval, args.days, rate_limiter, ticker_info)
        candles, start_date, err = result

        with count_lock:
            completed[0] += 1
            idx = completed[0]

        if err:
            print(f"[{idx}/{total}] SKIP {symbol}: {err}")
            append_progress(progress_file, progress_lock, symbol, "", 0, f"error: {err}")
        else:
            print(f"[{idx}/{total}] OK   {symbol}: {candles} candles from {start_date}")
            append_progress(progress_file, progress_lock, symbol, start_date, candles, "ok")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, s) for s in remaining]
        # Wait for all to complete (process() handles printing/progress)
        for f in as_completed(futures):
            f.result()  # propagate exceptions if any

    print(f"\nDone! All {total} symbols processed.")
    print(f"Progress file: {progress_file}")
    print(f"Data directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
