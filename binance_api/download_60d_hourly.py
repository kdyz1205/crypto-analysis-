"""
Download past 60 days of hourly OHLCV data for all Binance Futures USDT pairs.
Saves each coin to data/{symbol}_1h.csv and tracks progress in data/progress.txt
"""
import requests
import pandas as pd
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.txt")
TICKER_FILE = os.path.join(BASE_DIR, "binance_futures_usdt.txt")

DAYS = 60
INTERVAL = "1h"
LIMIT = DAYS * 24  # 1440 candles, within 1500 max

URL = "https://fapi.binance.com/fapi/v1/klines"


def load_symbols():
    with open(TICKER_FILE) as f:
        symbols = [line.strip() for line in f if line.strip() and not line.startswith("Total")]
    return symbols


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(symbol):
    with open(PROGRESS_FILE, "a") as f:
        f.write(symbol + "\n")


def download_symbol(symbol):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - DAYS * 24 * 60 * 60 * 1000

    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1500,
    }

    try:
        resp = requests.get(URL, params=params, timeout=15)
        data = resp.json()

        if isinstance(data, dict) and "code" in data:
            return None, f"API error: {data.get('msg')} (code {data.get('code')})"

        if not data:
            return None, "No data returned"

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)

        filename = os.path.join(DATA_DIR, f"{symbol.lower()}_{INTERVAL}.csv")
        df.to_csv(filename, index=False)
        return len(df), None

    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    symbols = load_symbols()
    done = load_progress()
    remaining = [s for s in symbols if s not in done]

    print(f"Total symbols: {len(symbols)}, Already done: {len(done)}, Remaining: {len(remaining)}")

    for i, symbol in enumerate(remaining):
        count, err = download_symbol(symbol)
        if err:
            print(f"[{i+1}/{len(remaining)}] SKIP {symbol}: {err}")
            # Still mark as processed to avoid retrying invalid symbols
            save_progress(symbol)
        else:
            print(f"[{i+1}/{len(remaining)}] OK   {symbol}: {count} candles")
            save_progress(symbol)

        # Rate limit: ~5 requests per second to stay well under Binance limits
        time.sleep(0.25)

    print("\nDone! All symbols processed.")


if __name__ == "__main__":
    main()
