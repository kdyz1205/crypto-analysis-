"""
Market data routes: symbols, OHLCV, chart overlays, top volume, data info.
"""

import csv
import logging

from fastapi import APIRouter, Query, HTTPException

from ..core.config import PROJECT_ROOT
from ..data_service import (
    load_symbols, load_okx_swap_symbols, get_ohlcv, get_ohlcv_with_df, API_ONLY,
)
from ..pattern_service import get_patterns_from_df

router = APIRouter(prefix="/api", tags=["market"])


# ── Helpers (moved verbatim from app.py) ─────────────────────────────────

def _symbols_from_data_folder() -> list[str]:
    """Collect unique symbols from data/*.csv and data/*.csv.gz."""
    from ..data_service import DATA_DIR, _scan_okx_csv_files
    symbols = set()
    if not DATA_DIR.exists():
        return []
    for p in DATA_DIR.iterdir():
        is_csv = p.suffix.lower() == ".csv" or p.name.lower().endswith(".csv.gz")
        if not is_csv:
            continue
        name = p.stem.lower()
        if name.endswith('.csv'):
            name = name[:-4]
        if name.startswith("okx_"):
            okx = _scan_okx_csv_files()
            for (sym, _), _ in okx.items():
                symbols.add(sym.upper())
        else:
            parts = name.split("_")
            if len(parts) >= 2 and parts[1] in ("1m", "5m", "15m", "1h", "2h", "4h", "1d"):
                symbols.add(parts[0].upper())
    return sorted(symbols)


def _symbols_from_ticker_info_csv() -> list[str]:
    """Fallback: parse symbols from binance_futures_ticker_info.csv."""
    path = PROJECT_ROOT / "binance_futures_ticker_info.csv"
    if not path.exists():
        return []
    out = set()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = (row.get("symbol") or row.get("Symbol") or row.get("ticker") or row.get("Ticker") or "").strip().upper()
                if sym and sym.endswith("USDT"):
                    out.add(sym)
    except Exception as e:
        logging.warning(f"Failed to load extended symbols: {e}")
        return []
    return sorted(out)


# ── Routes ───────────────────────────────────────────────────────────────

@router.get("/symbols")
async def get_symbols_route(include_extended: bool = Query(False, description="Include extended fallback universe from ticker info CSV")):
    """Return all available ticker symbols: API + symbols from data/*.csv."""
    from ..data_service import EXCHANGE

    from_data = _symbols_from_data_folder()
    from_info_csv = _symbols_from_ticker_info_csv() if include_extended else []

    if EXCHANGE.lower() == "okx":
        try:
            swap_symbols = await load_okx_swap_symbols()
            if swap_symbols and len(swap_symbols) > 0:
                api_symbols = sorted(swap_symbols.keys())
            else:
                api_symbols = []
        except Exception as e:
            print(f"Warning: Failed to load OKX symbols: {e}")
            api_symbols = []
        combined = sorted(set(api_symbols) | set(s.upper() for s in from_data) | set(from_info_csv))
        if not combined and not include_extended:
            fallback = _symbols_from_ticker_info_csv()
            if fallback:
                return fallback
        return combined if combined else (from_data or from_info_csv or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT'])
    else:
        try:
            file_symbols = load_symbols()
            combined = sorted(set(file_symbols or []) | set(s.upper() for s in from_data) | set(from_info_csv))
            if not combined and not include_extended:
                fallback = _symbols_from_ticker_info_csv()
                if fallback:
                    return fallback
            return combined if combined else (from_data or from_info_csv or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT'])
        except Exception as e:
            print(f"Warning: Failed to load symbols: {e}")
            return from_data or from_info_csv or ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT']


@router.get("/symbol-info")
async def get_symbol_info(symbol: str = Query(...)):
    """Return metadata for a specific symbol (e.g., price precision)."""
    from ..data_service import EXCHANGE, load_okx_swap_symbols

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    if EXCHANGE.lower() == "okx":
        swap_symbols = await load_okx_swap_symbols()
        if symbol in swap_symbols:
            info = swap_symbols[symbol]
            tick_sz = info["tickSz"]
            if "." in str(tick_sz):
                precision = len(str(tick_sz).split(".")[1])
            else:
                precision = 0
            return {
                "symbol": symbol,
                "instId": info["instId"],
                "pricePrecision": precision,
                "tickSz": tick_sz,
            }

    return {"symbol": symbol, "pricePrecision": None}


@router.get("/chart")
async def api_chart(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(365, description="Days of data"),
):
    """Return OHLCV + support/resistance lines in one response."""
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        df, ohlcv = await get_ohlcv_with_df(symbol, interval, end_time, days)
        patterns = get_patterns_from_df(df, symbol, interval, end_time)
        return {**ohlcv, **patterns}
    except ValueError as e:
        print(f"ValueError in /api/chart for {symbol} {interval}: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        print(f"Exception in /api/chart: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@router.get("/ohlcv")
async def api_ohlcv(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(30, description="Days of data to fetch"),
):
    """Return OHLCV data as JSON."""
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        result = await get_ohlcv(symbol, interval, end_time, days)
        return result
    except ValueError as e:
        print(f"ValueError in /api/ohlcv for {symbol} {interval}: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        error_detail = f"Error fetching data for {symbol} {interval}: {e}\n{traceback.format_exc()}"
        print(f"Exception in /api/ohlcv: {error_detail}")
        raise HTTPException(500, f"Error fetching data: {str(e)}")


@router.get("/top-volume")
async def api_top_volume(n: int = Query(20, ge=1, le=50)):
    """Get top N symbols by 24h trading volume."""
    from ..data_service import get_top_volume_symbols
    symbols = await get_top_volume_symbols(n)
    return {"symbols": symbols, "count": len(symbols)}


@router.get("/data-info")
async def api_data_info(
    symbol: str = Query(...),
    interval: str = Query("4h"),
):
    """Return data completeness metadata for a symbol/interval."""
    from ..data_service import _find_csv, _load_csv, INTERVAL_MS, DOWNLOAD_DAYS
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    info = {
        "symbol": symbol,
        "interval": interval,
        "has_local_csv": False,
        "total_bars": 0,
        "data_start": None,
        "data_end": None,
        "coverage_days": 0,
        "missing_bars_estimate": 0,
        "max_available_days": DOWNLOAD_DAYS.get(interval, 30),
    }

    csv_path = _find_csv(symbol, interval)
    if csv_path:
        info["has_local_csv"] = True
        try:
            df = _load_csv(csv_path)
            if not df.is_empty():
                info["total_bars"] = len(df)
                info["data_start"] = str(df["open_time"].min())
                info["data_end"] = str(df["open_time"].max())
                start_ts = df["open_time"].min().timestamp()
                end_ts = df["open_time"].max().timestamp()
                info["coverage_days"] = round((end_ts - start_ts) / 86400, 1)
                interval_sec = INTERVAL_MS.get(interval, 3600000) / 1000
                expected_bars = int((end_ts - start_ts) / interval_sec) + 1
                info["missing_bars_estimate"] = max(0, expected_bars - len(df))
        except Exception as e:
            info["error"] = str(e)

    return info
