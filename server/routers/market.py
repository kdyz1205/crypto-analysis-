"""
Market data routes: symbols, OHLCV, chart overlays, top volume, data info.
"""

import asyncio
import csv
import logging
from collections import OrderedDict

from fastapi import APIRouter, Query, HTTPException

from ..core.config import PROJECT_ROOT
from ..data_service import (
    load_symbols, load_swap_symbols, load_bitget_swap_symbols,
    get_symbol_metadata, get_ohlcv, get_ohlcv_with_df, API_ONLY,
)
from ..pattern_service import get_patterns_from_df

router = APIRouter(prefix="/api", tags=["market"])
VALID_HISTORY_MODES = {"fast_window", "full_history"}
STRUCTURE_SUMMARY_CACHE_LIMIT = 32
STRUCTURE_SUMMARY_LOOKBACK_BARS = {
    "5m": 600,
    "15m": 600,
    "1h": 720,
    "4h": 540,
    "1d": 365,
}
_structure_summary_cache: OrderedDict[tuple[object, ...], dict] = OrderedDict()


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

# Bitget lists equity/commodity perpetuals under the same usdt-futures
# product type. These bases are NOT crypto — filter them out so the symbol
# picker only shows what the user actually trades.
_NON_CRYPTO_BASES = frozenset({
    # US equities
    "AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META", "COIN", "MSTR",
    "PLTR", "HOOD", "AMD", "INTC", "NFLX", "DIS", "BABA", "PDD", "JD", "TSM",
    # Precious metals / commodities (XAUT is a crypto gold token — keep it)
    "XAU", "XAG", "XPT", "XPD", "CL", "BZ", "NG", "HG", "GC", "SI",
    "BRENT", "WTI", "GOLD", "SILVER", "OIL", "COPPER",
    # Equity indices / ETFs
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "UNG", "TLT", "IEF",
    "SPX", "NDX", "DJI", "RUT", "VIX",
})

def _is_crypto_symbol(sym: str) -> bool:
    if not sym.endswith("USDT"):
        return False
    base = sym[:-4]
    return base not in _NON_CRYPTO_BASES


@router.get("/symbols")
async def get_symbols_route(include_extended: bool = Query(False, description="[Deprecated]")):
    """Return Bitget USDT-M crypto perpetual symbols ranked by 24h volume."""
    from ..data_service import get_top_volume_symbols
    try:
        ranked = await get_top_volume_symbols(top_n=300)
        crypto_only = [s for s in ranked if _is_crypto_symbol(s)]
        if crypto_only:
            return crypto_only[:200]
    except Exception as e:
        print(f"Warning: Failed to load Bitget symbols: {e}")
    return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'HYPEUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT']


# 10-minute cache. Bitget ticker API returns ~540 symbols in one call —
# the user's dropdown re-opens many times per session, no reason to hit
# Bitget each time. TTL matches `get_top_volume_symbols` so both views
# stay consistent.
_extended_cache: tuple[float, list[dict]] | None = None
_EXTENDED_CACHE_TTL = 600


@router.get("/symbols/extended")
async def get_symbols_extended(top_n: int = Query(200, ge=1, le=500)):
    """Bitget USDT-M crypto perps with ticker columns for a sortable picker.

    Each row: `{symbol, last_price, change24h, volume_usdt}`. change24h is
    a fraction (e.g. -0.0028 = -0.28%); the frontend formats it. volume_usdt
    is the 24h quote-volume in USDT, matching the existing rank order.
    """
    global _extended_cache
    import time as _time
    if _extended_cache is not None:
        cached_ts, cached_rows = _extended_cache
        if _time.time() - cached_ts < _EXTENDED_CACHE_TTL:
            return cached_rows[:top_n]

    try:
        from ..data_service import _get_http_client, BITGET_PRODUCT_TYPE
        from ..market.bitget_client import BitgetPublicClient
        client = BitgetPublicClient(
            http_client=_get_http_client(),
            product_type=BITGET_PRODUCT_TYPE,
        )
        rows = await client.get_tickers()
    except Exception as exc:
        print(f"[symbols/extended] Bitget ticker fetch failed: {exc}")
        return []

    enriched: list[dict] = []
    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        if not _is_crypto_symbol(sym):
            continue
        try:
            vol = float(r.get("usdtVolume") or r.get("quoteVolume") or 0.0)
            last = float(r.get("lastPr") or r.get("markPrice") or 0.0)
            chg = float(r.get("change24h") or 0.0)
        except (TypeError, ValueError):
            continue
        if last <= 0:
            continue
        enriched.append({
            "symbol": sym,
            "last_price": last,
            "change24h": chg,
            "volume_usdt": vol,
        })

    enriched.sort(key=lambda x: x["volume_usdt"], reverse=True)
    _extended_cache = (_time.time(), enriched)
    return enriched[:top_n]


@router.get("/symbol-info")
async def get_symbol_info(symbol: str = Query(...)):
    """Return metadata for a specific symbol (e.g., price precision)."""
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    info = await get_symbol_metadata(symbol)
    if info:
        precision = info.get("pricePrecision")
        if precision is None:
            tick_sz = info.get("tickSz")
            if "." in str(tick_sz):
                precision = len(str(tick_sz).rstrip("0").split(".")[1])
            else:
                precision = 0
        return {
            "symbol": symbol,
            "instId": info.get("instId", symbol),
            "pricePrecision": precision,
            "tickSz": info.get("tickSz"),
        }

    return {"symbol": symbol, "pricePrecision": None}


@router.get("/chart")
async def api_chart(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(365, description="Days of data"),
    history_mode: str = Query("fast_window", description="fast_window | full_history"),
):
    """Return OHLCV + support/resistance lines in one response."""
    valid_intervals = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")
    if history_mode not in VALID_HISTORY_MODES:
        raise HTTPException(400, f"Invalid history_mode. Must be one of: {sorted(VALID_HISTORY_MODES)}")

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        df, ohlcv = await get_ohlcv_with_df(symbol, interval, end_time, days, history_mode=history_mode)
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
    history_mode: str = Query("fast_window", description="fast_window | full_history"),
    include_overlays: bool = Query(True, description="Server-computed MA/BB overlays (costs CPU + payload)"),
):
    """Return OHLCV data as JSON."""
    valid_intervals = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")
    if history_mode not in VALID_HISTORY_MODES:
        raise HTTPException(400, f"Invalid history_mode. Must be one of: {sorted(VALID_HISTORY_MODES)}")

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        result = await get_ohlcv(symbol, interval, end_time, days, history_mode=history_mode)
        # Drop overlays block from the response if client computes its own.
        # Saves 6 × n floats + field names off the wire. Cache key is the
        # same regardless — we just omit the field at serialization time.
        if not include_overlays and isinstance(result, dict):
            result = {k: v for k, v in result.items() if k != "overlays"}
        return result
    except ValueError as e:
        print(f"ValueError in /api/ohlcv for {symbol} {interval}: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        error_detail = f"Error fetching data for {symbol} {interval}: {e}\n{traceback.format_exc()}"
        print(f"Exception in /api/ohlcv: {error_detail}")
        raise HTTPException(500, f"Error fetching data: {str(e)}")


@router.get("/ohlcv/backfill")
async def api_ohlcv_backfill(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query(..., description="e.g. 5m, 1h, 4h"),
    before_ts: int = Query(..., description="Unix seconds — return bars CLOSING BEFORE this"),
    bars: int = Query(500, ge=50, le=1500, description="How many bars to fetch"),
):
    """Lazy-load older bars for a chart.

    Called when the user scrolls to the left edge of the chart. Returns
    `bars` candles whose `close_time` is strictly less than `before_ts`.
    Shape matches /api/ohlcv but without the overlays / metadata block —
    the client only cares about the candles to prepend.
    """
    valid_intervals = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    # Map bars × interval → days window with a safety margin so Bitget's
    # paginated history-candles yields enough rows even with weekend gaps.
    from ..data_service import _download_ohlcv_bitget, INTERVAL_MS
    interval_ms = INTERVAL_MS.get(interval, 60 * 60 * 1000)
    days_window = max(1, int(bars * interval_ms / (86400 * 1000)) + 2)
    end_ms = int(before_ts * 1000)

    try:
        df = await _download_ohlcv_bitget(symbol, interval, days=days_window, end_ms=end_ms)
    except Exception as e:
        raise HTTPException(500, f"backfill fetch failed: {e}")
    if df is None or df.is_empty():
        return {"candles": [], "volume": []}

    # Only keep bars strictly older than before_ts, trim to `bars` count
    import polars as pl
    df = df.filter(pl.col("open_time").cast(pl.Int64) // 1_000_000 < end_ms)
    df = df.tail(bars)

    candles = []
    volume = []
    for row in df.iter_rows(named=True):
        ts = int(row["open_time"].timestamp())
        o = float(row["open"]); h = float(row["high"])
        lo = float(row["low"]); c = float(row["close"])
        v = float(row["volume"])
        candles.append({"time": ts, "open": o, "high": h, "low": lo, "close": c})
        color = "#26a69a80" if c >= o else "#ef535080"
        volume.append({"time": ts, "value": v, "color": color})
    return {"candles": candles, "volume": volume, "count": len(candles)}


@router.get("/top-volume")
async def api_top_volume(n: int = Query(20, ge=1, le=50)):
    """Get top N symbols by 24h trading volume."""
    from ..data_service import get_top_volume_symbols
    symbols = await get_top_volume_symbols(n)
    return {"symbols": symbols, "count": len(symbols)}


# Live mark-price from Bitget public ticker. Bypasses all local caches so
# the UI always sees the same number Bitget sees. Meant to be polled every
# ~1 second from the frontend (no auth, so it's cheap).
_MARK_CACHE: dict[str, tuple[float, float]] = {}  # symbol -> (price, ts)

@router.get("/market/mark-price")
async def api_mark_price(symbol: str = Query(...)):
    import time as _time
    import httpx
    sym = symbol.upper().replace("/", "")
    now = _time.time()
    # Sub-second cache so burst polls don't hammer Bitget
    cached = _MARK_CACHE.get(sym)
    if cached and now - cached[1] < 0.8:
        return {"ok": True, "symbol": sym, "mark_price": cached[0], "ts": cached[1], "cached": True}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(
                "https://api.bitget.com/api/v2/mix/market/ticker",
                params={"symbol": sym, "productType": "USDT-FUTURES"},
            )
            j = r.json()
            data = j.get("data") or []
            row = data[0] if isinstance(data, list) and data else {}
            mp = float(row.get("markPrice") or row.get("lastPr") or 0.0)
            if mp <= 0:
                raise ValueError("no price in ticker response")
            _MARK_CACHE[sym] = (mp, now)
            return {
                "ok": True, "symbol": sym, "mark_price": mp, "ts": now,
                "last": float(row.get("lastPr") or 0) or None,
                "bid": float(row.get("bidPr") or 0) or None,
                "ask": float(row.get("askPr") or 0) or None,
                "cached": False,
            }
    except Exception as e:
        return {"ok": False, "symbol": sym, "error": str(e)}


@router.get("/market/snapshot")
async def api_market_snapshot(symbol: str = Query(...), interval: str = Query("4h")):
    """
    Lightweight summary for Workbench top strip + Decision Rail.
    Last price, change, volume, ATR, volatility regime, data freshness.
    """
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        df, _ = await get_ohlcv_with_df(symbol, interval, None, days=30)
    except Exception as e:
        return {"error": str(e), "symbol": symbol}

    if df is None or df.is_empty():
        return {"error": "no data", "symbol": symbol}

    close = df["close"].to_numpy().astype(float)
    high = df["high"].to_numpy().astype(float)
    low = df["low"].to_numpy().astype(float)
    vol = df["volume"].to_numpy().astype(float) if "volume" in df.columns else None

    last = float(close[-1]) if len(close) else 0
    prev = float(close[-2]) if len(close) >= 2 else last
    change_pct = ((last - prev) / prev * 100) if prev else 0

    # ATR-14
    from ..backtest_service import _atr
    import numpy as np
    atr_arr = _atr(high, low, close, 14)
    atr_val = float(atr_arr[-1]) if len(atr_arr) and not np.isnan(atr_arr[-1]) else 0
    atr_pct = (atr_val / last * 100) if last else 0

    if atr_pct < 1.5:
        regime = "low"
    elif atr_pct < 4.0:
        regime = "normal"
    else:
        regime = "high"

    # Volume 24h approx
    if vol is not None and len(vol):
        bars_per_day = {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(interval, 6)
        vol_24h = float(np.sum(vol[-bars_per_day:])) if len(vol) >= bars_per_day else float(np.sum(vol))
    else:
        vol_24h = 0

    # Data freshness
    try:
        last_ts = int(df["open_time"].to_list()[-1].timestamp())
        import time as _t
        freshness = int(_t.time() - last_ts)
    except Exception:
        freshness = None

    return {
        "symbol": symbol,
        "interval": interval,
        "last_price": round(last, 6),
        "change_pct": round(change_pct, 2),
        "volume_24h": round(vol_24h, 2),
        "atr": round(atr_val, 6),
        "atr_pct": round(atr_pct, 3),
        "volatility_regime": regime,
        "data_freshness_sec": freshness,
        "bars_count": len(close),
    }


@router.get("/market/structure-summary")
async def api_market_structure_summary(symbol: str = Query(...), interval: str = Query("4h")):
    """
    Decision Rail Market State card: trend, MA alignment, BB position,
    nearest support/resistance, ribbon score.
    """
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        df, _ = await get_ohlcv_with_df(
            symbol,
            interval,
            None,
            days=90,
            include_price_precision=False,
            include_render_payload=False,
        )
    except Exception as e:
        return {"error": str(e), "symbol": symbol}
    if df is None or df.is_empty():
        return {"error": "no data", "symbol": symbol}

    try:
        trimmed_df = _trim_structure_summary_df(df, interval)
        cache_key = _structure_summary_cache_key(trimmed_df, symbol, interval)
        cached = _get_cached_structure_summary(cache_key)
        if cached is not None:
            return cached
        response = await asyncio.to_thread(_build_structure_summary, trimmed_df, symbol, interval)
        return _store_cached_structure_summary(cache_key, response)
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def _trim_structure_summary_df(df, interval: str):
    max_bars = STRUCTURE_SUMMARY_LOOKBACK_BARS.get(interval, 540)
    if len(df) <= max_bars:
        return df
    return df.tail(max_bars)


def _structure_summary_cache_key(df, symbol: str, interval: str) -> tuple[object, ...]:
    return (
        symbol,
        interval,
        len(df),
        _recent_structure_signature(df),
    )


def _get_cached_structure_summary(cache_key: tuple[object, ...]) -> dict | None:
    cached = _structure_summary_cache.get(cache_key)
    if cached is None:
        return None
    _structure_summary_cache.move_to_end(cache_key)
    return cached


def _store_cached_structure_summary(cache_key: tuple[object, ...], response: dict) -> dict:
    _structure_summary_cache[cache_key] = response
    _structure_summary_cache.move_to_end(cache_key)
    while len(_structure_summary_cache) > STRUCTURE_SUMMARY_CACHE_LIMIT:
        _structure_summary_cache.popitem(last=False)
    return response


def _recent_structure_signature(df, *, window: int = 8) -> tuple[object, ...]:
    tail = df.tail(window)
    signature: list[object] = []
    for row in tail.iter_rows(named=True):
        signature.extend(
            (
                int(row["open_time"].timestamp()),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )
        )
    return tuple(signature)


def _build_structure_summary(df, symbol: str, interval: str) -> dict:
    import numpy as np
    from ..backtest_service import _sma, _ema, _atr, _bb_upper
    from ..pattern_service import get_patterns_from_df

    close = df["close"].to_numpy().astype(float)
    high = df["high"].to_numpy().astype(float)
    low = df["low"].to_numpy().astype(float)

    ma5 = _sma(close, 5)
    ma8 = _sma(close, 8)
    ema21 = _ema(close, 21)
    ma55 = _sma(close, 55)
    bb_up = _bb_upper(close, 21, 2.5)
    atr = _atr(high, low, close, 14)

    i = len(close) - 1
    price = float(close[i])

    # Trend based on slope of MA20
    ma20 = _sma(close, 20)
    if i >= 10 and not np.isnan(ma20[i]) and not np.isnan(ma20[i - 10]):
        slope_pct = (ma20[i] - ma20[i - 10]) / ma20[i - 10] * 100
    else:
        slope_pct = 0

    if slope_pct > 0.5: trend_label = "UPTREND"
    elif slope_pct < -0.5: trend_label = "DOWNTREND"
    else: trend_label = "SIDEWAYS"

    # MA alignment
    if not any(np.isnan([ma5[i], ma8[i], ema21[i], ma55[i]])):
        if price > ma5[i] > ma8[i] > ema21[i] > ma55[i]:
            ma_alignment = "BULL_ORDERED"
        elif price < ma5[i] < ma8[i] < ema21[i] < ma55[i]:
            ma_alignment = "BEAR_ORDERED"
        elif price > ma55[i]:
            ma_alignment = "ABOVE_MA55"
        else:
            ma_alignment = "BELOW_MA55"
    else:
        ma_alignment = "UNKNOWN"

    # Ribbon score (simple: count how many MAs below price for long)
    if not any(np.isnan([ma5[i], ma8[i], ema21[i], ma55[i]])):
        below = sum(1 for m in [ma5[i], ma8[i], ema21[i], ma55[i]] if price > m)
        ribbon_score = below  # 0-4
    else:
        ribbon_score = 0

    # BB position
    if not np.isnan(bb_up[i]):
        bb_dist_pct = (bb_up[i] - price) / price * 100
        bb_position = "AT_UPPER" if bb_dist_pct < 0.5 else "MID"
    else:
        bb_dist_pct = None
        bb_position = "UNKNOWN"

    # Nearest support/resistance via pattern_service
    try:
        patterns = get_patterns_from_df(df, symbol, interval)
        supports = patterns.get("supportLines", [])
        resists = patterns.get("resistanceLines", [])
    except Exception:
        supports = []
        resists = []

    def nearest(lines, direction):
        """direction 'below' for support, 'above' for resistance."""
        best = None
        best_dist = float("inf")
        for l in lines:
            y = l.get("v2") or l.get("y2") or l.get("value")
            if y is None: continue
            y = float(y)
            if direction == "below" and y < price:
                dist = price - y
            elif direction == "above" and y > price:
                dist = y - price
            else:
                continue
            if dist < best_dist:
                best_dist = dist
                best = y
        return best, best_dist if best else None

    nearest_support, sup_dist = nearest(supports, "below")
    nearest_resist, res_dist = nearest(resists, "above")

    return {
        "symbol": symbol,
        "interval": interval,
        "price": round(price, 6),
        "trend_label": trend_label,
        "trend_slope_pct": round(slope_pct, 3),
        "ma_alignment": ma_alignment,
        "ribbon_score": ribbon_score,
        "ribbon_max": 4,
        "bb_position": bb_position,
        "bb_distance_pct": round(bb_dist_pct, 3) if bb_dist_pct is not None else None,
        "nearest_support": round(nearest_support, 6) if nearest_support else None,
        "distance_to_support_pct": round((price - nearest_support) / price * 100, 2) if nearest_support else None,
        "nearest_resistance": round(nearest_resist, 6) if nearest_resist else None,
        "distance_to_resistance_pct": round((nearest_resist - price) / price * 100, 2) if nearest_resist else None,
    }


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
