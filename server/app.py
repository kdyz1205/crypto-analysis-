from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .data_service import load_symbols, get_ohlcv
from .pattern_service import get_patterns

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Crypto TA")

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")


@app.get("/")
async def index():
    return FileResponse(str(PROJECT_ROOT / "frontend" / "index.html"))


@app.get("/api/symbols")
async def get_symbols():
    """Return all available ticker symbols, sorted alphabetically."""
    return load_symbols()


@app.get("/api/ohlcv")
async def api_ohlcv(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(30, description="Days of data to fetch"),
):
    """Return OHLCV data as JSON. Auto-downloads from Binance if missing."""
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        return await get_ohlcv(symbol, interval, end_time, days)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error fetching data: {e}")


@app.get("/api/patterns")
async def api_patterns(
    symbol: str = Query(..., description="e.g. HYPEUSDT"),
    interval: str = Query("1h", description="e.g. 5m, 15m, 1h, 4h, 1d"),
    end_time: str | None = Query(None, description="Replay end time, ISO format"),
    days: int = Query(30, description="Days of data to fetch"),
):
    """Run SR pattern detection and return trendlines, zones, and trend info."""
    valid_intervals = {"5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval. Must be one of: {valid_intervals}")

    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    try:
        return get_patterns(symbol, interval, end_time, days)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pattern detection error: {e}")
