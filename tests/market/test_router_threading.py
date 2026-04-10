from datetime import datetime, timedelta, timezone

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.routers.market as market_router
import server.routers.patterns as patterns_router


def _sample_polars_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100.0],
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))


def test_market_structure_summary_offloads_heavy_work(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {}

    offload_calls = []

    async def fake_to_thread(func, *args, **kwargs):
        offload_calls.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(market_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(
        market_router,
        "_build_structure_summary",
        lambda df, symbol, interval: {"symbol": symbol, "interval": interval, "trend_label": "SIDEWAYS"},
    )
    monkeypatch.setattr(market_router.asyncio, "to_thread", fake_to_thread)

    app = FastAPI()
    app.include_router(market_router.router)
    client = TestClient(app)

    response = client.get("/api/market/structure-summary?symbol=BTCUSDT&interval=1h")

    assert response.status_code == 200
    assert response.json()["trend_label"] == "SIDEWAYS"
    assert offload_calls == ["<lambda>"]


def test_market_structure_summary_trims_and_caches(monkeypatch) -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = 800
    big_df = pl.DataFrame(
        {
            "open_time": [start + timedelta(hours=4 * i) for i in range(rows)],
            "open": [1.0 + (i * 0.001) for i in range(rows)],
            "high": [1.1 + (i * 0.001) for i in range(rows)],
            "low": [0.9 + (i * 0.001) for i in range(rows)],
            "close": [1.0 + (i * 0.001) for i in range(rows)],
            "volume": [100.0] * rows,
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))

    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return big_df, {}

    build_lengths = []

    def fake_build_structure_summary(df, symbol, interval):
        build_lengths.append(len(df))
        return {"symbol": symbol, "interval": interval, "trend_label": "SIDEWAYS", "bars": len(df)}

    monkeypatch.setattr(market_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(market_router, "_build_structure_summary", fake_build_structure_summary)
    market_router._structure_summary_cache.clear()

    app = FastAPI()
    app.include_router(market_router.router)
    client = TestClient(app)

    first = client.get("/api/market/structure-summary?symbol=BTCUSDT&interval=4h")
    second = client.get("/api/market/structure-summary?symbol=BTCUSDT&interval=4h")

    assert first.status_code == 200
    assert second.status_code == 200
    assert build_lengths == [market_router.STRUCTURE_SUMMARY_LOOKBACK_BARS["4h"]]
    assert first.json()["bars"] == market_router.STRUCTURE_SUMMARY_LOOKBACK_BARS["4h"]


def test_market_structure_summary_cache_invalidates_when_recent_history_changes(monkeypatch) -> None:
    base_df = pl.DataFrame(
        {
            "open_time": [
                datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 4, 1, 4, 0, tzinfo=timezone.utc),
                datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc),
            ],
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.0, 1.1, 1.2],
            "volume": [100.0, 100.0, 100.0],
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))
    revised_df = base_df.with_columns(pl.Series("close", [1.0, 9.9, 1.2]))
    responses = iter(((base_df, {}), (revised_df, {})))

    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return next(responses)

    call_count = {"count": 0}

    def fake_build_structure_summary(df, symbol, interval):
        call_count["count"] += 1
        return {"symbol": symbol, "interval": interval, "trend_label": "SIDEWAYS"}

    monkeypatch.setattr(market_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(market_router, "_build_structure_summary", fake_build_structure_summary)
    market_router._structure_summary_cache.clear()

    app = FastAPI()
    app.include_router(market_router.router)
    client = TestClient(app)

    first = client.get("/api/market/structure-summary?symbol=BTCUSDT&interval=1h")
    second = client.get("/api/market/structure-summary?symbol=BTCUSDT&interval=1h")

    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count["count"] == 2


def test_patterns_route_offloads_detection(monkeypatch) -> None:
    def fake_get_patterns(symbol, interval, end_time=None, days=30, mode="full"):
        return {
            "supportLines": [],
            "resistanceLines": [],
            "consolidationZones": [],
            "trendSegments": [],
            "patterns": [],
            "currentTrend": 0,
            "trendLabel": "SIDEWAYS",
            "trendSlope": 0.0,
        }

    offloaded = []

    async def fake_to_thread(func, *args, **kwargs):
        offloaded.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(patterns_router, "API_ONLY", False)
    monkeypatch.setattr(patterns_router, "get_patterns", fake_get_patterns)
    monkeypatch.setattr(patterns_router.asyncio, "to_thread", fake_to_thread)

    app = FastAPI()
    app.include_router(patterns_router.router)
    client = TestClient(app)

    response = client.get("/api/patterns?symbol=BTCUSDT&interval=1h&mode=full")

    assert response.status_code == 200
    assert response.json()["trendLabel"] == "SIDEWAYS"
    assert len(offloaded) == 1
    assert getattr(offloaded[0], "func", None) is fake_get_patterns
