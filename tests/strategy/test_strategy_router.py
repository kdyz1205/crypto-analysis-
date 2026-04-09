from fastapi import FastAPI
from fastapi.testclient import TestClient
import polars as pl
from datetime import datetime, timedelta, timezone

import server.routers.strategy as strategy_router


def _sample_polars_df() -> pl.DataFrame:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "open_time": [start + timedelta(hours=i) for i in range(4)],
            "open": [1.0, 1.01, 1.02, 1.03],
            "high": [1.05, 1.06, 1.07, 1.08],
            "low": [0.95, 0.96, 0.97, 0.98],
            "close": [1.01, 1.02, 1.03, 1.04],
            "volume": [100.0, 110.0, 120.0, 130.0],
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(strategy_router.router)
    return app


def test_strategy_config_route_returns_layer_defaults(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days):
        return _sample_polars_df(), {"pricePrecision": 4}

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)

    client = TestClient(_build_app())
    response = client.get("/api/strategy/config?symbol=BTCUSDT&interval=1h")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["pricePrecision"] == 4
    assert payload["layerDefaults"]["trendlines"] is True
    assert payload["tickSize"] == 0.0001


def test_strategy_snapshot_route_returns_frontend_ready_fields(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days):
        return _sample_polars_df(), {"pricePrecision": 4}

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)

    client = TestClient(_build_app())
    response = client.get("/api/strategy/snapshot?symbol=BTCUSDT&interval=1h&analysis_bars=120")

    assert response.status_code == 200
    payload = response.json()
    snapshot = payload["snapshot"]
    assert payload["symbol"] == "BTCUSDT"
    assert payload["analysisBarCount"] == 4
    assert "candidate_lines" in snapshot
    assert "active_lines" in snapshot
    assert "line_states" in snapshot
    assert "touch_points" in snapshot
    assert "signals" in snapshot
    assert "signal_states" in snapshot
    assert "invalidations" in snapshot


def test_strategy_replay_route_supports_tail(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days):
        return _sample_polars_df(), {"pricePrecision": 4}

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)

    client = TestClient(_build_app())
    response = client.get("/api/strategy/replay?symbol=BTCUSDT&interval=1h&analysis_bars=120&tail=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshotCount"] == 2
    assert len(payload["snapshots"]) == 2


def test_strategy_snapshot_route_offloads_heavy_work(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days):
        return _sample_polars_df(), {"pricePrecision": 4}

    offload_calls = []

    async def fake_to_thread(func, *args, **kwargs):
        offload_calls.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(strategy_router.asyncio, "to_thread", fake_to_thread)

    client = TestClient(_build_app())
    response = client.get("/api/strategy/snapshot?symbol=BTCUSDT&interval=1h&analysis_bars=120")

    assert response.status_code == 200
    assert offload_calls == ["_build_strategy_snapshot_response"]
