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
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)

    client = TestClient(_build_app())
    response = client.get("/api/strategy/config?symbol=BTCUSDT&interval=1h")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["pricePrecision"] == 4
    assert payload["layerDefaults"]["primaryTrendlines"] is True
    assert payload["layerDefaults"]["debugTrendlines"] is False
    assert payload["tickSize"] == 0.0001


def test_strategy_snapshot_route_returns_frontend_ready_fields(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
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
    assert snapshot["candidate_lines"] == [] or "display_class" in snapshot["candidate_lines"][0]


def test_strategy_replay_route_supports_tail(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)

    client = TestClient(_build_app())
    response = client.get("/api/strategy/replay?symbol=BTCUSDT&interval=1h&analysis_bars=120&tail=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshotCount"] == 2
    assert len(payload["snapshots"]) == 2


def test_strategy_snapshot_route_offloads_heavy_work(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    offload_calls = []

    async def fake_to_thread(func, *args, **kwargs):
        offload_calls.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(strategy_router.asyncio, "to_thread", fake_to_thread)
    strategy_router._snapshot_cache.clear()

    client = TestClient(_build_app())
    response = client.get("/api/strategy/snapshot?symbol=BTCUSDT&interval=1h&analysis_bars=120")

    assert response.status_code == 200
    assert offload_calls == ["_build_strategy_snapshot_response"]


def test_strategy_snapshot_route_reuses_cached_response(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    call_count = {"count": 0}

    def fake_build_snapshot(*args, **kwargs):
        call_count["count"] += 1
        return strategy_router.StrategySnapshotResponse(
            symbol="BTCUSDT",
            interval="1h",
            barCount=4,
            analysisBarCount=4,
            pricePrecision=4,
            tickSize=0.0001,
            config={},
            snapshot=strategy_router.StrategySnapshotModel(
                bar_index=3,
                timestamp=1,
                pivots=[],
                candidate_lines=[],
                active_lines=[],
                line_states=[],
                touch_points=[],
                signals=[],
                signal_states=[],
                invalidations=[],
                orders=[],
            ),
        )

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(strategy_router, "_build_strategy_snapshot_response", fake_build_snapshot)
    strategy_router._snapshot_cache.clear()

    client = TestClient(_build_app())
    first = client.get("/api/strategy/snapshot?symbol=BTCUSDT&interval=1h&analysis_bars=120")
    second = client.get("/api/strategy/snapshot?symbol=BTCUSDT&interval=1h&analysis_bars=120")

    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count["count"] == 1


def test_strategy_snapshot_cache_invalidates_when_recent_history_changes(monkeypatch) -> None:
    df_one = _sample_polars_df()
    df_two = _sample_polars_df().with_columns(
        pl.when(pl.arange(0, pl.len()) == 2)
        .then(pl.lit(1.99))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    responses = iter(((df_one, {"pricePrecision": 4}), (df_two, {"pricePrecision": 4})))

    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return next(responses)

    call_count = {"count": 0}

    def fake_build_snapshot(*args, **kwargs):
        call_count["count"] += 1
        return strategy_router.StrategySnapshotResponse(
            symbol="BTCUSDT",
            interval="1h",
            barCount=4,
            analysisBarCount=4,
            pricePrecision=4,
            tickSize=0.0001,
            config={},
            snapshot=strategy_router.StrategySnapshotModel(
                bar_index=3,
                timestamp=1,
                pivots=[],
                candidate_lines=[],
                active_lines=[],
                line_states=[],
                touch_points=[],
                signals=[],
                signal_states=[],
                invalidations=[],
                orders=[],
            ),
        )

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(strategy_router, "_build_strategy_snapshot_response", fake_build_snapshot)
    strategy_router._snapshot_cache.clear()

    client = TestClient(_build_app())
    first = client.get("/api/strategy/snapshot?symbol=BTCUSDT&interval=1h&analysis_bars=120")
    second = client.get("/api/strategy/snapshot?symbol=BTCUSDT&interval=1h&analysis_bars=120")

    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count["count"] == 2


def test_strategy_replay_route_reuses_cached_tail_response(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    call_count = {"count": 0}

    def fake_build_replay(*args, **kwargs):
        call_count["count"] += 1
        return strategy_router.StrategyReplayResponse(
            symbol="BTCUSDT",
            interval="1h",
            barCount=4,
            analysisBarCount=4,
            snapshotCount=1,
            pricePrecision=4,
            tickSize=0.0001,
            config={},
            snapshots=[
                strategy_router.StrategySnapshotModel(
                    bar_index=3,
                    timestamp=1,
                    pivots=[],
                    candidate_lines=[],
                    active_lines=[],
                    line_states=[],
                    touch_points=[],
                    signals=[],
                    signal_states=[],
                    invalidations=[],
                    orders=[],
                )
            ],
        )

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(strategy_router, "_build_strategy_replay_response", fake_build_replay)
    strategy_router._replay_cache.clear()

    client = TestClient(_build_app())
    first = client.get("/api/strategy/replay?symbol=BTCUSDT&interval=1h&analysis_bars=120&tail=2")
    second = client.get("/api/strategy/replay?symbol=BTCUSDT&interval=1h&analysis_bars=120&tail=2")

    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count["count"] == 1


def test_strategy_replay_route_uses_fast_tail_builder(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    calls = {"tail": 0}

    def fake_build_tail_snapshots(candles_df, cfg, **kwargs):
        calls["tail"] += 1
        snapshot = strategy_router.build_latest_snapshot(candles_df, cfg, symbol="BTCUSDT", timeframe="1h")
        return (snapshot,)

    def fail_replay_strategy(*args, **kwargs):
        raise AssertionError("full replay should not run for small tail requests")

    monkeypatch.setattr(strategy_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(strategy_router, "build_tail_snapshots", fake_build_tail_snapshots)
    monkeypatch.setattr(strategy_router, "replay_strategy", fail_replay_strategy)
    strategy_router._replay_cache.clear()

    client = TestClient(_build_app())
    response = client.get("/api/strategy/replay?symbol=BTCUSDT&interval=1h&analysis_bars=120&tail=1")

    assert response.status_code == 200
    assert response.json()["snapshotCount"] == 1
    assert calls["tail"] == 1
