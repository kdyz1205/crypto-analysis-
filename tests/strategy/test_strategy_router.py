from fastapi import FastAPI
from fastapi.testclient import TestClient
import polars as pl
from datetime import datetime, timedelta, timezone

import server.routers.strategy as strategy_router
from server.strategy.display_filter import build_display_line_meta
from server.strategy.state_machine import LineStateSnapshot
from server.strategy.types import Trendline


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


def _line(
    line_id: str,
    *,
    side: str = "resistance",
    state: str = "confirmed",
    score: float = 80.0,
    invalidation_index: int | None = None,
) -> Trendline:
    return Trendline(
        line_id=line_id,
        side=side,
        symbol="BTCUSDT",
        timeframe="1h",
        state=state,
        anchor_pivot_ids=("a", "b"),
        confirming_touch_pivot_ids=("a", "b", "c"),
        anchor_indices=(0, 2),
        anchor_prices=(1.05, 1.03),
        slope=-0.01 if side == "resistance" else 0.01,
        intercept=1.05 if side == "resistance" else 0.95,
        confirming_touch_indices=(0, 2, 3),
        bar_touch_indices=(0, 2, 3),
        confirming_touch_count=3,
        bar_touch_count=3,
        recent_bar_touch_count=0,
        residuals=(0.001, 0.001, 0.001),
        score=score,
        score_components={},
        projected_price_current=1.02 if side == "resistance" else 0.98,
        projected_price_next=1.01 if side == "resistance" else 0.99,
        latest_confirming_touch_index=3,
        latest_confirming_touch_price=1.02 if side == "resistance" else 0.98,
        bars_since_last_confirming_touch=0,
        recent_test_count=0,
        non_touch_cross_count=0,
        invalidation_reason="break_close_count" if invalidation_index is not None else None,
        invalidation_index=invalidation_index,
    )


def _line_state(line: Trendline) -> LineStateSnapshot:
    return LineStateSnapshot(
        line_id=line.line_id,
        state="invalidated" if line.invalidation_reason else line.state,
        previous_state="confirmed",
        side=line.side,
        symbol=line.symbol,
        timeframe=line.timeframe,
        line_score=line.score,
        confirming_touch_count=line.confirming_touch_count,
        bar_touch_count=line.bar_touch_count,
        projected_price_current=line.projected_price_current,
        projected_price_next=line.projected_price_next,
        latest_confirming_touch_index=line.latest_confirming_touch_index,
        bars_since_last_confirming_touch=line.bars_since_last_confirming_touch,
        invalidation_reason=line.invalidation_reason,
        transition_reason="test",
    )


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


def test_strategy_snapshot_cache_invalidates_when_manual_strategy_signature_changes(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    signatures = iter((("manual-a",), ("manual-b",)))
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
    monkeypatch.setattr(strategy_router, "manual_strategy_signature", lambda *args, **kwargs: next(signatures))
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


def test_serialize_touch_points_does_not_emit_bar_and_confirming_for_same_bar() -> None:
    candles_df = strategy_router._standardize_strategy_candles(_sample_polars_df())
    line = _line("line-a")
    display_meta = build_display_line_meta(candles_df, [line])

    touch_points = strategy_router._serialize_touch_points(
        [line],
        display_meta=display_meta,
        timestamps=candles_df["timestamp"].tolist(),
        highs=candles_df["high"].tolist(),
        lows=candles_df["low"].tolist(),
    )

    grouped: dict[tuple[str, int], list[str]] = {}
    for point in touch_points:
        grouped.setdefault((point.line_id, point.bar_index), []).append(point.touch_type)

    assert all(len(kinds) == 1 for kinds in grouped.values())


def test_serialize_invalidations_only_returns_displayworthy_markers() -> None:
    candles_df = strategy_router._standardize_strategy_candles(_sample_polars_df())
    lines = [
        _line("line-a", state="invalidated", score=90.0, invalidation_index=0),
        _line("line-b", state="invalidated", score=70.0, invalidation_index=10),
        _line("line-c", state="invalidated", score=50.0, invalidation_index=20),
    ]
    display_meta = build_display_line_meta(
        candles_df,
        lines,
        config=strategy_router.StrategyConfig(display_active_lines_per_side=2),
    )

    invalidations = strategy_router._serialize_invalidations(
        lines,
        [_line_state(line) for line in lines],
        display_meta=display_meta,
        timestamps=candles_df["timestamp"].tolist(),
    )

    assert [item.line_id for item in invalidations] == ["line-a", "line-b"]
    assert invalidations[0].display_class == "primary_invalidation"
    assert invalidations[1].display_class == "secondary_invalidation"
