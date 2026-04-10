from datetime import datetime, timedelta, timezone

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.routers.paper_execution as paper_execution_router
from server.strategy.replay import ReplayResult, ReplaySnapshot
from server.strategy.types import StrategySignal


def _sample_polars_df() -> pl.DataFrame:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "open_time": [start + timedelta(hours=i) for i in range(4)],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))


def _signal() -> StrategySignal:
    return StrategySignal(
        signal_id="sig-router",
        line_id="line-router",
        symbol="BTCUSDT",
        timeframe="1h",
        signal_type="PRE_LIMIT_SHORT",
        direction="short",
        trigger_mode="pre_limit",
        timestamp=1,
        trigger_bar_index=1,
        score=0.75,
        priority_rank=1,
        entry_price=102.0,
        stop_price=104.0,
        tp_price=98.0,
        risk_reward=2.0,
        confirming_touch_count=3,
        bars_since_last_confirming_touch=1,
        distance_to_line=0.1,
        line_side="resistance",
        reason_code="test",
        factor_components={},
    )


def _replay_result() -> ReplayResult:
    snapshots = []
    for bar_index in range(4):
        snapshots.append(
            ReplaySnapshot(
                bar_index=bar_index,
                timestamp=bar_index + 1,
                pivots=(),
                candidate_lines=(),
                active_lines=(),
                line_states=(),
                signals=(_signal(),) if bar_index == 0 else (),
                signal_states=(),
                invalidations=(),
            )
        )
    return ReplayResult(symbol="BTCUSDT", timeframe="1h", snapshots=tuple(snapshots))


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(paper_execution_router.router)
    return app


def test_paper_execution_router_state_reset_step_and_config(monkeypatch) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 2}

    def fake_build_latest_snapshot(candles_df, strategy_cfg, *, symbol="", timeframe="", **kwargs):
        bar_index = len(candles_df) - 1
        return ReplaySnapshot(
            bar_index=bar_index,
            timestamp=bar_index + 1,
            pivots=(),
            candidate_lines=(),
            active_lines=(),
            line_states=(),
            signals=(_signal(),) if bar_index == 0 else (),
            signal_states=(),
            invalidations=(),
        )

    monkeypatch.setattr(paper_execution_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    monkeypatch.setattr(paper_execution_router, "build_latest_snapshot", fake_build_latest_snapshot)

    paper_execution_router.paper_engine.reset()
    client = TestClient(_build_app())

    state_response = client.get("/api/paper-execution/state")
    assert state_response.status_code == 200
    assert state_response.json()["state"]["account"]["equity"] == 10000.0

    config_get = client.get("/api/paper-execution/config")
    assert config_get.status_code == 200
    assert config_get.json()["config"]["risk_per_trade"] == 0.003

    config_post = client.post(
        "/api/paper-execution/config",
        json={"risk_per_trade": 0.005, "cancel_after_bars": 5},
    )
    assert config_post.status_code == 200
    assert config_post.json()["config"]["risk_per_trade"] == 0.005
    assert config_post.json()["config"]["cancel_after_bars"] == 5

    step_response = client.post(
        "/api/paper-execution/step",
        json={"symbol": "BTCUSDT", "interval": "1h", "bar_index": 0, "analysis_bars": 120},
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["stream"] == "BTCUSDT:1h"
    assert step_payload["lastProcessedBar"] == 0
    assert step_payload["state"]["account"]["open_order_count"] == 1

    reset_response = client.post("/api/paper-execution/reset", json={"starting_equity": 12000})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["state"]["account"]["equity"] == 12000.0
    assert reset_payload["state"]["account"]["open_order_count"] == 0
