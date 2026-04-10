from datetime import datetime, timedelta, timezone

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.runtime.manager as runtime_manager_module
import server.routers.runtime as runtime_router
from server.execution.live_engine import LiveExecutionEngine
from server.runtime.manager import SubaccountRuntimeManager


class _FakeAdapter:
    exchange_name = "bitget"

    def has_api_keys(self) -> bool:
        return False

    async def reconcile_live_state(self, mode: str) -> dict:
        return {"ok": False, "blocked": True, "mode": mode, "reason": "api_keys_missing", "positions": [], "pending_orders": []}

    async def submit_live_entry(self, intent, mode: str) -> dict:
        return {"ok": False, "reason": "api_keys_missing", "exchange_order_id": "", "submitted_price": intent.entry_price, "submitted_notional": intent.entry_price * intent.quantity}

    async def submit_live_close(self, symbol: str, mode: str) -> dict:
        return {"ok": False, "reason": "api_keys_missing", "exchange_order_id": "", "symbol": symbol, "mode": mode}


def _sample_polars_df(rows: int = 30) -> pl.DataFrame:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "open_time": [start + timedelta(hours=4 * idx) for idx in range(rows)],
            "open": [100.0 + idx for idx in range(rows)],
            "high": [101.0 + idx for idx in range(rows)],
            "low": [99.0 + idx for idx in range(rows)],
            "close": [100.5 + idx for idx in range(rows)],
            "volume": [1000.0] * rows,
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))


def test_runtime_router_create_tick_and_stop(monkeypatch, tmp_path) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 2}

    monkeypatch.setattr(runtime_manager_module, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    manager = SubaccountRuntimeManager(
        store_path=tmp_path / "runtime.json",
        event_log_path=tmp_path / "runtime.jsonl",
        adapter_provider=lambda: _FakeAdapter(),
    )
    runtime_router.runtime_manager = manager

    app = FastAPI()
    app.include_router(runtime_router.router)
    client = TestClient(app)

    created = client.post(
        "/api/runtime/instances",
        json={
            "label": "BTC demo",
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "live_mode": "disabled",
            "strategy_config": {
                "enabled_trigger_modes": ["pre_limit"],
                "lookback_bars": 80,
                "window_bars": 100,
            },
        },
    )
    assert created.status_code == 200
    instance_id = created.json()["instance"]["config"]["instance_id"]
    assert created.json()["instance"]["config"]["strategy_config"]["enabled_trigger_modes"] == ["pre_limit"]
    assert created.json()["instance"]["config"]["strategy_config"]["lookback_bars"] == 80

    ticked = client.post(f"/api/runtime/instances/{instance_id}/tick", json={})
    assert ticked.status_code == 200
    assert ticked.json()["instance"]["status"]["paper_state"]["account"]["last_processed_bar_by_stream"]["BTCUSDT:4h"] == 0

    started = client.post(f"/api/runtime/instances/{instance_id}/start")
    assert started.status_code == 200
    assert started.json()["instance"]["status"]["runtime_state"] == "running"

    stopped = client.post(f"/api/runtime/instances/{instance_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["instance"]["status"]["runtime_state"] == "stopped"

    events = client.get("/api/runtime/events?limit=10")
    assert events.status_code == 200
    assert len(events.json()["events"]) >= 3


def test_runtime_manager_restores_persisted_state(monkeypatch, tmp_path) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 2}

    monkeypatch.setattr(runtime_manager_module, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    store_path = tmp_path / "runtime.json"
    event_log_path = tmp_path / "runtime.jsonl"

    manager = SubaccountRuntimeManager(
        store_path=store_path,
        event_log_path=event_log_path,
        adapter_provider=lambda: _FakeAdapter(),
    )
    record = manager.create_instance(label="ETH paper", symbol="ETHUSDT", timeframe="4h")
    instance_id = record.config.instance_id

    import asyncio
    asyncio.run(manager.tick_instance(instance_id))

    restored = SubaccountRuntimeManager(
        store_path=store_path,
        event_log_path=event_log_path,
        adapter_provider=lambda: _FakeAdapter(),
    )
    asyncio.run(restored.startup())
    restored_record = restored.get_instance(instance_id)

    assert restored_record.status.paper_state is not None
    assert restored_record.status.paper_state.account.last_processed_bar_by_stream["ETHUSDT:4h"] == 0
    assert restored_record.status.last_processed_bar == 0
    assert restored_record.config.strategy_config.enabled_trigger_modes == ("pre_limit",)
