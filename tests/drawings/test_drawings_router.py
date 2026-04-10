from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.routers.drawings as drawings_router
from server.drawings.store import ManualTrendlineStore


def _sample_polars_df() -> pl.DataFrame:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "open_time": [start + timedelta(hours=i) for i in range(40)],
            "open": [1.0 + (0.01 * i) for i in range(40)],
            "high": [1.05 + (0.01 * i) for i in range(40)],
            "low": [0.95 + (0.01 * i) for i in range(40)],
            "close": [1.01 + (0.01 * i) for i in range(40)],
            "volume": [100.0 + i for i in range(40)],
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(drawings_router.router)
    return app


def test_manual_drawing_crud(monkeypatch, tmp_path: Path) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    monkeypatch.setattr(drawings_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    drawings_router.store = ManualTrendlineStore(tmp_path / "manual_trendlines.json")

    client = TestClient(_build_app())
    create_response = client.post(
        "/api/drawings",
        json={
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "side": "resistance",
            "t_start": 1712592000,
            "t_end": 1712678400,
            "price_start": 100.0,
            "price_end": 105.0,
            "label": "manual resistance",
            "override_mode": "display_only",
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()["drawing"]
    assert created["manual_line_id"].startswith("manual-BTCUSDT-4h-resistance")
    assert created["comparison_status"] in {"supports_auto", "near_auto", "conflicts_auto", "no_nearby_auto"}

    list_response = client.get("/api/drawings?symbol=BTCUSDT&timeframe=4h")
    assert list_response.status_code == 200
    assert len(list_response.json()["drawings"]) == 1

    update_response = client.patch(
        f"/api/drawings/{created['manual_line_id']}",
        json={"override_mode": "suppress_nearest_auto_line", "locked": True},
    )
    assert update_response.status_code == 200
    updated = update_response.json()["drawing"]
    assert updated["override_mode"] == "suppress_nearest_auto_line"
    assert updated["locked"] is True

    delete_response = client.delete(f"/api/drawings/{created['manual_line_id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["removed"] == 1

    empty_response = client.get("/api/drawings?symbol=BTCUSDT&timeframe=4h")
    assert empty_response.status_code == 200
    assert empty_response.json()["drawings"] == []
