from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.routers.drawings as drawings_router
import server.strategy.drawing_learner as drawing_learner
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
    monkeypatch.setattr(drawing_learner, "ML_DRAWINGS_FILE", tmp_path / "user_drawings_ml.jsonl")

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
            "line_width": 3.0,
            "label": "manual resistance",
            "override_mode": "display_only",
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()["drawing"]
    assert created["manual_line_id"].startswith("manual-BTCUSDT-4h-resistance")
    assert created["comparison_status"] == "uncompared"
    assert created["line_width"] == 3.0

    list_response = client.get("/api/drawings?symbol=BTCUSDT&timeframe=4h")
    assert list_response.status_code == 200
    assert len(list_response.json()["drawings"]) == 1

    update_response = client.patch(
        f"/api/drawings/{created['manual_line_id']}",
        json={
            "override_mode": "suppress_nearest_auto_line",
            "locked": True,
            "extend_left": True,
            "line_width": 5.0,
            "label": "desk override",
            "notes": "manual review line",
        },
    )
    assert update_response.status_code == 200
    updated = update_response.json()["drawing"]
    assert updated["override_mode"] == "suppress_nearest_auto_line"
    assert updated["locked"] is True
    assert updated["extend_left"] is True
    assert updated["line_width"] == 5.0
    assert updated["label"] == "desk override"
    assert updated["notes"] == "manual review line"

    delete_response = client.delete(f"/api/drawings/{created['manual_line_id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["removed"] == 1

    empty_response = client.get("/api/drawings?symbol=BTCUSDT&timeframe=4h")
    assert empty_response.status_code == 200
    assert empty_response.json()["drawings"] == []


def test_clear_drawings_removes_only_requested_symbol_timeframe(monkeypatch, tmp_path: Path) -> None:
    drawings_router.store = ManualTrendlineStore(tmp_path / "manual_trendlines.json")
    monkeypatch.setattr(drawing_learner, "ML_DRAWINGS_FILE", tmp_path / "user_drawings_ml.jsonl")
    client = TestClient(_build_app())

    rows = [
        ("HYPEUSDT", "15m", 1712592000, 1712678400, 100.0, 101.0),
        ("HYPEUSDT", "15m", 1712682000, 1712768400, 102.0, 103.0),
        ("HYPEUSDT", "4h", 1712592000, 1712678400, 104.0, 105.0),
    ]
    for symbol, timeframe, t1, t2, p1, p2 in rows:
        response = client.post(
            "/api/drawings",
            json={
                "symbol": symbol,
                "timeframe": timeframe,
                "side": "support",
                "t_start": t1,
                "t_end": t2,
                "price_start": p1,
                "price_end": p2,
                "label": "manual support",
                "override_mode": "display_only",
            },
        )
        assert response.status_code == 200

    clear_response = client.post("/api/drawings/clear?symbol=HYPEUSDT&timeframe=15m")
    assert clear_response.status_code == 200
    assert clear_response.json()["removed"] == 2

    h15 = client.get("/api/drawings?symbol=HYPEUSDT&timeframe=15m")
    h4 = client.get("/api/drawings?symbol=HYPEUSDT&timeframe=4h")
    assert h15.status_code == 200
    assert h4.status_code == 200
    assert h15.json()["drawings"] == []
    assert len(h4.json()["drawings"]) == 1
