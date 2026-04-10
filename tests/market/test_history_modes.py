from datetime import datetime, timedelta, timezone

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.data_service as data_service
import server.routers.market as market_router


def _sample_df(rows: int = 5) -> pl.DataFrame:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "open_time": [start + timedelta(hours=4 * idx) for idx in range(rows)],
            "open": [1.0 + idx for idx in range(rows)],
            "high": [1.1 + idx for idx in range(rows)],
            "low": [0.9 + idx for idx in range(rows)],
            "close": [1.0 + idx for idx in range(rows)],
            "volume": [100.0] * rows,
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))


def test_apply_history_mode_trims_fast_window() -> None:
    df = _sample_df(rows=10)
    trimmed = data_service._apply_history_mode(df, "4h", days=1, history_mode="fast_window")

    assert len(trimmed) < len(df)
    assert trimmed["open_time"].min() >= df["open_time"].max() - timedelta(days=1)


def test_apply_history_mode_full_history_preserves_rows() -> None:
    df = _sample_df(rows=10)

    preserved = data_service._apply_history_mode(df, "4h", days=1, history_mode="full_history")

    assert len(preserved) == len(df)


def test_history_metadata_marks_truncation() -> None:
    df = _sample_df(rows=10)
    window = df.tail(3)

    metadata = data_service._history_metadata(df, window, "fast_window")

    assert metadata["historyMode"] == "fast_window"
    assert metadata["loadedBarCount"] == 3
    assert metadata["isTruncated"] is True
    assert metadata["truncationReason"] == "fast_window"
    assert metadata["listingStartTimestamp"] < metadata["latestLoadedTimestamp"]


def test_api_ohlcv_forwards_history_mode(monkeypatch) -> None:
    observed = {}

    async def fake_get_ohlcv(symbol, interval, end_time=None, days=30, history_mode="fast_window"):
        observed["history_mode"] = history_mode
        return {
            "candles": [],
            "volume": [],
            "historyMode": history_mode,
            "loadedBarCount": 0,
            "isFullHistory": history_mode == "full_history",
            "isTruncated": False,
        }

    monkeypatch.setattr(market_router, "get_ohlcv", fake_get_ohlcv)

    app = FastAPI()
    app.include_router(market_router.router)
    client = TestClient(app)

    response = client.get("/api/ohlcv?symbol=BTCUSDT&interval=4h&history_mode=full_history")

    assert response.status_code == 200
    assert observed["history_mode"] == "full_history"
    assert response.json()["historyMode"] == "full_history"


def test_api_chart_rejects_invalid_history_mode() -> None:
    app = FastAPI()
    app.include_router(market_router.router)
    client = TestClient(app)

    response = client.get("/api/chart?symbol=BTCUSDT&interval=4h&history_mode=oops")

    assert response.status_code == 400
    assert "history_mode" in response.json()["detail"]
