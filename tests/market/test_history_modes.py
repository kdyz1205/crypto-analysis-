from datetime import datetime, timedelta, timezone

import polars as pl
import pytest
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

    metadata = data_service._history_metadata(
        df,
        window,
        "fast_window",
        exchange="bitget",
        data_source_mode="hybrid",
        data_source_kind="hybrid",
        requested_days=30,
        base_interval="1h",
        resampled_from_interval=None,
    )

    assert metadata["historyMode"] == "fast_window"
    assert metadata["exchange"] == "bitget"
    assert metadata["dataSourceMode"] == "hybrid"
    assert metadata["dataSourceKind"] == "hybrid"
    assert metadata["requestedDays"] == 30
    assert metadata["baseInterval"] == "1h"
    assert metadata["sourceBarCount"] == 10
    assert metadata["loadedBarCount"] == 3
    assert metadata["isTruncated"] is True
    assert metadata["truncationReason"] == "fast_window"
    assert metadata["listingStartTimestamp"] < metadata["latestLoadedTimestamp"]


def test_utc_epoch_seconds_treats_naive_exchange_times_as_utc() -> None:
    naive = datetime(2026, 4, 17, 8, 0)
    aware = datetime(2026, 4, 17, 8, 0, tzinfo=timezone.utc)

    assert data_service._utc_epoch_seconds(naive) == int(aware.timestamp())


@pytest.mark.asyncio
async def test_get_ohlcv_updates_csv_tail_after_one_elapsed_interval(monkeypatch) -> None:
    data_service._ohlcv_result_cache.clear()
    start = datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc)
    existing = _sample_df(rows=3).with_columns(
        (pl.lit(start) + pl.duration(hours=4) * pl.int_range(0, 3)).alias("open_time")
    )
    updated = _sample_df(rows=4).with_columns(
        (pl.lit(start) + pl.duration(hours=4) * pl.int_range(0, 4)).alias("open_time")
    )
    now = datetime(2026, 4, 17, 13, 1, tzinfo=timezone.utc).timestamp()
    calls = []

    async def fake_incremental_update(symbol, interval):
        calls.append((symbol, interval))
        return updated

    async def fake_price_precision(symbol):
        return None

    monkeypatch.setattr(data_service, "OFFLINE_ONLY", False)
    monkeypatch.setattr(data_service, "API_ONLY", False)
    monkeypatch.setattr(data_service, "EXCHANGE", "bitget")
    monkeypatch.setattr(data_service, "_find_csv", lambda symbol, interval: data_service.Path(f"{symbol.lower()}_{interval}.csv"))
    monkeypatch.setattr(data_service, "_load_csv", lambda path: existing)
    monkeypatch.setattr(data_service, "_incremental_update", fake_incremental_update)
    monkeypatch.setattr(data_service, "get_symbol_price_precision", fake_price_precision)
    monkeypatch.setattr(data_service.time, "time", lambda: now)

    result = await data_service.get_ohlcv("TESTUSDT", "4h", days=1, history_mode="fast_window")

    assert calls == [("TESTUSDT", "4h")]
    assert result["candles"][-1]["time"] == int(datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc).timestamp())
    assert result["latestLoadedTimestamp"] == result["candles"][-1]["time"]


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
