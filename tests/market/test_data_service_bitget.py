import pytest
import polars as pl
import asyncio
from datetime import datetime, timedelta, timezone

from server import data_service


@pytest.mark.asyncio
async def test_load_swap_symbols_uses_bitget_loader(monkeypatch):
    async def fake_load_bitget():
        return {"BTCUSDT": {"tickSz": "0.1", "pricePrecision": 1}}

    monkeypatch.setattr(data_service, "EXCHANGE", "bitget")
    monkeypatch.setattr(data_service, "load_bitget_swap_symbols", fake_load_bitget)

    result = await data_service.load_swap_symbols()

    assert result["BTCUSDT"]["pricePrecision"] == 1


@pytest.mark.asyncio
async def test_get_symbol_price_precision_prefers_normalized_metadata(monkeypatch):
    async def fake_load_swap_symbols():
        return {"BTCUSDT": {"tickSz": "0.01", "pricePrecision": 2}}

    monkeypatch.setattr(data_service, "load_swap_symbols", fake_load_swap_symbols)

    precision = await data_service.get_symbol_price_precision("BTCUSDT")

    assert precision == 2


@pytest.mark.asyncio
async def test_download_ohlcv_reuses_larger_inflight_request(monkeypatch):
    calls = {"count": 0}

    async def fake_download(symbol, interval, days):
        calls["count"] += 1
        await asyncio.sleep(0.05)
        now = datetime.now(timezone.utc)
        return pl.DataFrame(
            {
                "open_time": [now - timedelta(hours=i) for i in range(10, 0, -1)],
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.0] * 10,
                "volume": [100.0] * 10,
            }
        )

    monkeypatch.setattr(data_service, "EXCHANGE", "bitget")
    monkeypatch.setattr(data_service, "_download_ohlcv_bitget", fake_download)
    data_service._ohlcv_cache.clear()
    data_service._ohlcv_inflight.clear()

    larger = asyncio.create_task(data_service.download_ohlcv("BTCUSDT", "4h", days=365))
    await asyncio.sleep(0.01)
    smaller = asyncio.create_task(data_service.download_ohlcv("BTCUSDT", "4h", days=90))

    first, second = await asyncio.gather(larger, smaller)

    assert calls["count"] == 1
    assert len(first) == len(second) == 10
