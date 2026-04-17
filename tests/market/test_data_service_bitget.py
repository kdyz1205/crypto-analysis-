from datetime import datetime, timezone

import pytest

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
async def test_bitget_incremental_update_includes_live_current_bar(monkeypatch):
    start_ms = int(datetime(2026, 4, 17, 8, 0, tzinfo=timezone.utc).timestamp() * 1000)
    live_ms = int(datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    now = datetime(2026, 4, 17, 13, 1, tzinfo=timezone.utc).timestamp()
    calls = []

    class FakeBitgetClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get_candles(self, symbol, granularity, *, start_time=None, end_time=None, limit=200, history=True):
            calls.append(history)
            if history:
                return [[str(start_ms), "43.8", "44.4", "43.2", "43.6", "100", "4300"]]
            return [
                [str(start_ms), "43.8", "44.5", "43.2", "43.7", "120", "5200"],
                [str(live_ms), "43.7", "44.7", "43.5", "44.4", "80", "3500"],
            ]

    monkeypatch.setattr(data_service, "BitgetPublicClient", FakeBitgetClient)
    monkeypatch.setattr(data_service.time, "time", lambda: now)

    df = await data_service._fetch_candles_since_bitget("HYPEUSDT", "4h", start_ms=start_ms)

    assert calls == [True, False]
    assert df["open_time"].max() == datetime(2026, 4, 17, 12, 0)
    assert float(df.filter(data_service.pl.col("open_time") == datetime(2026, 4, 17, 12, 0))["close"][0]) == 44.4
