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
