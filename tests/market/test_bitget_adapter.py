from server.market.bitget_adapter import (
    bitget_candles_to_records,
    bitget_contracts_to_symbol_map,
    bitget_tickers_to_volume_symbols,
)


def test_bitget_contracts_to_symbol_map_normalizes_precision():
    rows = [
        {
            "symbol": "BTCUSDT",
            "pricePlace": "1",
            "priceEndStep": "1",
            "volumePlace": "3",
            "sizeMultiplier": "0.001",
            "minTradeNum": "0.001",
            "symbolStatus": "normal",
            "productType": "usdt-futures",
        }
    ]

    result = bitget_contracts_to_symbol_map(rows)

    assert result["BTCUSDT"]["instId"] == "BTCUSDT"
    assert result["BTCUSDT"]["tickSz"] == "0.1"
    assert result["BTCUSDT"]["pricePrecision"] == 1
    assert result["BTCUSDT"]["lotSz"] == "0.001"


def test_bitget_tickers_to_volume_symbols_sorts_by_usdt_volume():
    rows = [
        {"symbol": "ETHUSDT", "usdtVolume": "1000"},
        {"symbol": "BTCUSDT", "usdtVolume": "5000"},
        {"symbol": "DOGEUSDT", "usdtVolume": "250"},
    ]

    assert bitget_tickers_to_volume_symbols(rows, top_n=2) == ["BTCUSDT", "ETHUSDT"]


def test_bitget_candles_to_records_match_shared_csv_shape():
    rows = [
        ["1710000000000", "100", "110", "95", "105", "12.5", "1300.0"],
    ]

    records = bitget_candles_to_records(rows)

    assert len(records) == 1
    assert records[0][0] == 1710000000000
    assert records[0][1:6] == [100.0, 110.0, 95.0, 105.0, 12.5]
    assert records[0][7] == 1300.0
    assert len(records[0]) == 12
