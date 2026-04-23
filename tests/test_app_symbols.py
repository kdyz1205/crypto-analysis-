from fastapi.testclient import TestClient

import server.app as app_module
import server.data_service as data_service
import server.routers.market as market_module


def test_health_endpoint_ok():
    client = TestClient(app_module.app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"


def test_symbols_returns_bitget_ranked_crypto_only(monkeypatch):
    async def fake_top_volume(top_n=300):
        assert top_n == 300
        return ["BTCUSDT", "TSLAUSDT", "ETHUSDT", "XAUUSDT", "SOLUSDT"]

    monkeypatch.setattr(data_service, "get_top_volume_symbols", fake_top_volume)
    client = TestClient(app_module.app)
    resp = client.get("/api/symbols?include_extended=false")

    assert resp.status_code == 200
    assert resp.json() == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_symbols_include_extended_is_deprecated_noop(monkeypatch):
    async def fake_top_volume(top_n=300):
        return ["HYPEUSDT", "MSTRUSDT", "BNBUSDT"]

    monkeypatch.setattr(data_service, "get_top_volume_symbols", fake_top_volume)
    client = TestClient(app_module.app)
    resp = client.get("/api/symbols?include_extended=true")

    assert resp.status_code == 200
    assert resp.json() == ["HYPEUSDT", "BNBUSDT"]


def test_symbols_falls_back_to_core_watchlist_when_bitget_unavailable(monkeypatch):
    async def fake_top_volume(top_n=300):
        raise RuntimeError("bitget unavailable")

    monkeypatch.setattr(data_service, "get_top_volume_symbols", fake_top_volume)
    client = TestClient(app_module.app)
    resp = client.get("/api/symbols?include_extended=false")

    assert resp.status_code == 200
    assert resp.json() == ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'HYPEUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT']


def test_symbol_info_uses_generic_exchange_metadata(monkeypatch):
    async def fake_symbol_metadata(symbol: str):
        assert symbol == "BTCUSDT"
        return {"instId": "BTCUSDT", "tickSz": "0.1", "pricePrecision": 1}

    monkeypatch.setattr(market_module, "get_symbol_metadata", fake_symbol_metadata)

    client = TestClient(app_module.app)
    resp = client.get("/api/symbol-info?symbol=BTCUSDT")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["pricePrecision"] == 1
    assert payload["tickSz"] == "0.1"


def test_symbols_screen_stub_is_fast_noop_for_frontend_filter():
    client = TestClient(app_module.app)

    resp = client.post(
        "/api/symbols/screen",
        json={
            "rules": [{"tf": "4h", "kind": "ema_cross", "fast": 20, "slow": 50}],
            "symbols": ["BTCUSDT"],
            "timeframe": "4h",
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["matched"] is None
    assert payload["implemented"] is False
