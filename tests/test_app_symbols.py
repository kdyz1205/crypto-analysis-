from fastapi.testclient import TestClient

import server.app as app_module


def test_health_endpoint_ok():
    client = TestClient(app_module.app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"


def test_symbols_default_excludes_extended_when_regular_sources_exist(monkeypatch):
    async def fake_okx_symbols():
        return {}

    monkeypatch.setattr(app_module, "load_okx_swap_symbols", fake_okx_symbols)
    monkeypatch.setattr(app_module, "_symbols_from_data_folder", lambda: ["LOCALUSDT"])
    monkeypatch.setattr(app_module, "_symbols_from_ticker_info_csv", lambda: ["EXTUSDT"])

    client = TestClient(app_module.app)
    resp = client.get("/api/symbols?include_extended=false")
    assert resp.status_code == 200
    symbols = resp.json()
    assert "LOCALUSDT" in symbols
    assert "EXTUSDT" not in symbols


def test_symbols_include_extended_true_merges_extended(monkeypatch):
    async def fake_okx_symbols():
        return {}

    monkeypatch.setattr(app_module, "load_okx_swap_symbols", fake_okx_symbols)
    monkeypatch.setattr(app_module, "_symbols_from_data_folder", lambda: ["LOCALUSDT"])
    monkeypatch.setattr(app_module, "_symbols_from_ticker_info_csv", lambda: ["EXTUSDT"])

    client = TestClient(app_module.app)
    resp = client.get("/api/symbols?include_extended=true")
    assert resp.status_code == 200
    symbols = resp.json()
    assert "LOCALUSDT" in symbols
    assert "EXTUSDT" in symbols


def test_symbols_default_falls_back_to_extended_if_regular_empty(monkeypatch):
    async def fake_okx_symbols():
        return {}

    monkeypatch.setattr(app_module, "load_okx_swap_symbols", fake_okx_symbols)
    monkeypatch.setattr(app_module, "_symbols_from_data_folder", lambda: [])
    monkeypatch.setattr(app_module, "_symbols_from_ticker_info_csv", lambda: ["EXTUSDT"])

    client = TestClient(app_module.app)
    resp = client.get("/api/symbols?include_extended=false")
    assert resp.status_code == 200
    symbols = resp.json()
    assert symbols == ["EXTUSDT"]
