from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Each test gets a clean state file in tmp_path."""
    state_path = tmp_path / "state.json"
    monkeypatch.setattr("server.routers.ma_ribbon_auto._STATE_PATH", state_path)
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._STATE_PATH_DEFAULT", state_path,
    )
    from server.strategy.ma_ribbon_auto_state import AutoState, save_state
    save_state(AutoState.default(), path=state_path)
    from server.routers.ma_ribbon_auto import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_status_returns_default_state_disabled(client):
    r = client.get("/api/ma_ribbon_auto/status")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["halted"] is False
    assert body["config"]["max_concurrent_orders"] == 25
    assert body["config"]["dd_halt_pct"] == 0.15
    assert body["current_ramp_cap_pct"] == 0.0  # never enabled → 0


def test_enable_requires_both_confirm_flags(client):
    r = client.post("/api/ma_ribbon_auto/enable", json={
        "confirm_acknowledged_p2_gate": True,
        # missing confirm_first_day_cap_2pct
        "strategy_capital_usd": 1000.0,
    })
    assert r.status_code == 400


def test_enable_with_both_flags_sets_first_enabled_and_enabled(client):
    r = client.post("/api/ma_ribbon_auto/enable", json={
        "confirm_acknowledged_p2_gate": True,
        "confirm_first_day_cap_2pct": True,
        "strategy_capital_usd": 1_000.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["first_enabled_at_utc"] is not None


def test_re_enable_does_not_reset_first_enabled_at(client):
    client.post("/api/ma_ribbon_auto/enable", json={
        "confirm_acknowledged_p2_gate": True,
        "confirm_first_day_cap_2pct": True,
        "strategy_capital_usd": 1_000.0,
    })
    first_ts = client.get("/api/ma_ribbon_auto/status").json()["first_enabled_at_utc"]
    client.post("/api/ma_ribbon_auto/disable")
    client.post("/api/ma_ribbon_auto/enable", json={
        "confirm_acknowledged_p2_gate": True,
        "confirm_first_day_cap_2pct": True,
        "strategy_capital_usd": 1_000.0,
    })
    again_ts = client.get("/api/ma_ribbon_auto/status").json()["first_enabled_at_utc"]
    assert again_ts == first_ts


def test_disable_flips_enabled_off(client):
    client.post("/api/ma_ribbon_auto/enable", json={
        "confirm_acknowledged_p2_gate": True,
        "confirm_first_day_cap_2pct": True,
        "strategy_capital_usd": 1_000.0,
    })
    r = client.post("/api/ma_ribbon_auto/disable")
    assert r.status_code == 200
    assert client.get("/api/ma_ribbon_auto/status").json()["enabled"] is False


def test_enable_rejects_zero_capital(client):
    r = client.post("/api/ma_ribbon_auto/enable", json={
        "confirm_acknowledged_p2_gate": True,
        "confirm_first_day_cap_2pct": True,
        "strategy_capital_usd": 0.0,
    })
    assert r.status_code == 400


def test_config_rejects_layer_risk_above_5pct(client):
    r = client.post("/api/ma_ribbon_auto/config", json={
        "layer_risk_pct": {"LV1": 0.06, "LV2": 0.0025, "LV3": 0.005, "LV4": 0.02},
    })
    assert r.status_code == 400


def test_config_accepts_valid_max_concurrent(client):
    r = client.post("/api/ma_ribbon_auto/config", json={
        "max_concurrent_orders": 10,
    })
    assert r.status_code == 200
    assert client.get("/api/ma_ribbon_auto/status").json()["config"]["max_concurrent_orders"] == 10


def test_config_rejects_dd_halt_above_50pct(client):
    r = client.post("/api/ma_ribbon_auto/config", json={
        "dd_halt_pct": 0.6,
    })
    assert r.status_code == 400
