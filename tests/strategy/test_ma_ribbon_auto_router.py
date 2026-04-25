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
