from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.conditionals import ConditionalOrder, OrderConfig, TriggerConfig
import server.routers.drawings as drawings_router
import server.strategy.drawing_learner as drawing_learner
from server.drawings.store import ManualTrendlineStore


def _sample_polars_df() -> pl.DataFrame:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "open_time": [start + timedelta(hours=i) for i in range(40)],
            "open": [1.0 + (0.01 * i) for i in range(40)],
            "high": [1.05 + (0.01 * i) for i in range(40)],
            "low": [0.95 + (0.01 * i) for i in range(40)],
            "close": [1.01 + (0.01 * i) for i in range(40)],
            "volume": [100.0 + i for i in range(40)],
        }
    ).with_columns(pl.col("open_time").cast(pl.Datetime("us")))


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(drawings_router.router)
    return app


class _FakeCascadeStore:
    def __init__(self, cond):
        self.cond = cond
        self.events = []

    def list_all(self, *, status=None, symbol=None, manual_line_id=None):
        if manual_line_id and manual_line_id != self.cond.manual_line_id:
            return []
        if status and status != self.cond.status:
            return []
        return [self.cond]

    def append_event(self, conditional_id, event):
        self.events.append((conditional_id, event))
        self.cond.events.append(event)
        return self.cond

    def set_status(self, conditional_id, status, *, reason=""):
        if conditional_id != self.cond.conditional_id:
            return None
        self.cond.status = status
        if status == "cancelled":
            self.cond.cancel_reason = reason
        return self.cond


class _FakeCascadeAdapter:
    plan_ok = False
    pending_oids = {"plan-order-1"}

    def has_api_keys(self):
        return True

    async def cancel_order(self, symbol, order_id, mode):
        return {"ok": False, "reason": "regular_failed"}

    async def cancel_plan_order_any_type(self, symbol, order_id, mode):
        return {"ok": self.plan_ok, "reason": "plan_failed"}

    async def get_pending_orders(self, mode, *, symbol=None):
        return [{"orderId": oid, "symbol": symbol or "BTCUSDT"} for oid in self.pending_oids]

    async def get_pending_plan_orders(self, mode, *, plan_type="normal_plan", symbol=None):
        return [{"orderId": oid, "symbol": symbol or "BTCUSDT"} for oid in self.pending_oids]


def _cond_for_line(line_id: str) -> ConditionalOrder:
    return ConditionalOrder(
        conditional_id="cond-for-delete",
        manual_line_id=line_id,
        symbol="BTCUSDT",
        timeframe="4h",
        side="resistance",
        t_start=1712592000,
        t_end=1712678400,
        price_start=100.0,
        price_end=105.0,
        pattern_stats_at_create={},
        trigger=TriggerConfig(poll_seconds=60),
        order=OrderConfig(
            direction="short",
            order_kind="bounce",
            tolerance_pct_of_line=0.05,
            stop_offset_pct_of_line=0.01,
            rr_target=8.0,
            notional_usd=10.0,
            submit_to_exchange=True,
            exchange_mode="live",
        ),
        status="triggered",
        created_at=1712592000,
        updated_at=1712592000,
        exchange_order_id="plan-order-1",
        extend_right=True,
    )


def test_manual_drawing_crud(monkeypatch, tmp_path: Path) -> None:
    async def fake_get_ohlcv_with_df(symbol, interval, end_time, days, **kwargs):
        return _sample_polars_df(), {"pricePrecision": 4}

    monkeypatch.setattr(drawings_router, "get_ohlcv_with_df", fake_get_ohlcv_with_df)
    drawings_router.store = ManualTrendlineStore(tmp_path / "manual_trendlines.json")
    monkeypatch.setattr(drawing_learner, "ML_DRAWINGS_FILE", tmp_path / "user_drawings_ml.jsonl")

    client = TestClient(_build_app())
    create_response = client.post(
        "/api/drawings",
        json={
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "side": "resistance",
            "t_start": 1712592000,
            "t_end": 1712678400,
            "price_start": 100.0,
            "price_end": 105.0,
            "line_width": 3.0,
            "label": "manual resistance",
            "override_mode": "display_only",
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()["drawing"]
    assert created["manual_line_id"].startswith("manual-BTCUSDT-4h-resistance")
    assert created["comparison_status"] == "uncompared"
    assert created["line_width"] == 3.0

    list_response = client.get("/api/drawings?symbol=BTCUSDT&timeframe=4h")
    assert list_response.status_code == 200
    assert len(list_response.json()["drawings"]) == 1

    update_response = client.patch(
        f"/api/drawings/{created['manual_line_id']}",
        json={
            "override_mode": "suppress_nearest_auto_line",
            "locked": True,
            "extend_left": True,
            "line_width": 5.0,
            "label": "desk override",
            "notes": "manual review line",
        },
    )
    assert update_response.status_code == 200
    updated = update_response.json()["drawing"]
    assert updated["override_mode"] == "suppress_nearest_auto_line"
    assert updated["locked"] is True
    assert updated["extend_left"] is True
    assert updated["line_width"] == 5.0
    assert updated["label"] == "desk override"
    assert updated["notes"] == "manual review line"

    delete_response = client.delete(f"/api/drawings/{created['manual_line_id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["removed"] == 1

    empty_response = client.get("/api/drawings?symbol=BTCUSDT&timeframe=4h")
    assert empty_response.status_code == 200
    assert empty_response.json()["drawings"] == []


def test_clear_drawings_removes_only_requested_symbol_timeframe(monkeypatch, tmp_path: Path) -> None:
    drawings_router.store = ManualTrendlineStore(tmp_path / "manual_trendlines.json")
    monkeypatch.setattr(drawing_learner, "ML_DRAWINGS_FILE", tmp_path / "user_drawings_ml.jsonl")
    client = TestClient(_build_app())

    rows = [
        ("HYPEUSDT", "15m", 1712592000, 1712678400, 100.0, 101.0),
        ("HYPEUSDT", "15m", 1712682000, 1712768400, 102.0, 103.0),
        ("HYPEUSDT", "4h", 1712592000, 1712678400, 104.0, 105.0),
    ]
    for symbol, timeframe, t1, t2, p1, p2 in rows:
        response = client.post(
            "/api/drawings",
            json={
                "symbol": symbol,
                "timeframe": timeframe,
                "side": "support",
                "t_start": t1,
                "t_end": t2,
                "price_start": p1,
                "price_end": p2,
                "label": "manual support",
                "override_mode": "display_only",
            },
        )
        assert response.status_code == 200

    clear_response = client.post("/api/drawings/clear?symbol=HYPEUSDT&timeframe=15m")
    assert clear_response.status_code == 200
    assert clear_response.json()["removed"] == 2

    h15 = client.get("/api/drawings?symbol=HYPEUSDT&timeframe=15m")
    h4 = client.get("/api/drawings?symbol=HYPEUSDT&timeframe=4h")
    assert h15.status_code == 200
    assert h4.status_code == 200
    assert h15.json()["drawings"] == []
    assert len(h4.json()["drawings"]) == 1


def test_delete_drawing_refuses_when_line_has_active_order(monkeypatch, tmp_path: Path) -> None:
    drawings_router.store = ManualTrendlineStore(tmp_path / "manual_trendlines.json")
    monkeypatch.setattr(drawing_learner, "ML_DRAWINGS_FILE", tmp_path / "user_drawings_ml.jsonl")
    client = TestClient(_build_app())

    create_response = client.post(
        "/api/drawings",
        json={
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "side": "resistance",
            "t_start": 1712592000,
            "t_end": 1712678400,
            "price_start": 100.0,
            "price_end": 105.0,
            "label": "manual resistance",
            "override_mode": "display_only",
        },
    )
    line_id = create_response.json()["drawing"]["manual_line_id"]
    cond = _cond_for_line(line_id)
    fake_store = _FakeCascadeStore(cond)

    import server.conditionals as conditionals_pkg

    monkeypatch.setattr(conditionals_pkg, "ConditionalOrderStore", lambda: fake_store)

    delete_response = client.delete(f"/api/drawings/{line_id}")

    assert delete_response.status_code == 409
    assert delete_response.json()["detail"]["reason"] == "active_conditionals_protect_line"
    assert drawings_router.store.get(line_id) is not None
    assert cond.status == "triggered"
    assert fake_store.events == []


def test_delete_drawing_removes_line_only_after_orders_are_inactive(monkeypatch, tmp_path: Path) -> None:
    drawings_router.store = ManualTrendlineStore(tmp_path / "manual_trendlines.json")
    monkeypatch.setattr(drawing_learner, "ML_DRAWINGS_FILE", tmp_path / "user_drawings_ml.jsonl")
    client = TestClient(_build_app())

    create_response = client.post(
        "/api/drawings",
        json={
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "side": "resistance",
            "t_start": 1712592000,
            "t_end": 1712678400,
            "price_start": 100.0,
            "price_end": 105.0,
            "label": "manual resistance",
            "override_mode": "display_only",
        },
    )
    line_id = create_response.json()["drawing"]["manual_line_id"]
    cond = _cond_for_line(line_id)
    cond.status = "cancelled"
    fake_store = _FakeCascadeStore(cond)

    import server.conditionals as conditionals_pkg

    monkeypatch.setattr(conditionals_pkg, "ConditionalOrderStore", lambda: fake_store)

    delete_response = client.delete(f"/api/drawings/{line_id}")

    assert delete_response.status_code == 200
    assert drawings_router.store.get(line_id) is None
    assert cond.status == "cancelled"
    assert fake_store.events == []
