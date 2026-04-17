import json
import time

import pytest

from server.execution.types import OrderIntent
from server.strategy import trendline_order_manager as tom
from pathlib import Path


class _FakeTrendlineAdapter:
    instances: list["_FakeTrendlineAdapter"] = []
    open_position_symbols: set[str] = set()
    pending_plan_order_ids: set[str] = set()
    pending_plan_rows: list[dict] = []

    def __init__(self) -> None:
        self.intents: list[tuple[OrderIntent, str, float]] = []
        self.requests: list[tuple[str, str, str, dict | None]] = []
        _FakeTrendlineAdapter.instances.append(self)

    def has_api_keys(self) -> bool:
        return True

    async def _bitget_request(self, method: str, path: str, *, mode: str, body=None, params=None):
        self.requests.append((method, path, mode, body))
        return {"code": "00000", "data": {"orderId": "cancel-ok"}}

    async def get_open_position_symbols(self, mode: str) -> set[str]:
        return set(self.open_position_symbols)

    async def get_pending_plan_orders(self, mode: str, *, plan_type: str = "normal_plan", symbol: str | None = None):
        if self.pending_plan_rows:
            return [
                row for row in self.pending_plan_rows
                if symbol is None or row.get("symbol") == symbol
            ]
        rows = []
        for order_id in self.pending_plan_order_ids:
            rows.append({
                "symbol": symbol or "LINKUSDT",
                "orderId": order_id,
                "planType": plan_type,
            })
        return rows

    async def cancel_plan_order(self, symbol: str, order_id: str, mode: str, *, plan_type: str = "normal_plan"):
        body = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "orderId": order_id,
            "planType": plan_type,
        }
        self.requests.append(("POST", "/api/v2/mix/order/cancel-plan-order", mode, body))
        return {"ok": True, "order_id": order_id, "plan_type": plan_type}

    async def submit_live_plan_entry(self, intent: OrderIntent, mode: str, trigger_price: float):
        self.intents.append((intent, mode, trigger_price))
        return {"ok": True, "exchange_order_id": f"plan-{len(self.intents)}"}


@pytest.fixture(autouse=True)
def _fake_adapter(monkeypatch):
    _FakeTrendlineAdapter.instances.clear()
    _FakeTrendlineAdapter.open_position_symbols.clear()
    _FakeTrendlineAdapter.pending_plan_order_ids.clear()
    _FakeTrendlineAdapter.pending_plan_rows.clear()
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeTrendlineAdapter)


def _cfg(**overrides):
    cfg = {
        "buffer_pct": 0.10,
        "rr": 15.0,
        "prices": {"LINKUSDT": 101.0},
        "leverage": 30,
        "equity": 1000.0,
        "risk_pct": 0.01,
        "max_position_pct": 0.50,
        "mode": "demo",
        "tf_risk": {"5m": 0.003, "15m": 0.007, "1h": 0.015, "4h": 0.030},
        "tf_buffer": {"5m": 0.05, "15m": 0.10, "1h": 0.20, "4h": 0.30},
    }
    cfg.update(overrides)
    return cfg


def _test_active_file(name: str) -> Path:
    path = Path("data/codex_tmp") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_new_order_uses_percent_tf_buffer_limit_entry_and_tf_risk(monkeypatch):
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", _test_active_file("active_new.json"))
    signal = {
        "symbol": "LINKUSDT",
        "timeframe": "1h",
        "kind": "support",
        "slope": 0.0,
        "intercept": 100.0,
        "bar_count": 10,
        "anchor1_bar": 1,
        "anchor2_bar": 5,
    }

    result = await tom.update_trendline_orders([signal], current_bar_index=9, cfg=_cfg())

    adapter = _FakeTrendlineAdapter.instances[-1]
    intent, mode, trigger = adapter.intents[-1]
    assert result["placed"] == 1
    assert mode == "demo"
    assert intent.order_type == "limit"
    assert trigger == pytest.approx(100.2)
    assert intent.entry_price == pytest.approx(100.2)
    assert intent.stop_price == pytest.approx(100.0)
    assert intent.tp_price == pytest.approx(103.206)
    assert intent.quantity == pytest.approx(75.0)
    saved = json.loads(tom.ACTIVE_LINES_FILE.read_text(encoding="utf-8"))
    assert saved[0]["line_ref_price"] == pytest.approx(100.0)
    assert saved[0]["line_ref_ts"] > 0


@pytest.mark.asyncio
async def test_existing_move_uses_tf_risk_instead_of_fallback_risk(monkeypatch):
    active_file = _test_active_file("active_move.json")
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", active_file)
    _FakeTrendlineAdapter.pending_plan_order_ids.add("old-plan")
    active_file.write_text(json.dumps([
        {
            "symbol": "LINKUSDT",
            "timeframe": "1h",
            "kind": "support",
            "slope": 0.0,
            "intercept": 100.0,
            "anchor1_bar": 1,
            "anchor2_bar": 5,
            "bar_count": 10,
            "current_projected_price": 100.0,
            "limit_price": 100.2,
            "stop_price": 100.0,
            "tp_price": 103.206,
            "exchange_order_id": "old-plan",
            "created_ts": time.time() - 7200,
            "last_updated_ts": time.time() - 3700,
            "status": "placed",
        }
    ]), encoding="utf-8")

    result = await tom.update_trendline_orders([], current_bar_index=-1, cfg=_cfg())

    adapter = _FakeTrendlineAdapter.instances[-1]
    intent, mode, trigger = adapter.intents[-1]
    assert result["updated"] == 1
    assert mode == "demo"
    assert trigger == pytest.approx(100.2)
    assert intent.quantity == pytest.approx(75.0)
    cancel_bodies = [
        req[3] for req in adapter.requests
        if req[1] == "/api/v2/mix/order/cancel-plan-order"
    ]
    assert cancel_bodies[-1]["planType"] == "normal_plan"


@pytest.mark.asyncio
async def test_missing_exchange_plan_order_is_marked_stale_without_replace(monkeypatch):
    active_file = _test_active_file("active_stale.json")
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", active_file)
    active_file.write_text(json.dumps([
        {
            "symbol": "LINKUSDT",
            "timeframe": "1h",
            "kind": "support",
            "slope": 0.0,
            "intercept": 100.0,
            "anchor1_bar": 1,
            "anchor2_bar": 5,
            "bar_count": 10,
            "current_projected_price": 100.0,
            "limit_price": 100.2,
            "stop_price": 100.0,
            "tp_price": 103.206,
            "exchange_order_id": "missing-plan",
            "created_ts": time.time() - 7200,
            "last_updated_ts": time.time() - 3700,
            "status": "placed",
        }
    ]), encoding="utf-8")

    result = await tom.update_trendline_orders([], current_bar_index=-1, cfg=_cfg())

    saved = json.loads(active_file.read_text(encoding="utf-8"))
    adapter = _FakeTrendlineAdapter.instances[-1]
    cancel_bodies = [
        req[3] for req in adapter.requests
        if req[1] == "/api/v2/mix/order/cancel-plan-order"
    ]
    assert result == {"placed": 0, "updated": 0, "cancelled": 0}
    assert saved[0]["status"] == "stale"
    assert adapter.intents == []
    assert cancel_bodies == []


@pytest.mark.asyncio
async def test_existing_support_order_broken_before_move_is_cancelled_without_replace(monkeypatch):
    active_file = _test_active_file("active_broken_before_move.json")
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", active_file)
    _FakeTrendlineAdapter.pending_plan_order_ids.add("old-plan")
    active_file.write_text(json.dumps([
        {
            "symbol": "LINKUSDT",
            "timeframe": "5m",
            "kind": "support",
            "slope": 0.0,
            "intercept": 100.0,
            "anchor1_bar": 1,
            "anchor2_bar": 5,
            "bar_count": 10,
            "current_projected_price": 100.0,
            "limit_price": 100.05,
            "stop_price": 100.0,
            "tp_price": 100.8,
            "exchange_order_id": "old-plan",
            "created_ts": time.time() - 1200,
            "last_updated_ts": time.time() - 600,
            "status": "placed",
        }
    ]), encoding="utf-8")

    result = await tom.update_trendline_orders(
        [],
        current_bar_index=-1,
        cfg=_cfg(prices={"LINKUSDT": 99.0}),
    )

    saved = json.loads(active_file.read_text(encoding="utf-8"))
    adapter = _FakeTrendlineAdapter.instances[-1]
    assert result == {"placed": 0, "updated": 0, "cancelled": 1}
    assert saved[0]["status"] == "broken"
    assert adapter.intents == []
    cancel_bodies = [
        req[3] for req in adapter.requests
        if req[1] == "/api/v2/mix/order/cancel-plan-order"
    ]
    assert cancel_bodies[-1]["orderId"] == "old-plan"
    assert cancel_bodies[-1]["planType"] == "normal_plan"


@pytest.mark.asyncio
async def test_exchange_orphan_trendline_plan_is_cancelled(monkeypatch):
    active_file = _test_active_file("active_orphan.json")
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", active_file)
    active_file.write_text(json.dumps([
        {
            "symbol": "LINKUSDT",
            "timeframe": "1h",
            "kind": "support",
            "slope": 0.0,
            "intercept": 100.0,
            "anchor1_bar": 1,
            "anchor2_bar": 5,
            "bar_count": 10,
            "current_projected_price": 100.0,
            "limit_price": 100.2,
            "stop_price": 100.0,
            "tp_price": 103.206,
            "exchange_order_id": "managed-plan",
            "created_ts": time.time(),
            "last_updated_ts": time.time(),
            "status": "placed",
        }
    ]), encoding="utf-8")
    _FakeTrendlineAdapter.pending_plan_rows.extend([
        {"symbol": "LINKUSDT", "orderId": "managed-plan", "clientOid": "tl_LINKUSDT_managed"},
        {"symbol": "ETHUSDT", "orderId": "orphan-plan", "clientOid": "tl_ETHUSDT_orphan"},
        {"symbol": "BTCUSDT", "orderId": "manual-plan", "clientOid": "manual_123"},
    ])

    result = await tom.update_trendline_orders([], current_bar_index=-1, cfg=_cfg())

    adapter = _FakeTrendlineAdapter.instances[-1]
    cancel_bodies = [
        req[3] for req in adapter.requests
        if req[1] == "/api/v2/mix/order/cancel-plan-order"
    ]
    assert result == {"placed": 0, "updated": 0, "cancelled": 1}
    assert [body["orderId"] for body in cancel_bodies] == ["orphan-plan"]
    assert cancel_bodies[0]["symbol"] == "ETHUSDT"
    assert cancel_bodies[0]["planType"] == "normal_plan"


@pytest.mark.asyncio
async def test_stale_local_order_does_not_block_fresh_signal(monkeypatch):
    active_file = _test_active_file("active_stale_fresh.json")
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", active_file)
    active_file.write_text(json.dumps([
        {
            "symbol": "LINKUSDT",
            "timeframe": "1h",
            "kind": "support",
            "slope": 0.0,
            "intercept": 100.0,
            "anchor1_bar": 1,
            "anchor2_bar": 5,
            "bar_count": 10,
            "current_projected_price": 100.0,
            "limit_price": 100.2,
            "stop_price": 100.0,
            "tp_price": 103.206,
            "exchange_order_id": "old-stale",
            "created_ts": time.time() - 7200,
            "last_updated_ts": time.time() - 3700,
            "status": "stale",
        }
    ]), encoding="utf-8")
    signal = {
        "symbol": "LINKUSDT",
        "timeframe": "1h",
        "kind": "support",
        "slope": 0.0,
        "intercept": 100.0,
        "bar_count": 10,
        "anchor1_bar": 1,
        "anchor2_bar": 5,
    }

    result = await tom.update_trendline_orders([signal], current_bar_index=9, cfg=_cfg())

    saved = json.loads(active_file.read_text(encoding="utf-8"))
    adapter = _FakeTrendlineAdapter.instances[-1]
    assert result["placed"] == 1
    assert len(adapter.intents) == 1
    assert {row["status"] for row in saved} == {"stale", "placed"}


@pytest.mark.asyncio
async def test_held_symbol_marks_active_order_filled_without_replacing(monkeypatch):
    active_file = _test_active_file("active_held.json")
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", active_file)
    active_file.write_text(json.dumps([
        {
            "symbol": "LINKUSDT",
            "timeframe": "1h",
            "kind": "support",
            "slope": 0.0,
            "intercept": 100.0,
            "anchor1_bar": 1,
            "anchor2_bar": 5,
            "bar_count": 10,
            "current_projected_price": 100.0,
            "limit_price": 100.2,
            "stop_price": 100.0,
            "tp_price": 103.206,
            "exchange_order_id": "old-plan",
            "created_ts": time.time() - 7200,
            "last_updated_ts": time.time() - 3700,
            "status": "placed",
        }
    ]), encoding="utf-8")

    result = await tom.update_trendline_orders(
        [],
        current_bar_index=-1,
        cfg=_cfg(held_symbols={"LINKUSDT"}),
    )

    saved = json.loads(active_file.read_text(encoding="utf-8"))
    adapter = _FakeTrendlineAdapter.instances[-1]
    assert result == {"placed": 0, "updated": 0, "cancelled": 0}
    assert saved[0]["status"] == "filled"
    assert adapter.intents == []


@pytest.mark.asyncio
async def test_exchange_held_symbol_blocks_new_plan_order(monkeypatch):
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", _test_active_file("active_exchange_held.json"))
    _FakeTrendlineAdapter.open_position_symbols.add("LINKUSDT")
    signal = {
        "symbol": "LINKUSDT",
        "timeframe": "1h",
        "kind": "support",
        "slope": 0.0,
        "intercept": 100.0,
        "bar_count": 10,
        "anchor1_bar": 1,
        "anchor2_bar": 5,
    }

    result = await tom.update_trendline_orders([signal], current_bar_index=9, cfg=_cfg(held_symbols=set()))

    adapter = _FakeTrendlineAdapter.instances[-1]
    assert result == {"placed": 0, "updated": 0, "cancelled": 0}
    assert adapter.intents == []


@pytest.mark.asyncio
async def test_cancel_all_trendline_plan_orders_for_risk_halt_skips_manual_orders(monkeypatch):
    active_file = _test_active_file("active_halt_cancel.json")
    monkeypatch.setattr(tom, "ACTIVE_LINES_FILE", active_file)
    active_file.write_text(json.dumps([
        {
            "symbol": "LINKUSDT",
            "timeframe": "1h",
            "kind": "support",
            "slope": 0.0,
            "intercept": 100.0,
            "anchor1_bar": 1,
            "anchor2_bar": 5,
            "bar_count": 10,
            "current_projected_price": 100.0,
            "limit_price": 100.2,
            "stop_price": 100.0,
            "tp_price": 103.206,
            "exchange_order_id": "managed-plan",
            "created_ts": time.time(),
            "last_updated_ts": time.time(),
            "status": "placed",
        }
    ]), encoding="utf-8")
    _FakeTrendlineAdapter.pending_plan_rows.extend([
        {"symbol": "LINKUSDT", "orderId": "managed-plan", "clientOid": "tl_LINKUSDT_managed"},
        {"symbol": "ETHUSDT", "orderId": "orphan-plan", "clientOid": "tl_ETHUSDT_orphan"},
        {"symbol": "BTCUSDT", "orderId": "manual-plan", "clientOid": "manual_123"},
    ])

    result = await tom.cancel_all_trendline_plan_orders(_cfg(), status="daily_halt")

    saved = json.loads(active_file.read_text(encoding="utf-8"))
    adapter = _FakeTrendlineAdapter.instances[-1]
    cancel_bodies = [
        req[3] for req in adapter.requests
        if req[1] == "/api/v2/mix/order/cancel-plan-order"
    ]
    assert result == {"cancelled": 2, "failed": 0, "status": "daily_halt"}
    assert saved[0]["status"] == "daily_halt"
    assert [body["orderId"] for body in cancel_bodies] == ["managed-plan", "orphan-plan"]
    assert all(body["planType"] == "normal_plan" for body in cancel_bodies)
