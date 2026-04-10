from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.routers.live_execution as live_execution_router
from server.execution.live_engine import LiveExecutionEngine
from server.execution.types import PaperExecutionConfig, RiskDecision
from server.strategy.types import StrategySignal


class _FakeAdapter:
    exchange_name = "bitget"

    def __init__(self, *, has_keys: bool = True, reconcile_blocked: bool = False) -> None:
        self._has_keys = has_keys
        self._reconcile_blocked = reconcile_blocked

    def has_api_keys(self) -> bool:
        return self._has_keys

    async def reconcile_live_state(self, mode: str) -> dict:
        return {
            "ok": not self._reconcile_blocked,
            "blocked": self._reconcile_blocked,
            "mode": mode,
            "reason": "exchange_positions_detected" if self._reconcile_blocked else "",
            "positions": [{"instId": "BTC-USDT-SWAP"}] if self._reconcile_blocked else [],
            "pending_orders": [],
            "exchange_response_excerpt": None,
        }

    async def submit_live_entry(self, intent, mode: str) -> dict:
        return {
            "ok": True,
            "mode": mode,
            "symbol": intent.symbol,
            "side": intent.side,
            "exchange_order_id": "ord-demo-1",
            "submitted_price": intent.entry_price,
            "submitted_notional": intent.entry_price * intent.quantity,
            "exchange_response_excerpt": {"ordId": "ord-demo-1"},
        }

    async def submit_live_close(self, symbol: str, mode: str) -> dict:
        return {
            "ok": True,
            "mode": mode,
            "symbol": symbol,
            "exchange_order_id": "close-1",
            "exchange_response_excerpt": {"ordId": "close-1"},
        }


def _signal(symbol: str = "BTCUSDT", timeframe: str = "1h", trigger_mode: str = "rejection") -> StrategySignal:
    return StrategySignal(
        signal_id=f"sig-{symbol}-{trigger_mode}",
        line_id=f"line-{symbol}",
        symbol=symbol,
        timeframe=timeframe,
        signal_type="REJECTION_SHORT",
        direction="short",
        trigger_mode=trigger_mode,
        timestamp=1,
        trigger_bar_index=1,
        score=0.8,
        priority_rank=1,
        entry_price=100.0,
        stop_price=105.0,
        tp_price=90.0,
        risk_reward=2.0,
        confirming_touch_count=3,
        bars_since_last_confirming_touch=1,
        distance_to_line=0.1,
        line_side="resistance",
        reason_code="test",
        factor_components={},
    )


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(live_execution_router.router)
    return app


def _seed_intent(signal: StrategySignal):
    live_execution_router.paper_engine.reset()
    decision = RiskDecision(
        signal_id=signal.signal_id,
        approved=True,
        blocking_reason="",
        risk_amount=30.0,
        proposed_quantity=0.5,
        stop_distance=5.0,
        exposure_after_fill=50.0,
    )
    return live_execution_router.paper_engine.order_manager.create_order_intent_from_signal(
        signal,
        decision,
        PaperExecutionConfig(),
        current_bar=1,
        current_ts=1,
    )


def test_live_execution_router_status_preview_submit_and_reconcile(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.delenv("CONFIRM_LIVE_TRADING", raising=False)
    live_execution_router.live_engine = LiveExecutionEngine(adapter_provider=lambda: _FakeAdapter())
    intent = _seed_intent(_signal())

    client = TestClient(_build_app())

    status_response = client.get("/api/live-execution/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"]["exchange"] == "bitget"
    assert status_response.json()["status"]["api_key_ready"] is True
    assert status_response.json()["status"]["enabled_flags"]["dry_run"] is True

    preview_response = client.post(
        "/api/live-execution/preview",
        json={"order_intent_id": intent.order_intent_id, "mode": "demo"},
    )
    assert preview_response.status_code == 200
    assert preview_response.json()["result"]["reason"] == "reconciliation_required"

    reconcile = client.post("/api/live-execution/reconcile", json={"mode": "demo"})
    assert reconcile.status_code == 200
    assert reconcile.json()["reconciliation"]["blocked"] is False

    preview_after_reconcile = client.post(
        "/api/live-execution/preview",
        json={"order_intent_id": intent.order_intent_id, "mode": "demo"},
    )
    assert preview_after_reconcile.status_code == 200
    assert preview_after_reconcile.json()["result"]["ok"] is True

    submit_without_confirm = client.post(
        "/api/live-execution/submit",
        json={"order_intent_id": intent.order_intent_id, "mode": "demo", "confirm": False},
    )
    assert submit_without_confirm.status_code == 200
    assert submit_without_confirm.json()["result"]["reason"] == "confirm_required"

    submit_demo = client.post(
        "/api/live-execution/submit",
        json={"order_intent_id": intent.order_intent_id, "mode": "demo", "confirm": True},
    )
    assert submit_demo.status_code == 200
    assert submit_demo.json()["result"]["ok"] is True
    assert submit_demo.json()["result"]["exchange_order_id"] == "ord-demo-1"

    submit_demo_again = client.post(
        "/api/live-execution/submit",
        json={"order_intent_id": intent.order_intent_id, "mode": "demo", "confirm": True},
    )
    assert submit_demo_again.status_code == 200
    assert submit_demo_again.json()["result"]["idempotent_replay"] is True


def test_live_execution_router_blocks_non_whitelist_and_live_without_confirm_flag(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.delenv("CONFIRM_LIVE_TRADING", raising=False)
    live_execution_router.live_engine = LiveExecutionEngine(adapter_provider=lambda: _FakeAdapter())
    non_whitelist_intent = _seed_intent(_signal(symbol="ETHUSDT"))

    client = TestClient(_build_app())

    blocked_symbol = client.post(
        "/api/live-execution/preview",
        json={"order_intent_id": non_whitelist_intent.order_intent_id, "mode": "demo"},
    )
    assert blocked_symbol.status_code == 200
    assert blocked_symbol.json()["result"]["reason"] == "reconciliation_required"

    allowed_intent = _seed_intent(_signal())
    reconcile_live = client.post("/api/live-execution/reconcile", json={"mode": "live"})
    assert reconcile_live.status_code == 200
    live_preview = client.post(
        "/api/live-execution/preview",
        json={"order_intent_id": allowed_intent.order_intent_id, "mode": "live"},
    )
    assert live_preview.status_code == 200
    assert live_preview.json()["result"]["reason"] == "confirm_live_trading_disabled"

    reconcile_demo = client.post("/api/live-execution/reconcile", json={"mode": "demo"})
    assert reconcile_demo.status_code == 200
    non_whitelist_after_reconcile = _seed_intent(_signal(symbol="ETHUSDT"))
    blocked_symbol_after_reconcile = client.post(
        "/api/live-execution/preview",
        json={"order_intent_id": non_whitelist_after_reconcile.order_intent_id, "mode": "demo"},
    )
    assert blocked_symbol_after_reconcile.status_code == 200
    assert blocked_symbol_after_reconcile.json()["result"]["reason"] == "symbol_not_whitelisted"


def test_live_execution_router_reconcile_reports_blocked_state(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    live_execution_router.live_engine = LiveExecutionEngine(
        adapter_provider=lambda: _FakeAdapter(reconcile_blocked=True)
    )
    _seed_intent(_signal())
    client = TestClient(_build_app())

    reconcile = client.post("/api/live-execution/reconcile", json={"mode": "demo"})
    assert reconcile.status_code == 200
    payload = reconcile.json()["reconciliation"]
    assert payload["blocked"] is True
    assert payload["reason"] == "exchange_positions_detected"
