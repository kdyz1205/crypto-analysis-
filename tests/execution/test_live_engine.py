from __future__ import annotations

import pytest

from server.execution.live_engine import LiveExecutionEngine
from server.execution.types import OrderIntent


class _FakeAdapter:
    def __init__(self, *, has_keys: bool = True, reconcile_blocked: bool = False) -> None:
        self._has_keys = has_keys
        self._reconcile_blocked = reconcile_blocked
        self.submit_calls = 0

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
        self.submit_calls += 1
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


def _intent() -> OrderIntent:
    return OrderIntent(
        order_intent_id="intent-1",
        signal_id="sig-1",
        line_id="line-1",
        client_order_id="client-1",
        symbol="BTCUSDT",
        timeframe="1h",
        side="short",
        order_type="market",
        trigger_mode="rejection",
        entry_price=100.0,
        stop_price=105.0,
        tp_price=90.0,
        quantity=0.5,
        status="approved",
        created_at_bar=1,
        created_at_ts=1,
    )


@pytest.mark.asyncio
async def test_live_engine_preview_requires_explicit_reconciliation(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.delenv("CONFIRM_LIVE_TRADING", raising=False)
    engine = LiveExecutionEngine(adapter_provider=lambda: _FakeAdapter())

    preview_before = await engine.preview_live_submission(_intent(), mode="demo")
    assert preview_before["ok"] is False
    assert preview_before["reason"] == "reconciliation_required"

    await engine.reconcile_startup("demo")
    preview_after = await engine.preview_live_submission(_intent(), mode="demo")
    assert preview_after["ok"] is True


@pytest.mark.asyncio
async def test_live_engine_live_mode_requires_dry_run_false(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", "true")
    monkeypatch.setenv("DRY_RUN", "true")
    engine = LiveExecutionEngine(adapter_provider=lambda: _FakeAdapter())

    await engine.reconcile_startup("live")
    preview_with_dry_run = await engine.preview_live_submission(_intent(), mode="live")
    assert preview_with_dry_run["ok"] is False
    assert "dry_run_enabled" in preview_with_dry_run["blocking_reasons"]

    monkeypatch.setenv("DRY_RUN", "false")
    preview_live = await engine.preview_live_submission(_intent(), mode="live")
    assert preview_live["ok"] is True


@pytest.mark.asyncio
async def test_live_engine_blocks_stale_reconciliation(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("DRY_RUN", "true")
    adapter = _FakeAdapter()
    engine = LiveExecutionEngine(adapter_provider=lambda: adapter)

    await engine.reconcile_startup("demo")
    stale_report = dict(engine.reconciliation_by_mode["demo"])
    stale_report["checked_at"] = 1
    engine.reconciliation_by_mode["demo"] = stale_report

    preview = await engine.preview_live_submission(_intent(), mode="demo")
    assert preview["ok"] is False
    assert "reconciliation_stale" in preview["blocking_reasons"]


@pytest.mark.asyncio
async def test_live_engine_is_idempotent_for_repeated_submit(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("DRY_RUN", "true")
    adapter = _FakeAdapter()
    engine = LiveExecutionEngine(adapter_provider=lambda: adapter)

    await engine.reconcile_startup("demo")
    first = await engine.execute_live_submission(_intent(), mode="demo", confirm=True)
    second = await engine.execute_live_submission(_intent(), mode="demo", confirm=True)

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["idempotent_replay"] is True
    assert adapter.submit_calls == 1


def test_live_engine_status_exposes_flags_and_reconciliation_requirement(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.delenv("CONFIRM_LIVE_TRADING", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)
    engine = LiveExecutionEngine(adapter_provider=lambda: _FakeAdapter())

    status = engine.get_status()

    assert status["enabled_flags"]["enable_live_trading"] is False
    assert status["enabled_flags"]["confirm_live_trading"] is False
    assert status["enabled_flags"]["dry_run"] is True
    assert status["reconciliation_required_by_mode"] == {"demo": True, "live": True}
    assert status["blocked_reason"] == "enable_live_trading_disabled"
