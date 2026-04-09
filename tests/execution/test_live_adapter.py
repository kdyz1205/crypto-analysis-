from __future__ import annotations

import pytest

from server.execution.live_adapter import LiveExecutionAdapter
from server.execution.types import OrderIntent


class _FakeTrader:
    def __init__(self, *, has_keys: bool = True) -> None:
        self.api_key = "key" if has_keys else ""
        self.api_secret = "secret" if has_keys else ""
        self.passphrase = "pass" if has_keys else ""

    def has_api_keys(self) -> bool:
        return bool(self.api_key and self.api_secret and self.passphrase)

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        return "signed"

    def _inst_id(self, symbol: str) -> str:
        return f"{symbol.replace('USDT', '')}-USDT-SWAP"

    async def _get_contract_size(self, inst_id: str) -> float:
        return 0.01

    async def get_price(self, symbol: str) -> float:
        return 100.0


class _RecordingAdapter(LiveExecutionAdapter):
    def __init__(self, trader) -> None:
        super().__init__(trader)
        self.calls: list[tuple[str, str]] = []
        self.status_payload: dict | None = None

    async def _okx_request(self, method: str, path: str, *, mode: str, body: str = "") -> dict:
        self.calls.append((method, mode))
        if path == "/api/v5/account/balance":
            return {
                "code": "0",
                "data": [{"totalEq": "1000", "details": [{"ccy": "USDT", "availBal": "500"}]}],
            }
        if path == "/api/v5/account/positions?instType=SWAP":
            return self.status_payload or {"code": "0", "data": []}
        if path == "/api/v5/trade/orders-pending?instType=SWAP":
            return {"code": "0", "data": []}
        return {"code": "0", "data": [{"ordId": "ord-1", "sCode": "0", "sMsg": ""}]}


def _intent(symbol: str = "BTCUSDT") -> OrderIntent:
    return OrderIntent(
        order_intent_id="intent-1",
        signal_id="sig-1",
        line_id="line-1",
        client_order_id="client-1",
        symbol=symbol,
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
async def test_live_adapter_rejects_when_api_keys_missing() -> None:
    adapter = LiveExecutionAdapter(_FakeTrader(has_keys=False))
    result = await adapter.submit_live_entry(_intent(), "demo")
    assert result["ok"] is False
    assert result["reason"] == "api_keys_missing"


@pytest.mark.asyncio
async def test_live_adapter_demo_mode_path_is_callable() -> None:
    adapter = _RecordingAdapter(_FakeTrader())
    result = await adapter.submit_live_entry(_intent(), "demo")
    assert result["ok"] is True
    assert result["mode"] == "demo"
    assert ("POST", "demo") in adapter.calls


@pytest.mark.asyncio
async def test_live_adapter_reconcile_blocks_when_exchange_position_exists() -> None:
    adapter = _RecordingAdapter(_FakeTrader())
    adapter.status_payload = {
        "code": "0",
        "data": [{"instId": "BTC-USDT-SWAP", "pos": "1", "posSide": "short", "avgPx": "100", "markPx": "99"}],
    }
    result = await adapter.reconcile_live_state("demo")
    assert result["blocked"] is True
    assert result["reason"] == "exchange_positions_detected"
