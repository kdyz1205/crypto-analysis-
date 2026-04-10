from __future__ import annotations

import pytest

from server.execution.live_adapter import BitgetCredentials, LiveExecutionAdapter
from server.execution.types import OrderIntent


class _RecordingAdapter(LiveExecutionAdapter):
    def __init__(self, credentials: BitgetCredentials | None = None) -> None:
        super().__init__(credentials or BitgetCredentials("key", "secret", "pass"))
        self.calls: list[tuple[str, str, str]] = []
        self.position_payload: dict | None = None
        self.pending_payload: dict | None = None
        self.last_body: dict | None = None

    async def _get_contract(self, symbol: str) -> dict | None:
        return {
            "symbol": symbol.upper(),
            "lotSz": "0.001",
            "minTradeNum": "0.001",
        }

    async def _bitget_request(self, method: str, path: str, *, mode: str, params=None, body=None) -> dict:
        self.calls.append((method, path, mode))
        self.last_body = body
        if path == "/api/v2/mix/account/accounts":
            return {
                "code": "00000",
                "data": [
                    {
                        "marginCoin": "USDT",
                        "accountEquity": "1000",
                        "available": "500",
                    }
                ],
            }
        if path == "/api/v2/mix/position/all-position":
            return self.position_payload or {"code": "00000", "data": []}
        if path == "/api/v2/mix/order/orders-pending":
            return self.pending_payload or {"code": "00000", "data": []}
        if path == "/api/v2/mix/order/close-positions":
            return {"code": "00000", "data": {"orderId": "close-1"}}
        return {
            "code": "00000",
            "data": {
                "orderId": "ord-1",
                "clientOid": body.get("clientOid") if isinstance(body, dict) else "",
            },
        }


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
    adapter = LiveExecutionAdapter(BitgetCredentials())
    result = await adapter.submit_live_entry(_intent(), "demo")
    assert result["ok"] is False
    assert result["reason"] == "api_keys_missing"


@pytest.mark.asyncio
async def test_live_adapter_demo_mode_path_is_callable() -> None:
    adapter = _RecordingAdapter()
    result = await adapter.submit_live_entry(_intent(), "demo")
    assert result["ok"] is True
    assert result["mode"] == "demo"
    assert ("POST", "/api/v2/mix/order/place-order", "demo") in adapter.calls
    assert adapter.last_body["clientOid"] == "client-1"
    assert adapter.last_body["productType"] == "USDT-FUTURES"


@pytest.mark.asyncio
async def test_live_adapter_reconcile_blocks_when_exchange_position_exists() -> None:
    adapter = _RecordingAdapter()
    adapter.position_payload = {
        "code": "00000",
        "data": [
            {
                "symbol": "BTCUSDT",
                "holdSide": "short",
                "total": "1",
                "averageOpenPrice": "100",
                "markPrice": "99",
            }
        ],
    }
    result = await adapter.reconcile_live_state("demo")
    assert result["blocked"] is True
    assert result["reason"] == "exchange_positions_detected"


def test_bitget_credentials_support_secret_aliases(monkeypatch) -> None:
    monkeypatch.setenv("BITGET_API_KEY", "key")
    monkeypatch.setenv("BITGET_SECRET", "secret-from-alias")
    monkeypatch.setenv("BITGET_PASSPHRASE", "pass")
    monkeypatch.delenv("BITGET_SECRET_KEY", raising=False)

    creds = BitgetCredentials.from_env()

    assert creds.api_key == "key"
    assert creds.api_secret == "secret-from-alias"
    assert creds.passphrase == "pass"
    assert creds.is_ready() is True
