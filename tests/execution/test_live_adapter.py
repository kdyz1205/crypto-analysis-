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
        self.plan_pending_payloads: dict[str, dict] = {}
        self.last_body: dict | None = None
        self.bodies: list[dict | None] = []
        self.requests: list[tuple[str, str, str, dict | None]] = []

    async def _get_contract(self, symbol: str) -> dict | None:
        return {
            "symbol": symbol.upper(),
            "lotSz": "0.001",
            "minTradeNum": "0.001",
            "tickSz": "0.01",
        }

    async def _bitget_request(self, method: str, path: str, *, mode: str, params=None, body=None) -> dict:
        self.calls.append((method, path, mode))
        self.last_body = body
        self.bodies.append(body)
        self.requests.append((method, path, mode, body))
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
        if path == "/api/v2/mix/order/orders-plan-pending":
            plan_type = (params or {}).get("planType", "")
            return self.plan_pending_payloads.get(plan_type) or {"code": "00000", "data": {"entrustedList": []}}
        if path == "/api/v2/mix/order/close-positions":
            return {"code": "00000", "data": {"orderId": "close-1"}}
        if path == "/api/v2/mix/order/cancel-plan-order":
            order_id = body.get("orderId") if isinstance(body, dict) else ""
            if isinstance(body, dict) and body.get("planType"):
                return {"code": "00000", "data": {"successList": [{"orderId": order_id}], "failureList": []}}
            return {"code": "00000", "data": {"successList": [], "failureList": [{"orderId": order_id}]}}
        return {
            "code": "00000",
            "data": {
                "orderId": "ord-1",
                "clientOid": body.get("clientOid") if isinstance(body, dict) else "",
            },
        }


def _intent(symbol: str = "BTCUSDT", *, order_type: str = "market", post_only: bool = False) -> OrderIntent:
    return OrderIntent(
        order_intent_id="intent-1",
        signal_id="sig-1",
        line_id="line-1",
        client_order_id="client-1",
        symbol=symbol,
        timeframe="1h",
        side="short",
        order_type=order_type,  # type: ignore[arg-type]
        trigger_mode="rejection",
        entry_price=100.0,
        stop_price=105.0,
        tp_price=90.0,
        quantity=0.5,
        status="approved",
        created_at_bar=1,
        created_at_ts=1,
        post_only=post_only,
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
async def test_submit_live_entry_can_send_post_only_limit_with_preset_sl_tp() -> None:
    adapter = _RecordingAdapter()

    result = await adapter.submit_live_entry(_intent(order_type="limit", post_only=True), "demo")

    assert result["ok"] is True
    assert adapter.last_body["orderType"] == "limit"
    assert adapter.last_body["price"] == "100"
    assert adapter.last_body["force"] == "post_only"
    assert adapter.last_body["presetStopLossPrice"] == "105"
    assert adapter.last_body["presetStopSurplusPrice"] == "90"
    assert result["request_body_excerpt"]["force"] == "post_only"


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


@pytest.mark.asyncio
async def test_submit_live_plan_entry_uses_limit_trigger_with_preset_sl_tp() -> None:
    adapter = _RecordingAdapter()

    result = await adapter.submit_live_plan_entry(_intent(), "demo", trigger_price=100.0)

    assert result["ok"] is True
    assert ("POST", "/api/v2/mix/order/place-plan-order", "demo") in adapter.calls
    assert adapter.last_body["orderType"] == "limit"
    assert adapter.last_body["triggerPrice"] == "100"
    assert adapter.last_body["executePrice"] == "100"
    assert adapter.last_body["price"] == "100"
    assert adapter.last_body["stopLossTriggerPrice"] == "105"
    assert adapter.last_body["stopSurplusTriggerPrice"] == "90"
    assert result["request_body_excerpt"]["orderType"] == "limit"
    assert result["request_body_excerpt"]["price"] == "100"


@pytest.mark.asyncio
async def test_update_position_sl_tp_cancels_only_sl_not_tp_for_long() -> None:
    adapter = _RecordingAdapter()
    adapter.position_payload = {
        "code": "00000",
        "data": [
            {
                "symbol": "BTCUSDT",
                "holdSide": "long",
                "total": "1",
                "averageOpenPrice": "100",
            }
        ],
    }
    adapter.plan_pending_payloads = {
        "profit_loss": {
            "code": "00000",
            "data": {
                "entrustedList": [
                    {
                        "orderId": "trail-sl",
                        "planType": "pos_loss",
                        "side": "sell",
                        "tradeSide": "close",
                        "triggerPrice": "96",
                    },
                    {
                        "orderId": "preset-sl",
                        "planType": "loss_plan",
                        "side": "sell",
                        "tradeSide": "close",
                        "triggerPrice": "95",
                    },
                    {
                        "orderId": "preset-tp",
                        "planType": "profit_plan",
                        "side": "sell",
                        "tradeSide": "close",
                        "triggerPrice": "130",
                    },
                ]
            },
        },
    }

    result = await adapter.update_position_sl_tp("BTCUSDT", "long", new_sl=101.234, new_tp=None, mode="demo")

    assert result["ok"] is True
    cancelled_ids = [
        body["orderId"]
        for method, path, _, body in adapter.requests
        if method == "POST" and path == "/api/v2/mix/order/cancel-plan-order" and isinstance(body, dict)
    ]
    assert "trail-sl" in cancelled_ids
    assert "preset-sl" in cancelled_ids
    assert "preset-tp" not in cancelled_ids
    cancel_bodies = [
        body
        for method, path, _, body in adapter.requests
        if method == "POST" and path == "/api/v2/mix/order/cancel-plan-order" and isinstance(body, dict)
    ]
    assert {body["planType"] for body in cancel_bodies} == {"pos_loss", "loss_plan"}
    place_bodies = [
        body for method, path, _, body in adapter.requests
        if method == "POST" and path == "/api/v2/mix/order/place-tpsl-order" and isinstance(body, dict)
    ]
    assert place_bodies[-1]["planType"] == "pos_loss"
    assert place_bodies[-1]["triggerPrice"] == "101.23"


@pytest.mark.asyncio
async def test_cancel_plan_order_requires_plan_type_success_list() -> None:
    adapter = _RecordingAdapter()

    ok = await adapter.cancel_plan_order("BTCUSDT", "plan-1", "demo", plan_type="normal_plan")
    missing = await adapter._bitget_request(
        "POST",
        "/api/v2/mix/order/cancel-plan-order",
        mode="demo",
        body={"symbol": "BTCUSDT", "productType": "USDT-FUTURES", "orderId": "plan-2"},
    )

    assert ok["ok"] is True
    assert ok["plan_type"] == "normal_plan"
    assert missing["data"]["successList"] == []


@pytest.mark.asyncio
async def test_get_pending_plan_orders_returns_exchange_order_rows() -> None:
    adapter = _RecordingAdapter()
    adapter.plan_pending_payloads["normal_plan"] = {
        "code": "00000",
        "data": {
            "entrustedList": [
                {"symbol": "BTCUSDT", "orderId": "plan-1", "planType": "normal_plan"},
                {"symbol": "BTCUSDT", "orderId": "", "planType": "normal_plan"},
            ]
        },
    }

    rows = await adapter.get_pending_plan_orders("demo", plan_type="normal_plan", symbol="BTCUSDT")

    assert [row["orderId"] for row in rows] == ["plan-1"]
    get_requests = [
        req for req in adapter.requests
        if req[0] == "GET" and req[1] == "/api/v2/mix/order/orders-plan-pending"
    ]
    assert get_requests[-1][3] is None
