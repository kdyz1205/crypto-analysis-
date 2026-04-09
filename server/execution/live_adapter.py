from __future__ import annotations

import json
import time
from typing import Any, Literal

import httpx

from ..okx_trader import OKX_REST_BASE
from .types import OrderIntent

LiveMode = Literal["demo", "live"]


class LiveExecutionAdapter:
    def __init__(self, trader) -> None:
        self.trader = trader

    def has_api_keys(self) -> bool:
        return bool(self.trader and self.trader.has_api_keys())

    async def submit_live_entry(self, intent: OrderIntent, mode: LiveMode) -> dict[str, Any]:
        if not self.has_api_keys():
            return self._error_result("api_keys_missing", mode=mode, intent=intent)

        symbol = intent.symbol.upper()
        inst_id = self.trader._inst_id(symbol)
        reference_price = float(await self.trader.get_price(symbol) or intent.entry_price or 0.0)
        if reference_price <= 0:
            return self._error_result("reference_price_unavailable", mode=mode, intent=intent)

        ct_val = float(await self.trader._get_contract_size(inst_id))
        if ct_val <= 0:
            return self._error_result("contract_size_unavailable", mode=mode, intent=intent)

        intended_notional = max(float(intent.quantity) * float(intent.entry_price), reference_price * ct_val)
        contracts = max(1, int(intended_notional / (reference_price * ct_val)))
        submitted_notional = float(contracts * reference_price * ct_val)

        body = json.dumps(
            {
                "instId": inst_id,
                "tdMode": "cross",
                "side": "buy" if intent.side == "long" else "sell",
                "posSide": "long" if intent.side == "long" else "short",
                "ordType": "market",
                "sz": str(contracts),
            }
        )
        response = await self._okx_request("POST", "/api/v5/trade/order", mode=mode, body=body)
        if response.get("code") == "0":
            data = (response.get("data") or [{}])[0]
            return {
                "ok": True,
                "mode": mode,
                "symbol": symbol,
                "side": intent.side,
                "exchange_order_id": data.get("ordId", ""),
                "submitted_price": reference_price,
                "submitted_notional": submitted_notional,
                "exchange_response_excerpt": {
                    "sCode": data.get("sCode"),
                    "sMsg": data.get("sMsg"),
                    "ordId": data.get("ordId"),
                },
            }
        return self._error_result(
            self._extract_error_reason(response),
            mode=mode,
            intent=intent,
            response=response,
        )

    async def submit_live_close(self, symbol: str, mode: LiveMode) -> dict[str, Any]:
        if not self.has_api_keys():
            return {"ok": False, "mode": mode, "symbol": symbol, "reason": "api_keys_missing"}

        inst_id = self.trader._inst_id(symbol.upper())
        body = json.dumps({"instId": inst_id, "mgnMode": "cross"})
        response = await self._okx_request("POST", "/api/v5/trade/close-position", mode=mode, body=body)
        if response.get("code") == "0":
            data = (response.get("data") or [{}])[0]
            return {
                "ok": True,
                "mode": mode,
                "symbol": symbol.upper(),
                "exchange_order_id": data.get("ordId", ""),
                "exchange_response_excerpt": data,
            }
        return {
            "ok": False,
            "mode": mode,
            "symbol": symbol.upper(),
            "reason": self._extract_error_reason(response),
            "exchange_response_excerpt": self._excerpt(response),
        }

    async def get_live_account_status(self, mode: LiveMode) -> dict[str, Any]:
        if not self.has_api_keys():
            return {"ok": False, "mode": mode, "reason": "api_keys_missing"}

        balance = await self._okx_request("GET", "/api/v5/account/balance", mode=mode)
        positions = await self._okx_request("GET", "/api/v5/account/positions?instType=SWAP", mode=mode)
        pending = await self._okx_request("GET", "/api/v5/trade/orders-pending?instType=SWAP", mode=mode)

        if balance.get("code") != "0":
            return {
                "ok": False,
                "mode": mode,
                "reason": self._extract_error_reason(balance),
                "exchange_response_excerpt": self._excerpt(balance),
            }

        balance_payload = (balance.get("data") or [{}])[0]
        details = balance_payload.get("details") or []
        usdt = next((row for row in details if row.get("ccy") == "USDT"), None)
        active_positions = [
            {
                "instId": row.get("instId"),
                "pos": row.get("pos"),
                "posSide": row.get("posSide"),
                "avgPx": row.get("avgPx"),
                "markPx": row.get("markPx"),
            }
            for row in (positions.get("data") or [])
            if float(row.get("pos") or 0) != 0
        ]
        pending_orders = [
            {
                "instId": row.get("instId"),
                "ordId": row.get("ordId"),
                "side": row.get("side"),
                "sz": row.get("sz"),
                "ordType": row.get("ordType"),
            }
            for row in (pending.get("data") or [])
        ]

        return {
            "ok": True,
            "mode": mode,
            "account_accessible": True,
            "total_equity": float(balance_payload.get("totalEq") or 0.0),
            "usdt_available": float((usdt or {}).get("availBal") or 0.0),
            "positions": active_positions,
            "pending_orders": pending_orders,
            "exchange_response_excerpt": {
                "positions_count": len(active_positions),
                "pending_orders_count": len(pending_orders),
            },
        }

    async def reconcile_live_state(self, mode: LiveMode) -> dict[str, Any]:
        status = await self.get_live_account_status(mode)
        if not status.get("ok"):
            return {
                "ok": False,
                "blocked": True,
                "mode": mode,
                "reason": status.get("reason", "live_status_unavailable"),
                "positions": [],
                "pending_orders": [],
                "exchange_response_excerpt": status.get("exchange_response_excerpt"),
            }

        positions = status.get("positions", [])
        pending_orders = status.get("pending_orders", [])
        reasons: list[str] = []
        if positions:
            reasons.append("exchange_positions_detected")
        if pending_orders:
            reasons.append("exchange_pending_orders_detected")

        return {
            "ok": len(reasons) == 0,
            "blocked": len(reasons) > 0,
            "mode": mode,
            "reason": ",".join(reasons) if reasons else "",
            "positions": positions,
            "pending_orders": pending_orders,
            "account_accessible": True,
            "total_equity": status.get("total_equity", 0.0),
            "usdt_available": status.get("usdt_available", 0.0),
            "checked_at": int(time.time()),
            "exchange_response_excerpt": status.get("exchange_response_excerpt"),
        }

    async def _okx_request(self, method: str, path: str, *, mode: LiveMode, body: str = "") -> dict[str, Any]:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        headers = {
            "OK-ACCESS-KEY": self.trader.api_key,
            "OK-ACCESS-SIGN": self.trader._sign(timestamp, method, path, body),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.trader.passphrase,
            "Content-Type": "application/json",
        }
        if mode == "demo":
            headers["x-simulated-trading"] = "1"

        async with httpx.AsyncClient(timeout=15.0) as client:
            if method == "GET":
                response = await client.get(f"{OKX_REST_BASE}{path}", headers=headers)
            else:
                response = await client.post(f"{OKX_REST_BASE}{path}", headers=headers, content=body)
        if response.status_code >= 400:
            return {"code": "-1", "msg": f"HTTP {response.status_code}: {response.text[:200]}"}
        return response.json()

    @staticmethod
    def _extract_error_reason(response: dict[str, Any]) -> str:
        if response.get("msg"):
            return str(response["msg"])
        data = response.get("data") or []
        if data and isinstance(data[0], dict):
            return str(data[0].get("sMsg") or data[0].get("msg") or "unknown_exchange_error")
        return "unknown_exchange_error"

    @staticmethod
    def _excerpt(response: dict[str, Any] | None) -> dict[str, Any] | None:
        if not response:
            return None
        data = response.get("data")
        excerpt_data = data[0] if isinstance(data, list) and data else data
        return {
            "code": response.get("code"),
            "msg": response.get("msg"),
            "data": excerpt_data,
        }

    def _error_result(
        self,
        reason: str,
        *,
        mode: LiveMode,
        intent: OrderIntent | None = None,
        response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "mode": mode,
            "symbol": intent.symbol if intent else "",
            "side": intent.side if intent else "",
            "reason": reason,
            "exchange_order_id": "",
            "submitted_price": float(intent.entry_price) if intent else 0.0,
            "submitted_notional": float(intent.entry_price * intent.quantity) if intent else 0.0,
            "exchange_response_excerpt": self._excerpt(response),
        }


__all__ = ["LiveExecutionAdapter", "LiveMode"]
