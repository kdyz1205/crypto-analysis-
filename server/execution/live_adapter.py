from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Any, Literal
from urllib.parse import urlencode

import httpx

from ..market import BitgetPublicClient, bitget_contracts_to_symbol_map
from .types import OrderIntent

BITGET_REST_BASE = "https://api.bitget.com"
BITGET_PRODUCT_TYPE = "USDT-FUTURES"
BITGET_MARGIN_COIN = "USDT"
BITGET_MARGIN_MODE = "crossed"

LiveMode = Literal["demo", "live"]


@dataclass(slots=True)
class BitgetCredentials:
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""

    @classmethod
    def from_env(cls) -> "BitgetCredentials":
        return cls(
            api_key=os.environ.get("BITGET_API_KEY", ""),
            api_secret=(
                os.environ.get("BITGET_SECRET_KEY", "")
                or os.environ.get("BITGET_SECRET", "")
                or os.environ.get("BITGET_API_SECRET", "")
            ),
            passphrase=os.environ.get("BITGET_PASSPHRASE", ""),
        )

    def is_ready(self) -> bool:
        return bool(self.api_key and self.api_secret and self.passphrase)


class LiveExecutionAdapter:
    exchange_name = "bitget"

    def __init__(
        self,
        credentials: BitgetCredentials | None = None,
        *,
        public_client: BitgetPublicClient | None = None,
        base_url: str = BITGET_REST_BASE,
        product_type: str = BITGET_PRODUCT_TYPE,
        margin_coin: str = BITGET_MARGIN_COIN,
        margin_mode: str = BITGET_MARGIN_MODE,
        locale: str = "en-US",
    ) -> None:
        self.credentials = credentials or BitgetCredentials.from_env()
        self.base_url = base_url.rstrip("/")
        self.product_type = product_type
        self.margin_coin = margin_coin
        self.margin_mode = margin_mode
        self.locale = locale
        self.public_client = public_client or BitgetPublicClient(
            base_url=self.base_url,
            product_type=self.product_type.lower(),
        )
        self._contract_cache: dict[str, dict[str, Any]] = {}

    def has_api_keys(self) -> bool:
        return self.credentials.is_ready()

    async def submit_live_entry(self, intent: OrderIntent, mode: LiveMode) -> dict[str, Any]:
        if not self.has_api_keys():
            return self._error_result("api_keys_missing", mode=mode, intent=intent)

        contract = await self._get_contract(intent.symbol)
        if not contract:
            return self._error_result("contract_not_found", mode=mode, intent=intent)

        normalized_size = self._normalize_size(intent.quantity, contract)
        if normalized_size is None:
            return self._error_result("size_below_min_trade", mode=mode, intent=intent)

        reference_price = float(intent.entry_price or 0.0)
        if reference_price <= 0:
            return self._error_result("reference_price_unavailable", mode=mode, intent=intent)

        side = "buy" if intent.side == "long" else "sell"
        body = {
            "symbol": intent.symbol.upper(),
            "productType": self.product_type,
            "marginMode": self.margin_mode,
            "marginCoin": self.margin_coin,
            "side": side,
            "tradeSide": "open",
            "orderType": "market",
            "size": normalized_size,
            "clientOid": self._client_order_id(intent),
        }
        response = await self._bitget_request("POST", "/api/v2/mix/order/place-order", mode=mode, body=body)
        if response.get("code") == "00000":
            data = response.get("data") or {}
            submitted_notional = float(Decimal(normalized_size) * Decimal(str(reference_price)))
            return {
                "ok": True,
                "mode": mode,
                "symbol": intent.symbol.upper(),
                "side": intent.side,
                "exchange_order_id": str(data.get("orderId") or data.get("ordId") or ""),
                "submitted_price": reference_price,
                "submitted_notional": submitted_notional,
                "exchange_response_excerpt": {
                    "orderId": data.get("orderId"),
                    "clientOid": data.get("clientOid"),
                },
            }
        return self._error_result(
            self._extract_error_reason(response),
            mode=mode,
            intent=intent,
            response=response,
        )

    async def submit_live_close(self, symbol: str, mode: LiveMode) -> dict[str, Any]:
        normalized_symbol = symbol.upper().replace("/", "")
        if not self.has_api_keys():
            return {"ok": False, "mode": mode, "symbol": normalized_symbol, "reason": "api_keys_missing"}

        body = {
            "symbol": normalized_symbol,
            "productType": self.product_type,
        }
        response = await self._bitget_request("POST", "/api/v2/mix/order/close-positions", mode=mode, body=body)
        if response.get("code") == "00000":
            data = response.get("data") or {}
            return {
                "ok": True,
                "mode": mode,
                "symbol": normalized_symbol,
                "exchange_order_id": str(data.get("orderId") or data.get("orderIdList") or ""),
                "exchange_response_excerpt": data,
            }
        return {
            "ok": False,
            "mode": mode,
            "symbol": normalized_symbol,
            "reason": self._extract_error_reason(response),
            "exchange_response_excerpt": self._excerpt(response),
        }

    async def get_live_account_status(self, mode: LiveMode) -> dict[str, Any]:
        if not self.has_api_keys():
            return {"ok": False, "mode": mode, "reason": "api_keys_missing"}

        accounts = await self._bitget_request(
            "GET",
            "/api/v2/mix/account/accounts",
            mode=mode,
            params={"productType": self.product_type},
        )
        positions = await self._bitget_request(
            "GET",
            "/api/v2/mix/position/all-position",
            mode=mode,
            params={"productType": self.product_type, "marginCoin": self.margin_coin},
        )
        pending = await self._bitget_request(
            "GET",
            "/api/v2/mix/order/orders-pending",
            mode=mode,
            params={"productType": self.product_type},
        )

        if accounts.get("code") != "00000":
            return {
                "ok": False,
                "mode": mode,
                "reason": self._extract_error_reason(accounts),
                "exchange_response_excerpt": self._excerpt(accounts),
            }

        account_rows = self._as_rows(accounts.get("data"))
        usdt_account = next(
            (
                row
                for row in account_rows
                if str(row.get("marginCoin") or row.get("coin") or "").upper() == self.margin_coin
            ),
            account_rows[0] if account_rows else {},
        )
        position_rows = self._as_rows(positions.get("data"))
        active_positions = [
            {
                "symbol": row.get("symbol"),
                "holdSide": row.get("holdSide") or row.get("posSide"),
                "total": row.get("total") or row.get("available"),
                "averageOpenPrice": row.get("averageOpenPrice") or row.get("openPriceAvg"),
                "markPrice": row.get("markPrice"),
            }
            for row in position_rows
            if self._position_size(row) > 0
        ]
        pending_rows = self._as_rows(pending.get("data"))
        pending_orders = [
            {
                "symbol": row.get("symbol"),
                "orderId": row.get("orderId"),
                "clientOid": row.get("clientOid"),
                "side": row.get("side"),
                "size": row.get("size"),
                "orderType": row.get("orderType"),
                "status": row.get("state") or row.get("status"),
            }
            for row in pending_rows
        ]

        return {
            "ok": True,
            "mode": mode,
            "account_accessible": True,
            "total_equity": float(usdt_account.get("accountEquity") or usdt_account.get("equity") or 0.0),
            "usdt_available": float(
                usdt_account.get("available")
                or usdt_account.get("availableBalance")
                or usdt_account.get("maxOpenPosAvailable")
                or 0.0
            ),
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

    async def _get_contract(self, symbol: str) -> dict[str, Any] | None:
        normalized_symbol = symbol.upper().replace("/", "")
        cached = self._contract_cache.get(normalized_symbol)
        if cached is not None:
            return cached

        rows = await self.public_client.get_contracts(normalized_symbol)
        contracts = bitget_contracts_to_symbol_map(rows)
        contract = contracts.get(normalized_symbol)
        if contract is not None:
            self._contract_cache[normalized_symbol] = contract
        return contract

    async def _bitget_request(
        self,
        method: str,
        path: str,
        *,
        mode: LiveMode,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = str(int(time.time() * 1000))
        query = urlencode(params or {})
        body_text = json.dumps(body or {}, separators=(",", ":")) if body is not None else ""
        request_path = path if not query else f"{path}?{query}"
        message = f"{timestamp}{method.upper()}{request_path}{body_text}"
        signature = base64.b64encode(
            hmac.new(
                self.credentials.api_secret.encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        headers = {
            "ACCESS-KEY": self.credentials.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.credentials.passphrase,
            "locale": self.locale,
            "Content-Type": "application/json",
        }
        if mode == "demo":
            headers["paptrading"] = "1"

        async with httpx.AsyncClient(timeout=15.0) as client:
            if method.upper() == "GET":
                response = await client.get(f"{self.base_url}{path}", headers=headers, params=params)
            else:
                response = await client.post(f"{self.base_url}{path}", headers=headers, content=body_text)
        if response.status_code >= 400:
            return {"code": f"HTTP_{response.status_code}", "msg": response.text[:200]}
        return response.json()

    @staticmethod
    def _normalize_size(quantity: float, contract: dict[str, Any]) -> str | None:
        try:
            requested = Decimal(str(quantity))
            lot = Decimal(str(contract.get("lotSz") or "0.001"))
            min_trade = Decimal(str(contract.get("minTradeNum") or contract.get("lotSz") or "0.001"))
        except (InvalidOperation, TypeError, ValueError):
            return None

        if requested <= 0 or lot <= 0 or min_trade <= 0:
            return None
        normalized = (requested / lot).to_integral_value(rounding=ROUND_DOWN) * lot
        if normalized <= 0 or normalized < min_trade:
            return None
        return LiveExecutionAdapter._decimal_to_string(normalized)

    @staticmethod
    def _position_size(row: dict[str, Any]) -> Decimal:
        for key in ("total", "available", "openDelegateSize"):
            raw = row.get(key)
            if raw in (None, "", "0", 0, 0.0):
                continue
            try:
                return abs(Decimal(str(raw)))
            except InvalidOperation:
                continue
        return Decimal("0")

    @staticmethod
    def _as_rows(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for key in ("list", "rows", "entrustedList", "orderList"):
                value = data.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [data]
        return []

    @staticmethod
    def _extract_error_reason(response: dict[str, Any]) -> str:
        if response.get("msg"):
            return str(response["msg"])
        data = response.get("data")
        if isinstance(data, dict):
            for key in ("errorMsg", "message", "msg"):
                if data.get(key):
                    return str(data[key])
        if isinstance(data, list) and data and isinstance(data[0], dict):
            for key in ("errorMsg", "message", "msg"):
                if data[0].get(key):
                    return str(data[0][key])
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

    @staticmethod
    def _client_order_id(intent: OrderIntent) -> str:
        raw = str(intent.client_order_id or intent.order_intent_id or "")[:32]
        return raw or f"live-{intent.order_intent_id[:27]}"

    @staticmethod
    def _decimal_to_string(value: Decimal) -> str:
        normalized = value.normalize()
        text = format(normalized, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text


__all__ = [
    "BITGET_MARGIN_COIN",
    "BITGET_MARGIN_MODE",
    "BITGET_PRODUCT_TYPE",
    "BITGET_REST_BASE",
    "BitgetCredentials",
    "LiveExecutionAdapter",
    "LiveMode",
]
