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

# ─────────────────────────────────────────────────────────────
# Shared HTTP client for Bitget private REST.
# Was: every _bitget_request() built a NEW httpx.AsyncClient, doing a
# fresh TCP/TLS handshake. Under Windows socket limits + high concurrency
# (watcher reconcile + replan + user polls), this hit the socket-exhaust
# wall and blocked the whole event loop for 15-30s per call. The fix is
# a single lazily-created client with a big connection pool, short
# connect timeout (Bitget is fast when it's fast; slow = failed),
# moderate read timeout, no keep-alive starvation.
# ─────────────────────────────────────────────────────────────
_bitget_client: httpx.AsyncClient | None = None


def _get_bitget_client() -> httpx.AsyncClient:
    global _bitget_client
    if _bitget_client is None or _bitget_client.is_closed:
        _bitget_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=2.0),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            http2=False,  # Bitget doesn't benefit from h2 here + simpler failure modes
        )
    return _bitget_client

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
        order_type = getattr(intent, "order_type", "market") or "market"
        body: dict[str, Any] = {
            "symbol": intent.symbol.upper(),
            "productType": self.product_type,
            "marginMode": self.margin_mode,
            "marginCoin": self.margin_coin,
            "side": side,
            "tradeSide": "open",
            "orderType": order_type,
            "size": normalized_size,
            "clientOid": self._client_order_id(intent),
        }
        # For limit orders, include the price
        if order_type == "limit":
            limit_price = self._normalize_price(reference_price, contract)
            if limit_price is None:
                return self._error_result("price_precision_error", mode=mode, intent=intent)
            body["price"] = limit_price
            if getattr(intent, "post_only", False):
                body["force"] = "post_only"

        # Attach preset SL/TP so when the entry fills, Bitget auto-creates
        # plan orders to close the position. Without this, the user has to
        # watch fills manually — which defeats the whole "set and forget"
        # idea of the line-based order workflow.
        if intent.stop_price and intent.stop_price > 0:
            sl = self._normalize_price(float(intent.stop_price), contract)
            if sl:
                body["presetStopLossPrice"] = sl
        if intent.tp_price and intent.tp_price > 0:
            tp = self._normalize_price(float(intent.tp_price), contract)
            if tp:
                body["presetStopSurplusPrice"] = tp

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
                "request_body_excerpt": {
                    "orderType": body["orderType"],
                    "price": body.get("price"),
                    "force": body.get("force"),
                    "presetStopLossPrice": body.get("presetStopLossPrice"),
                    "presetStopSurplusPrice": body.get("presetStopSurplusPrice"),
                    "size": body["size"],
                },
            }
        return self._error_result(
            self._extract_error_reason(response),
            mode=mode,
            intent=intent,
            response=response,
        )

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        mode: LiveMode,
    ) -> dict[str, Any]:
        """Cancel a regular Bitget order from the open-orders book."""
        normalized_symbol = symbol.upper().replace("/", "")
        oid = str(order_id or "")
        if not oid:
            return {
                "ok": False,
                "symbol": normalized_symbol,
                "order_id": oid,
                "reason": "missing_order_id",
            }
        body = {
            "symbol": normalized_symbol,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "orderId": oid,
        }
        response = await self._bitget_request(
            "POST", "/api/v2/mix/order/cancel-order", mode=mode, body=body,
        )
        if response.get("code") == "00000":
            return {
                "ok": True,
                "symbol": normalized_symbol,
                "order_id": oid,
                "exchange_response_excerpt": self._excerpt(response),
            }
        return {
            "ok": False,
            "symbol": normalized_symbol,
            "order_id": oid,
            "reason": self._extract_error_reason(response),
            "exchange_response_excerpt": self._excerpt(response),
        }

    async def submit_live_plan_entry(
        self,
        intent: OrderIntent,
        mode: LiveMode,
        trigger_price: float,
        trigger_type: str = "mark_price",  # 'mark_price' or 'fill_price'
    ) -> dict[str, Any]:
        """Submit a TRIGGER (plan) order to Bitget.

        Unlike submit_live_entry which sends an immediate limit/market
        order, this places a plan order that *waits* on the exchange until
        the trigger price is hit, then fires a limit entry. SL and TP
        presets are attached so when the entry fills, Bitget auto-arms
        the close orders.
        """
        if not self.has_api_keys():
            return self._error_result("api_keys_missing", mode=mode, intent=intent)

        contract = await self._get_contract(intent.symbol)
        if not contract:
            return self._error_result("contract_not_found", mode=mode, intent=intent)

        normalized_size = self._normalize_size(intent.quantity, contract)
        if normalized_size is None:
            return self._error_result("size_below_min_trade", mode=mode, intent=intent)

        normalized_trigger = self._normalize_price(float(trigger_price), contract)
        if normalized_trigger is None:
            return self._error_result("trigger_price_precision_error", mode=mode, intent=intent)

        side = "buy" if intent.side == "long" else "sell"
        body: dict[str, Any] = {
            "symbol": intent.symbol.upper(),
            "productType": self.product_type,
            "marginMode": self.margin_mode,
            "marginCoin": self.margin_coin,
            "planType": "normal_plan",
            "size": normalized_size,
            "side": side,
            "tradeSide": "open",
            "orderType": "limit",
            "triggerPrice": normalized_trigger,
            "executePrice": normalized_trigger,
            "price": normalized_trigger,
            "triggerType": trigger_type,
            "clientOid": self._client_order_id(intent),
        }
        # Preset SL/TP — position protected from moment of fill
        if intent.stop_price and intent.stop_price > 0:
            sl = self._normalize_price(float(intent.stop_price), contract)
            if sl:
                body["stopLossTriggerPrice"] = sl
                body["stopLossTriggerType"] = "mark_price"
        if intent.tp_price and intent.tp_price > 0:
            tp = self._normalize_price(float(intent.tp_price), contract)
            if tp:
                body["stopSurplusTriggerPrice"] = tp
                body["stopSurplusTriggerType"] = "mark_price"

        response = await self._bitget_request(
            "POST", "/api/v2/mix/order/place-plan-order", mode=mode, body=body,
        )
        if response.get("code") == "00000":
            data = response.get("data") or {}
            submitted_notional = float(Decimal(normalized_size) * Decimal(str(trigger_price)))
            return {
                "ok": True,
                "mode": mode,
                "symbol": intent.symbol.upper(),
                "side": intent.side,
                "exchange_order_id": str(data.get("orderId") or data.get("clientOid") or ""),
                "submitted_price": float(trigger_price),
                "submitted_notional": submitted_notional,
                "submitted_order_type": body["orderType"],
                "submitted_trigger_type": trigger_type,
                "submitted_limit_price": normalized_trigger,
                "exchange_response_excerpt": data,
                "request_body_excerpt": {
                    "orderType": body["orderType"],
                    "triggerPrice": body["triggerPrice"],
                    "executePrice": body["executePrice"],
                    "price": body["price"],
                    "stopLossTriggerPrice": body.get("stopLossTriggerPrice"),
                    "stopSurplusTriggerPrice": body.get("stopSurplusTriggerPrice"),
                    "size": body["size"],
                },
            }
        return self._error_result(
            self._extract_error_reason(response),
            mode=mode,
            intent=intent,
            response=response,
        )

    async def cancel_plan_order(
        self,
        symbol: str,
        order_id: str,
        mode: LiveMode,
        *,
        plan_type: str = "normal_plan",
    ) -> dict[str, Any]:
        """Cancel a single Bitget plan order and require successList evidence.

        Bitget returns code=00000 with an empty successList when planType is
        missing or wrong. Treat that as a failed cancel so callers do not place
        replacement orders over still-live originals.
        """
        normalized_symbol = symbol.upper().replace("/", "")
        oid = str(order_id or "")
        if not oid:
            return {
                "ok": False,
                "symbol": normalized_symbol,
                "plan_type": plan_type,
                "reason": "missing_order_id",
            }
        body = {
            "symbol": normalized_symbol,
            "productType": self.product_type,
            "marginCoin": self.margin_coin,
            "orderId": oid,
            "planType": plan_type,
        }
        response = await self._bitget_request(
            "POST", "/api/v2/mix/order/cancel-plan-order",
            mode=mode,
            body=body,
        )
        success_ids = self._success_order_ids(response)
        if response.get("code") == "00000" and oid in success_ids:
            return {
                "ok": True,
                "symbol": normalized_symbol,
                "order_id": oid,
                "plan_type": plan_type,
                "exchange_response_excerpt": self._excerpt(response),
            }
        reason = self._extract_error_reason(response)
        if response.get("code") == "00000" and not success_ids:
            reason = "empty_successList"
        return {
            "ok": False,
            "symbol": normalized_symbol,
            "order_id": oid,
            "plan_type": plan_type,
            "reason": reason,
            "exchange_response_excerpt": self._excerpt(response),
        }

    async def cancel_plan_order_any_type(
        self,
        symbol: str,
        order_id: str,
        mode: LiveMode,
        *,
        plan_types: tuple[str, ...] = ("normal_plan", "pos_loss", "loss_plan", "pos_profit", "profit_plan"),
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        for plan_type in plan_types:
            result = await self.cancel_plan_order(symbol, order_id, mode, plan_type=plan_type)
            attempts.append(result)
            if result.get("ok"):
                result["attempts"] = attempts
                return result
        return {
            "ok": False,
            "symbol": symbol.upper().replace("/", ""),
            "order_id": str(order_id or ""),
            "reason": "cancel_plan_order_all_types_failed",
            "attempts": attempts,
        }

    async def get_pending_plan_orders(
        self,
        mode: LiveMode,
        *,
        plan_type: str = "normal_plan",
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return Bitget pending plan orders for exchange-truth sync."""
        params: dict[str, str] = {
            "productType": self.product_type,
            "planType": plan_type,
        }
        if symbol:
            params["symbol"] = symbol.upper().replace("/", "")
        response = await self._bitget_request(
            "GET",
            "/api/v2/mix/order/orders-plan-pending",
            mode=mode,
            body=None,
            params=params,
        )
        if response.get("code") != "00000":
            raise RuntimeError(f"plan pending fetch failed: {self._extract_error_reason(response)}")
        rows = []
        for row in self._as_rows(response.get("data")):
            order_id = row.get("orderId") or row.get("order_id")
            row_symbol = str(row.get("symbol") or "").upper()
            if not order_id:
                continue
            if symbol and row_symbol and row_symbol != symbol.upper().replace("/", ""):
                continue
            rows.append(row)
        return rows

    async def get_pending_orders(
        self,
        mode: LiveMode,
        *,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return Bitget regular pending orders for exchange-truth sync."""
        params: dict[str, str] = {"productType": self.product_type}
        response = await self._bitget_request(
            "GET",
            "/api/v2/mix/order/orders-pending",
            mode=mode,
            body=None,
            params=params,
        )
        if response.get("code") != "00000":
            raise RuntimeError(f"pending order fetch failed: {self._extract_error_reason(response)}")
        rows = []
        normalized_symbol = symbol.upper().replace("/", "") if symbol else ""
        for row in self._as_rows(response.get("data")):
            order_id = row.get("orderId") or row.get("order_id")
            row_symbol = str(row.get("symbol") or "").upper()
            if not order_id:
                continue
            if normalized_symbol and row_symbol and row_symbol != normalized_symbol:
                continue
            rows.append(row)
        return rows

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

    async def update_position_sl_tp(
        self, symbol: str, hold_side: str,
        new_sl: float | None = None, new_tp: float | None = None,
        mode: LiveMode = "live",
    ) -> dict[str, Any]:
        """Move SL and/or TP on an existing position.

        Strategy: cancel existing SL/TP plan orders, then place new ones
        via /api/v2/mix/order/place-tpsl-order.
        """
        normalized_symbol = symbol.upper().replace("/", "")
        if not self.has_api_keys():
            return {"ok": False, "reason": "api_keys_missing"}

        contract = await self._get_contract(normalized_symbol)
        if not contract:
            return {"ok": False, "reason": "contract_not_found"}

        normalized_new_sl = self._normalize_price(float(new_sl), contract) if new_sl is not None and new_sl > 0 else None
        normalized_new_tp = self._normalize_price(float(new_tp), contract) if new_tp is not None and new_tp > 0 else None
        if new_sl is not None and new_sl > 0 and normalized_new_sl is None:
            return {"ok": False, "reason": "sl_price_precision_error"}
        if new_tp is not None and new_tp > 0 and normalized_new_tp is None:
            return {"ok": False, "reason": "tp_price_precision_error"}

        entry_price = await self._get_position_entry_price(normalized_symbol, hold_side, mode)
        cancelled: list[str] = []

        # Cancel ALL existing SL orders, leave TP untouched.
        # Key: cancel body MUST include the correct planType or it silently fails.
        # Bitget sub-types: pos_loss (仓位止损), loss_plan (部分止损/preset SL)
        try:
            # Query all SL/TP under "profit_loss" umbrella (returns pos_loss, pos_profit, loss_plan, profit_plan)
            pending = await self._bitget_request(
                "GET", "/api/v2/mix/order/orders-plan-pending",
                mode=mode, body=None,
                params={"symbol": normalized_symbol,
                        "productType": self.product_type,
                        "planType": "profit_loss"},
            )
            for order in ((pending.get("data") or {}).get("entrustedList") or []):
                oid = order.get("orderId")
                trigger = float(order.get("triggerPrice") or 0)
                actual_plan_type = order.get("planType", "")

                # Identify SL vs TP by trigger direction against entry
                is_sl = False
                if entry_price and entry_price > 0 and trigger > 0:
                    if hold_side == "short":
                        is_sl = trigger > entry_price
                    elif hold_side == "long":
                        is_sl = trigger < entry_price
                # Also catch by planType name
                if actual_plan_type in ("pos_loss", "loss_plan"):
                    is_sl = True

                if is_sl and oid:
                    cancel_resp = await self.cancel_plan_order(
                        normalized_symbol,
                        str(oid),
                        mode,
                        plan_type=actual_plan_type or "pos_loss",
                    )
                    if cancel_resp.get("ok"):
                        cancelled.append(str(oid))
                        print(f"[live_adapter] cancelled {actual_plan_type} SL {trigger} for {normalized_symbol}", flush=True)
                    else:
                        print(f"[live_adapter] cancel SL {trigger} empty successList: {cancel_resp}", flush=True)
        except Exception as e:
            print(f"[live_adapter] SL cancel error: {e}", flush=True)

        results = []
        actual_sl_after: float | None = None
        sl_verified: bool | None = None
        # Place new SL
        if normalized_new_sl is not None:
            try:
                resp = await self._bitget_request(
                    "POST", "/api/v2/mix/order/place-tpsl-order",
                    mode=mode, body={
                        "symbol": normalized_symbol,
                        "productType": self.product_type,
                        "marginCoin": "USDT",
                        "planType": "pos_loss",
                        "triggerPrice": normalized_new_sl,
                        "triggerType": "mark_price",
                        "holdSide": hold_side,
                    },
                )
                if resp.get("code") == "00000":
                    results.append("sl_ok")
                    actual_sl_after = await self.get_position_sl_trigger_price(
                        normalized_symbol,
                        hold_side,
                        mode,
                        entry_price=entry_price,
                    )
                    expected_sl = float(normalized_new_sl)
                    sl_verified = (
                        actual_sl_after is not None
                        and abs(actual_sl_after - expected_sl) <= max(abs(expected_sl) * 1e-8, 1e-8)
                    )
                    if not sl_verified:
                        print(
                            f"[live_adapter] SL verify mismatch {normalized_symbol}: "
                            f"expected={normalized_new_sl} actual={actual_sl_after}",
                            flush=True,
                        )
                else:
                    return {"ok": False, "reason": f"SL: {self._extract_error_reason(resp)}"}
            except Exception as e:
                return {"ok": False, "reason": f"SL: {e}"}

        # Place TP only on first call (when no TP exists yet)
        if normalized_new_tp is not None:
            try:
                resp = await self._bitget_request(
                    "POST", "/api/v2/mix/order/place-tpsl-order",
                    mode=mode, body={
                        "symbol": normalized_symbol,
                        "productType": self.product_type,
                        "marginCoin": "USDT",
                        "planType": "pos_profit",
                        "triggerPrice": normalized_new_tp,
                        "triggerType": "mark_price",
                        "holdSide": hold_side,
                    },
                )
                if resp.get("code") == "00000":
                    results.append("tp_ok")
                else:
                    return {"ok": False, "reason": f"TP: {self._extract_error_reason(resp)}"}
            except Exception as e:
                return {"ok": False, "reason": f"TP: {e}"}

        return {
            "ok": True,
            "symbol": normalized_symbol,
            "updates": results,
            "cancelled_order_ids": cancelled,
            "new_sl": normalized_new_sl,
            "new_tp": normalized_new_tp,
            "actual_sl_after": actual_sl_after,
            "sl_verified": sl_verified,
        }

    async def get_position_sl_trigger_price(
        self,
        symbol: str,
        hold_side: str,
        mode: LiveMode,
        *,
        entry_price: float | None = None,
    ) -> float | None:
        """Return the tightest current SL trigger price for a position."""
        normalized_symbol = symbol.upper().replace("/", "")
        if entry_price is None or entry_price <= 0:
            entry_price = await self._get_position_entry_price(normalized_symbol, hold_side, mode)
        pending = await self._bitget_request(
            "GET", "/api/v2/mix/order/orders-plan-pending",
            mode=mode,
            body=None,
            params={
                "symbol": normalized_symbol,
                "productType": self.product_type,
                "planType": "profit_loss",
            },
        )
        if pending.get("code") != "00000":
            print(f"[live_adapter] SL verify fetch failed: {self._extract_error_reason(pending)}", flush=True)
            return None

        sl_prices: list[float] = []
        for order in ((pending.get("data") or {}).get("entrustedList") or []):
            plan_type = str(order.get("planType") or "")
            try:
                trigger = float(order.get("triggerPrice") or 0)
            except (TypeError, ValueError):
                trigger = 0.0
            if trigger <= 0:
                continue
            is_sl = plan_type in ("pos_loss", "loss_plan")
            if not is_sl and entry_price and entry_price > 0:
                if hold_side == "short":
                    is_sl = trigger > entry_price
                elif hold_side == "long":
                    is_sl = trigger < entry_price
            if is_sl:
                sl_prices.append(trigger)
        if not sl_prices:
            return None
        if hold_side == "short":
            return min(sl_prices)
        return max(sl_prices)

    async def _get_position_entry_price(
        self,
        normalized_symbol: str,
        hold_side: str,
        mode: LiveMode,
    ) -> float | None:
        resp = await self._bitget_request(
            "GET", "/api/v2/mix/position/all-position",
            mode=mode,
            params={"productType": self.product_type, "marginCoin": self.margin_coin},
        )
        if resp.get("code") != "00000":
            print(f"[live_adapter] position entry fetch failed: {self._extract_error_reason(resp)}", flush=True)
            return None
        for row in self._as_rows(resp.get("data")):
            symbol = str(row.get("symbol") or "").upper()
            side = str(row.get("holdSide") or row.get("posSide") or "").lower()
            if symbol != normalized_symbol or side != hold_side:
                continue
            if self._position_size(row) <= 0:
                continue
            try:
                entry = float(row.get("openPriceAvg") or row.get("averageOpenPrice") or row.get("openAvgPrice") or 0)
            except (TypeError, ValueError):
                entry = 0.0
            if entry > 0:
                return entry
        return None

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
                "openPriceAvg": row.get("openPriceAvg"),
                "averageOpenPrice": row.get("openPriceAvg") or row.get("averageOpenPrice") or row.get("openAvgPrice"),
                "openAvgPrice": row.get("openAvgPrice"),
                "openTime": row.get("openTime") or row.get("openTimestamp") or row.get("cTime") or row.get("ctime") or row.get("createdTime") or row.get("createTime"),
                "openTimestamp": row.get("openTimestamp"),
                "cTime": row.get("cTime") or row.get("ctime"),
                "createdTime": row.get("createdTime") or row.get("createTime"),
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
                # Bitget uses different field names depending on order type:
                #   regular limit: "price"
                #   plan / trigger: "triggerPrice" / "executePrice"
                "price": row.get("price") or row.get("executePrice")
                          or row.get("triggerPrice") or row.get("priceAvg"),
            }
            for row in pending_rows
            # Filter Bitget's all-null ghost rows
            if row.get("symbol") and row.get("orderId")
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

    async def get_open_position_symbols(self, mode: LiveMode) -> set[str]:
        """Return symbols with non-zero open futures positions."""
        response = await self._bitget_request(
            "GET", "/api/v2/mix/position/all-position",
            mode=mode,
            params={"productType": self.product_type, "marginCoin": self.margin_coin},
        )
        if response.get("code") != "00000":
            raise RuntimeError(f"position fetch failed: {self._extract_error_reason(response)}")
        symbols: set[str] = set()
        for row in self._as_rows(response.get("data")):
            if self._position_size(row) <= 0:
                continue
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                symbols.add(symbol)
        return symbols

    async def has_open_position(self, symbol: str, mode: LiveMode) -> bool:
        return symbol.upper().replace("/", "") in await self.get_open_position_symbols(mode)

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

        client = _get_bitget_client()
        try:
            if method.upper() == "GET":
                response = await client.get(f"{self.base_url}{path}", headers=headers, params=params)
            else:
                response = await client.post(f"{self.base_url}{path}", headers=headers, content=body_text)
        except httpx.TimeoutException as exc:
            # Fail fast — a Bitget hang must NEVER block the whole event loop.
            # The caller (watcher reconcile / user endpoint) will see a clear
            # error and move on instead of waiting 30s per attempt.
            return {"code": "TIMEOUT", "msg": f"bitget timeout: {exc!s}"[:160]}
        except httpx.HTTPError as exc:
            return {"code": "HTTP_ERROR", "msg": f"bitget http err: {exc!s}"[:160]}
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
    def _normalize_price(price: float, contract: dict[str, Any]) -> str | None:
        """Normalize price to contract tick size for limit orders."""
        try:
            requested = Decimal(str(price))
            tick_raw = contract.get("tickSz")
            if tick_raw in (None, ""):
                price_place_raw = contract.get("pricePlace")
                price_end_step_raw = contract.get("priceEndStep") or "1"
                price_place_text = "" if price_place_raw in (None, "") else str(price_place_raw)
                if "." in price_place_text:
                    # Backward-compatible tests pass pricePlace as an already
                    # normalized tick. Raw Bitget contracts use integer
                    # decimals plus priceEndStep.
                    tick = Decimal(price_place_text)
                elif price_place_text:
                    tick = Decimal(str(price_end_step_raw)) / (Decimal(10) ** int(price_place_text))
                else:
                    tick = Decimal("0.01")
            else:
                tick = Decimal(str(tick_raw))
            if tick <= 0:
                tick = Decimal("0.01")
        except (InvalidOperation, TypeError, ValueError):
            return None
        if requested <= 0:
            return None
        normalized = (requested / tick).to_integral_value(rounding=ROUND_DOWN) * tick
        if normalized <= 0:
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
    def _success_order_ids(response: dict[str, Any]) -> set[str]:
        data = response.get("data")
        if not isinstance(data, dict):
            return set()
        raw_items = data.get("successList") or []
        ids: set[str] = set()
        for item in raw_items:
            if isinstance(item, dict):
                raw = item.get("orderId") or item.get("order_id")
            else:
                raw = item
            if raw not in (None, ""):
                ids.add(str(raw))
        return ids

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
