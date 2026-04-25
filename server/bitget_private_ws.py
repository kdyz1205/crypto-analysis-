"""Bitget private WebSocket subscriber — pushes order/account updates
directly to the event bus so the UI sees fills in <200ms instead of
waiting for the 10–30s poll.

SAFETY PHILOSOPHY (2026-04-23):
    This is ADDITIVE. It does not replace any polling. If the WS fails,
    hangs, or disconnects, the existing /api/live-execution/account and
    /api/conditionals polls still catch every state change — just slower.
    So execution correctness is preserved regardless of WS health.

    Disabled by default (env BITGET_PRIVATE_WS_ENABLED=0). Set to 1 in
    .env after you've verified one or two real orders arrive via WS
    events in the browser console.

Protocol: Bitget v2 private WS
    URL: wss://ws.bitget.com/v2/ws/private
    Login: {"op":"login","args":[{apiKey, passphrase, timestamp, sign}]}
    Sign: HMAC-SHA256(secret, timestamp + "GET" + "/user/verify") base64
    Subscribe: {"op":"subscribe","args":[{instType, channel}]}
    Channels used: orders (plan + entry fills), account (balance/margin)

Events published:
    order.ws_update    - one incoming order frame (entry, cancel, fill)
    account.ws_update  - one incoming account frame (equity change)
The frontend already listens for order.* via stream.js — no frontend
changes required for the event to flow into the conditional_panel.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from .core.events import bus, Event

_BITGET_PRIVATE_WS = "wss://ws.bitget.com/v2/ws/private"
_LOGIN_TIMEOUT_SEC = 10
_PING_INTERVAL_SEC = 20
_RECONNECT_INITIAL_SEC = 2
_RECONNECT_MAX_SEC = 60

_task: asyncio.Task | None = None
_stopped = False
_stats = {"connects": 0, "orders": 0, "accounts": 0, "errors": 0}


def _sign(secret: str, timestamp: str) -> str:
    """Bitget v2 private WS login signature.

    Payload: timestamp + 'GET' + '/user/verify'
    Algo:    HMAC-SHA256, then base64-encoded (not hex).
    """
    payload = f"{timestamp}GET/user/verify"
    mac = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode("utf-8")


def _credentials() -> tuple[str, str, str] | None:
    api_key = os.environ.get("BITGET_API_KEY", "")
    secret = (
        os.environ.get("BITGET_SECRET_KEY", "")
        or os.environ.get("BITGET_SECRET", "")
        or os.environ.get("BITGET_API_SECRET", "")
    )
    passphrase = os.environ.get("BITGET_PASSPHRASE", "")
    if not (api_key and secret and passphrase):
        return None
    return api_key, secret, passphrase


async def _subscribe_once(ws) -> None:
    """Subscribe to orders + account + ordersAlgo channels after login.

    Bitget v2 channel subscribe args (verified empirically 2026-04-25):
        - orders        → requires instId (use 'default' for all symbols)
        - account       → requires coin (use 'default' for all)
        - positions     → requires instId
        - ordersAlgo    → requires NO instId; instId='default' returns
                          code 30016 "Param error". Plan-orders (trigger
                          market / TPSL / trailing) push through this
                          channel only.
    """
    sub_msg = {
        "op": "subscribe",
        "args": [
            {"instType": "USDT-FUTURES", "channel": "orders", "instId": "default"},
            {"instType": "USDT-FUTURES", "channel": "account", "coin": "default"},
            {"instType": "USDT-FUTURES", "channel": "positions", "instId": "default"},
            # ordersAlgo: NO instId (Bitget rejects 'default' there)
            {"instType": "USDT-FUTURES", "channel": "ordersAlgo"},
        ],
    }
    await ws.send(json.dumps(sub_msg))


async def _handle_frame(msg: dict) -> None:
    """Dispatch one incoming WS frame to the event bus."""
    arg = msg.get("arg") or {}
    channel = arg.get("channel")
    data = msg.get("data") or []
    if not channel or not data:
        return
    if channel == "orders" or channel == "ordersAlgo":
        _stats["orders"] += 1
        for row in data:
            # Brief log so user can see WS push activity in server log
            sym = row.get("symbol") or row.get("instId") or "?"
            status = row.get("status") or "?"
            oid = row.get("orderId") or "?"
            print(f"[bitget_private_ws] {channel} push: {sym} status={status} oid={oid}", flush=True)
            await bus.publish(Event(
                type="order.ws_update",
                payload={"channel": channel, "data": row},
            ))
    elif channel == "account":
        _stats["accounts"] += 1
        for row in data:
            await bus.publish(Event(
                type="account.ws_update",
                payload={"channel": channel, "data": row},
            ))
    elif channel == "positions":
        for row in data:
            await bus.publish(Event(
                type="position.ws_update",
                payload={"channel": channel, "data": row},
            ))


async def _run_once() -> None:
    """Connect, log in, subscribe, read until error/close."""
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(f"websockets lib missing: {exc}") from exc
    creds = _credentials()
    if not creds:
        raise RuntimeError("API keys not set in .env (BITGET_API_KEY / SECRET / PASSPHRASE)")
    api_key, secret, passphrase = creds

    async with websockets.connect(
        _BITGET_PRIVATE_WS,
        ping_interval=_PING_INTERVAL_SEC,
        ping_timeout=10,
    ) as ws:
        _stats["connects"] += 1
        # Login
        ts = str(int(time.time()))
        login_msg = {
            "op": "login",
            "args": [{
                "apiKey": api_key,
                "passphrase": passphrase,
                "timestamp": ts,
                "sign": _sign(secret, ts),
            }],
        }
        await ws.send(json.dumps(login_msg))

        logged_in = False
        login_deadline = time.time() + _LOGIN_TIMEOUT_SEC
        async for raw in ws:
            if _stopped:
                return
            if raw == "pong":
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            # Login ack
            if msg.get("event") == "login":
                if msg.get("code") == "0" or msg.get("code") == 0:
                    logged_in = True
                    await _subscribe_once(ws)
                    print("[bitget_private_ws] logged in + subscribed")
                else:
                    raise RuntimeError(f"login failed: {msg}")
                continue

            # Login timeout guard
            if not logged_in:
                if time.time() > login_deadline:
                    raise RuntimeError("login timeout")
                continue

            # Subscribe ack (ignore)
            if msg.get("event") == "subscribe":
                continue

            # Error
            if msg.get("event") == "error":
                _stats["errors"] += 1
                print(f"[bitget_private_ws] error frame: {msg}")
                continue

            # Data frame
            await _handle_frame(msg)


async def _supervisor() -> None:
    """Reconnect loop with exponential backoff, until stop_ws() fires."""
    backoff = _RECONNECT_INITIAL_SEC
    while not _stopped:
        try:
            await _run_once()
            backoff = _RECONNECT_INITIAL_SEC   # reset on clean close
        except Exception as exc:
            _stats["errors"] += 1
            print(f"[bitget_private_ws] failure: {exc}; reconnecting in {backoff}s")
            # Sleep in small increments so stop_ws() is responsive.
            slept = 0
            while slept < backoff and not _stopped:
                await asyncio.sleep(1)
                slept += 1
            backoff = min(int(backoff * 1.5), _RECONNECT_MAX_SEC)


def start_ws() -> None:
    """Fire-and-forget: start the background supervisor."""
    global _task, _stopped
    if _task is not None and not _task.done():
        return
    if os.environ.get("BITGET_PRIVATE_WS_ENABLED", "0") != "1":
        print("[bitget_private_ws] disabled (set BITGET_PRIVATE_WS_ENABLED=1 in .env)")
        return
    if _credentials() is None:
        print("[bitget_private_ws] disabled (missing API credentials)")
        return
    _stopped = False
    _task = asyncio.create_task(_supervisor())
    print("[bitget_private_ws] started")


def stop_ws() -> None:
    global _stopped
    _stopped = True


def ws_stats() -> dict[str, int]:
    return dict(_stats)
