"""Bitget WebSocket feed — real-time order book + trades.

Connects to wss://ws.bitget.com/v2/ws/public for:
- books{depth} channel: incremental order book updates
- trade channel: real-time trade prints

Event-driven: calls back on every update, no polling delay.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Coroutine

import websockets

BITGET_WS_PUBLIC = "wss://ws.bitget.com/v2/ws/public"
BITGET_WS_PRIVATE = "wss://ws.bitget.com/v2/ws/private"


class BitgetWebSocketFeed:
    """Real-time WebSocket feed from Bitget."""

    def __init__(
        self,
        symbol: str = "HYPEUSDT",
        product_type: str = "USDT-FUTURES",
        on_book: Callable[[dict], Coroutine] | None = None,
        on_trade: Callable[[dict], Coroutine] | None = None,
        on_ticker: Callable[[dict], Coroutine] | None = None,
    ):
        self.symbol = symbol
        self.product_type = product_type
        self.on_book = on_book
        self.on_trade = on_trade
        self.on_ticker = on_ticker
        self._ws = None
        self._running = False
        self._reconnect_delay = 1
        self._last_ping = 0
        self.stats = {"book_updates": 0, "trade_updates": 0, "ticker_updates": 0, "reconnects": 0}

    async def start(self):
        """Connect and start receiving data. Reconnects on failure."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                if not self._running:
                    break
                self.stats["reconnects"] += 1
                print(f"[ws] disconnected: {e}, reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect_and_listen(self):
        async with websockets.connect(BITGET_WS_PUBLIC, ping_interval=20, ping_timeout=10) as ws:
            self._ws = ws
            self._reconnect_delay = 1
            print(f"[ws] connected to {BITGET_WS_PUBLIC}")

            # Subscribe to channels
            await self._subscribe(ws)

            # Listen for messages
            async for raw in ws:
                if not self._running:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Handle pong
                if msg.get("event") == "pong":
                    continue

                # Handle subscription confirmation
                if msg.get("event") == "subscribe":
                    print(f"[ws] subscribed: {msg.get('arg', {}).get('channel', '?')}")
                    continue

                if msg.get("event") == "error":
                    print(f"[ws] error: {msg}")
                    continue

                # Handle data
                action = msg.get("action")
                data = msg.get("data")
                arg = msg.get("arg", {})
                channel = arg.get("channel", "")

                if not data:
                    continue

                if "books" in channel:
                    self.stats["book_updates"] += 1
                    if self.on_book:
                        await self.on_book({"action": action, "data": data, "ts": msg.get("ts", "")})

                elif channel == "trade":
                    self.stats["trade_updates"] += 1
                    if self.on_trade:
                        await self.on_trade({"data": data, "ts": msg.get("ts", "")})

                elif channel == "ticker":
                    self.stats["ticker_updates"] += 1
                    if self.on_ticker:
                        await self.on_ticker({"data": data, "ts": msg.get("ts", "")})

                # Periodic ping
                now = time.time()
                if now - self._last_ping > 25:
                    await ws.send("ping")
                    self._last_ping = now

    async def _subscribe(self, ws):
        """Subscribe to order book, trades, and ticker."""
        subs = {
            "op": "subscribe",
            "args": [
                {
                    "instType": self.product_type,
                    "channel": "books5",  # top 5 levels, fast updates
                    "instId": self.symbol,
                },
                {
                    "instType": self.product_type,
                    "channel": "trade",
                    "instId": self.symbol,
                },
                {
                    "instType": self.product_type,
                    "channel": "ticker",
                    "instId": self.symbol,
                },
            ],
        }
        await ws.send(json.dumps(subs))
        print(f"[ws] subscribing to books5 + trade + ticker for {self.symbol}")


class TradeAccumulator:
    """Accumulates trade prints for flow analysis."""

    def __init__(self, window_ms: int = 5000):
        self.window_ms = window_ms
        self.trades: list[dict] = []  # {ts, price, size, side}

    def add(self, trade_data: list[dict]):
        """Add new trades from WebSocket."""
        now = time.time() * 1000
        for t in trade_data:
            self.trades.append({
                "ts": float(t.get("ts", now)),
                "price": float(t.get("price", 0)),
                "size": float(t.get("size", t.get("baseVolume", 0))),
                "side": t.get("side", ""),
            })
        # Prune old trades
        cutoff = now - self.window_ms
        self.trades = [t for t in self.trades if t["ts"] > cutoff]

    @property
    def buy_volume(self) -> float:
        return sum(t["size"] for t in self.trades if t["side"] == "buy")

    @property
    def sell_volume(self) -> float:
        return sum(t["size"] for t in self.trades if t["side"] == "sell")

    @property
    def trade_imbalance(self) -> float:
        bv, sv = self.buy_volume, self.sell_volume
        total = bv + sv
        return (bv - sv) / total if total > 0 else 0.0

    @property
    def flow_burst(self) -> float:
        """How intense recent flow is vs average."""
        if len(self.trades) < 5:
            return 0.0
        recent = [t for t in self.trades if t["ts"] > time.time() * 1000 - 1000]
        recent_vol = sum(t["size"] for t in recent)
        avg_vol = sum(t["size"] for t in self.trades) / max(len(self.trades), 1) * (1000 / self.window_ms * len(self.trades))
        return recent_vol / avg_vol if avg_vol > 0 else 0.0
