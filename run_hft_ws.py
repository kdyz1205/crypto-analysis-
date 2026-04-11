"""HFT Runner v2 — WebSocket-driven, event-based, no polling delay.

Uses Bitget WebSocket for real-time order book + trade stream.
Strategies react to every book/trade update, not on fixed intervals.

Usage: python run_hft_ws.py --symbol HYPEUSDT --mode live
"""

import asyncio
import json
import os
import sys
import time
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass

import httpx

from server.hft.data_feed.book_builder import BookBuilder
from server.hft.data_feed.features import compute_features
from server.hft.data_feed.ws_feed import BitgetWebSocketFeed, TradeAccumulator
from server.hft.router import RegimeRouter
from server.hft.risk import HFTKillSwitch


BITGET_REST = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"


class BitgetOrderClient:
    """REST client for order placement (WS is read-only for public data)."""

    def __init__(self, mode="live"):
        import base64, hashlib, hmac
        self.key = os.environ.get("BITGET_API_KEY", "")
        self.secret = os.environ.get("BITGET_SECRET_KEY", "") or os.environ.get("BITGET_SECRET", "")
        self.passphrase = os.environ.get("BITGET_PASSPHRASE", "")
        self.mode = mode
        self._b64, self._hmac, self._hash = base64, hmac, hashlib
        self._client = httpx.AsyncClient(timeout=3)

    def _headers(self, method, path, body=""):
        ts = str(int(time.time() * 1000))
        sig = self._b64.b64encode(self._hmac.new(self.secret.encode(), f"{ts}{method}{path}{body}".encode(), self._hash.sha256).digest()).decode()
        h = {"ACCESS-KEY": self.key, "ACCESS-SIGN": sig, "ACCESS-TIMESTAMP": ts, "ACCESS-PASSPHRASE": self.passphrase, "Content-Type": "application/json", "locale": "en-US"}
        if self.mode == "demo": h["paptrading"] = "1"
        return h

    async def place(self, sym, side, price, size, force="gtc"):
        path = "/api/v2/mix/order/place-order"
        body = json.dumps({"symbol": sym, "productType": PRODUCT_TYPE, "marginMode": "crossed", "marginCoin": MARGIN_COIN, "side": side, "orderType": "limit", "price": str(price), "size": str(size), "tradeSide": "open", "force": force})
        r = await self._client.post(f"{BITGET_REST}{path}", content=body, headers=self._headers("POST", path, body))
        return r.json()

    async def cancel_all(self, sym):
        path = "/api/v2/mix/order/cancel-all-orders"
        body = json.dumps({"symbol": sym, "productType": PRODUCT_TYPE, "marginCoin": MARGIN_COIN})
        r = await self._client.post(f"{BITGET_REST}{path}", content=body, headers=self._headers("POST", path, body))
        return r.json()

    async def net_position(self, sym) -> float:
        path = f"/api/v2/mix/position/single-position?productType={PRODUCT_TYPE}&symbol={sym}&marginCoin={MARGIN_COIN}"
        r = await self._client.get(f"{BITGET_REST}{path}", headers=self._headers("GET", path))
        d = r.json()
        pos = 0.0
        for p in (d.get("data") or []):
            if isinstance(p, dict):
                t = float(p.get("total", 0) or 0)
                pos += t if p.get("holdSide") == "long" else -t
        return pos

    async def close(self):
        await self._client.aclose()


class HFTEngine:
    """Event-driven HFT engine. Reacts to every WebSocket update."""

    def __init__(self, symbol: str, order_client: BitgetOrderClient, tick_size: float = 0.001):
        self.symbol = symbol
        self.client = order_client
        self.tick_size = tick_size
        self.builder = BookBuilder()
        self.trades = TradeAccumulator(window_ms=5000)
        self.router = RegimeRouter()
        self.kill = HFTKillSwitch()
        self.inventory = 0.0
        self.last_order_time = 0.0
        self.min_order_interval = 2.0  # min seconds between orders (avoid spam)
        self.start_time = time.time()
        self.update_count = 0
        self.order_count = 0

    async def on_book(self, msg: dict):
        """Called on every order book WebSocket update."""
        data = msg.get("data")
        if not data:
            return

        # books5 sends array of snapshots
        for snapshot in (data if isinstance(data, list) else [data]):
            self.builder.update(snapshot)

        self.update_count += 1
        self.kill.check(data_time=time.time())

        # Compute features and route
        if self.update_count % 2 == 0:  # every 2nd update to reduce noise
            await self._evaluate()

    async def on_trade(self, msg: dict):
        """Called on every trade WebSocket update."""
        data = msg.get("data")
        if data:
            self.trades.add(data if isinstance(data, list) else [data])

    async def _evaluate(self):
        """Evaluate features → route → execute."""
        if self.kill.blocked:
            return

        features = compute_features(self.builder, self.tick_size)
        if features.mid <= 0:
            return

        # Rate limit orders
        now = time.time()
        if now - self.last_order_time < self.min_order_interval:
            return

        # Enrich features with trade flow data
        features.toxicity = max(features.toxicity,
            0.3 * abs(self.trades.trade_imbalance) * min(self.trades.flow_burst / 3, 1.0))

        # Route
        decision = self.router.route(features, self.inventory)
        elapsed = _elapsed(self.start_time)

        if decision.strategy == "no_trade":
            if self.update_count % 20 == 0:  # log every 20th no_trade
                print(f"[{elapsed}] {decision.regime} | spread={features.spread_bps:.1f}bps imb={features.imbalance_3:+.2f} tox={features.toxicity:.2f} flow={self.trades.trade_imbalance:+.2f}")
            return

        # Execute
        self.last_order_time = now
        self.order_count += 1

        if decision.strategy == "inventory_mm":
            quote = decision.signal
            await self.client.cancel_all(self.symbol)
            br = await self.client.place(self.symbol, "buy", quote.bid_price, quote.size)
            sr = await self.client.place(self.symbol, "sell", quote.ask_price, quote.size)
            print(f"[{elapsed}] MM #{self.order_count} | bid={quote.bid_price} ask={quote.ask_price} skew={quote.skew:+.1f}bps | b={br.get('code','?')} s={sr.get('code','?')}")

        elif decision.strategy == "imbalance_mr":
            sig = decision.signal
            r = await self.client.place(self.symbol, sig.side, sig.price, sig.size)
            print(f"[{elapsed}] IMB #{self.order_count} | {sig.side} @{sig.price} | {sig.reason} | {r.get('code','?')}")

        elif decision.strategy == "sweep_breakout":
            sig = decision.signal
            r = await self.client.place(self.symbol, sig.side, sig.price, sig.size)
            print(f"[{elapsed}] BURST #{self.order_count} | {sig.side} @{sig.price} | {sig.reason} | {r.get('code','?')}")

    async def update_inventory(self):
        """Periodically refresh inventory from exchange."""
        try:
            self.inventory = await self.client.net_position(self.symbol)
        except Exception:
            pass


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="HYPEUSDT")
    parser.add_argument("--mode", default="live", choices=["demo", "live"])
    parser.add_argument("--min-interval", type=float, default=2.0, help="Min seconds between orders")
    args = parser.parse_args()

    order_client = BitgetOrderClient(mode=args.mode)
    if not order_client.key:
        print("ERROR: BITGET_API_KEY not set"); return

    engine = HFTEngine(args.symbol, order_client, tick_size=0.001)
    engine.min_order_interval = args.min_interval

    print(f"{'='*60}")
    print(f"  HFT v3 — WebSocket-Driven | {args.symbol} [{args.mode}]")
    print(f"  Data: Bitget WebSocket (real-time books5 + trades)")
    print(f"  Strategies: ImbalanceMR + SweepBreakout + InventoryMM")
    print(f"  Router: Regime-based | Kill: $10 dd / 8 losses")
    print(f"  Min order interval: {args.min_interval}s")
    print(f"{'='*60}\n")

    # WebSocket feed
    feed = BitgetWebSocketFeed(
        symbol=args.symbol,
        product_type=PRODUCT_TYPE,
        on_book=engine.on_book,
        on_trade=engine.on_trade,
    )

    # Periodic inventory check (every 30s)
    async def inventory_loop():
        while True:
            await engine.update_inventory()
            await asyncio.sleep(30)

    # Periodic stats (every 60s)
    async def stats_loop():
        while True:
            await asyncio.sleep(60)
            elapsed = _elapsed(engine.start_time)
            print(f"[{elapsed}] STATS | updates={engine.update_count} orders={engine.order_count} inv={engine.inventory:.4f} ws={feed.stats}")

    try:
        await asyncio.gather(
            feed.start(),
            inventory_loop(),
            stats_loop(),
        )
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        await feed.stop()
        await order_client.cancel_all(args.symbol)
        await order_client.close()
        print(f"Done. Orders: {engine.order_count} Updates: {engine.update_count}")
        print(f"Router stats: {json.dumps(engine.router.stats)}")


def _elapsed(start):
    e = int(time.time() - start)
    return f"{e//60}:{e%60:02d}"


if __name__ == "__main__":
    asyncio.run(main())
