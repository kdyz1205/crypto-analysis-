"""HFT Runner — uses the proper module architecture.

Connects to Bitget, builds order book, computes features,
routes through regime router, executes via strategies.

Usage: python run_hft.py --symbol HYPEUSDT --mode live
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

from server.hft.data_feed import BookBuilder, compute_features
from server.hft.router import RegimeRouter
from server.hft.risk import HFTKillSwitch
from server.hft.strategies.inventory_mm import MMQuote
from server.hft.strategies.imbalance_mr import ImbalanceSignal
from server.hft.strategies.sweep_breakout import BreakoutSignal

BITGET_REST = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"


class BitgetHFT:
    """Minimal Bitget client for HFT."""

    def __init__(self, mode="live"):
        import base64, hashlib, hmac
        self.key = os.environ.get("BITGET_API_KEY", "")
        self.secret = os.environ.get("BITGET_SECRET_KEY", "") or os.environ.get("BITGET_SECRET", "")
        self.passphrase = os.environ.get("BITGET_PASSPHRASE", "")
        self.mode = mode
        self._b64, self._hmac, self._hash = base64, hmac, hashlib

    def _headers(self, method, path, body=""):
        ts = str(int(time.time() * 1000))
        sig = self._b64.b64encode(self._hmac.new(self.secret.encode(), f"{ts}{method}{path}{body}".encode(), self._hash.sha256).digest()).decode()
        h = {"ACCESS-KEY": self.key, "ACCESS-SIGN": sig, "ACCESS-TIMESTAMP": ts, "ACCESS-PASSPHRASE": self.passphrase, "Content-Type": "application/json", "locale": "en-US"}
        if self.mode == "demo": h["paptrading"] = "1"
        return h

    async def orderbook(self, sym, limit=10):
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{BITGET_REST}/api/v2/mix/market/merge-depth", params={"productType": PRODUCT_TYPE, "symbol": sym, "limit": str(limit)})
            d = r.json()
            return d.get("data") if d.get("code") == "00000" else None

    async def ticker(self, sym):
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{BITGET_REST}/api/v2/mix/market/ticker", params={"productType": PRODUCT_TYPE, "symbol": sym})
            d = r.json()
            if d.get("code") == "00000" and d.get("data"):
                rows = d["data"] if isinstance(d["data"], list) else [d["data"]]
                return rows[0] if rows else None
        return None

    async def place_order(self, sym, side, price, size, force="gtc"):
        path = "/api/v2/mix/order/place-order"
        body = json.dumps({"symbol": sym, "productType": PRODUCT_TYPE, "marginMode": "crossed", "marginCoin": MARGIN_COIN, "side": side, "orderType": "limit", "price": str(price), "size": str(size), "tradeSide": "open", "force": force})
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.post(f"{BITGET_REST}{path}", content=body, headers=self._headers("POST", path, body))
            return r.json()

    async def cancel_all(self, sym):
        path = "/api/v2/mix/order/cancel-all-orders"
        body = json.dumps({"symbol": sym, "productType": PRODUCT_TYPE, "marginCoin": MARGIN_COIN})
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.post(f"{BITGET_REST}{path}", content=body, headers=self._headers("POST", path, body))
            return r.json()

    async def net_position(self, sym) -> float:
        path = f"/api/v2/mix/position/single-position?productType={PRODUCT_TYPE}&symbol={sym}&marginCoin={MARGIN_COIN}"
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{BITGET_REST}{path}", headers=self._headers("GET", path))
            d = r.json()
            pos = 0.0
            for p in (d.get("data") or []):
                if isinstance(p, dict):
                    t = float(p.get("total", 0) or 0)
                    pos += t if p.get("holdSide") == "long" else -t
            return pos


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="HYPEUSDT")
    parser.add_argument("--mode", default="live", choices=["demo", "live"])
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    client = BitgetHFT(mode=args.mode)
    if not client.key:
        print("ERROR: BITGET_API_KEY not set"); return

    builder = BookBuilder()
    router = RegimeRouter()
    kill = HFTKillSwitch()

    # Get tick size
    tk = await client.ticker(args.symbol)
    if not tk:
        print("ERROR: no ticker"); return

    price = float(tk["lastPr"])
    tick_size = 0.001  # HYPE default

    print(f"{'='*55}")
    print(f"  HFT System v2 — {args.symbol} [{args.mode}]")
    print(f"  Price: ${price} | Tick: {tick_size}")
    print(f"  Strategies: ImbalanceMR + SweepBreakout + InventoryMM")
    print(f"  Router: Regime-based | Kill: $10 drawdown / 8 losses")
    print(f"{'='*55}\n")

    start = time.time()

    try:
        while True:
            t0 = time.time()

            try:
                # 1. Fetch order book
                ob = await client.orderbook(args.symbol, 10)
                if not ob:
                    print(f"[{_elapsed(start)}] no book data")
                    await asyncio.sleep(args.interval)
                    continue

                # 2. Build book + features
                book = builder.update(ob)
                features = compute_features(builder, tick_size)
                latency = (time.time() - t0) * 1000

                # 3. Kill switch
                if not kill.check(latency_ms=latency, data_time=time.time()):
                    print(f"[{_elapsed(start)}] KILLED: {kill.block_reason}")
                    await client.cancel_all(args.symbol)
                    break

                # 4. Get inventory
                inventory = await client.net_position(args.symbol)

                # 5. Route + execute
                decision = router.route(features, inventory)

                if decision.strategy == "no_trade":
                    print(f"[{_elapsed(start)}] {decision.regime} | no_trade | spread={features.spread_bps:.1f}bps imb={features.imbalance_3:+.2f} tox={features.toxicity:.2f}")

                elif decision.strategy == "inventory_mm":
                    quote = decision.signal
                    await client.cancel_all(args.symbol)
                    br = await client.place_order(args.symbol, "buy", quote.bid_price, quote.size)
                    sr = await client.place_order(args.symbol, "sell", quote.ask_price, quote.size)
                    print(f"[{_elapsed(start)}] MM | bid={quote.bid_price} ask={quote.ask_price} skew={quote.skew:+.1f}bps inv={inventory:.4f} | b={br.get('code','?')} s={sr.get('code','?')}")

                elif decision.strategy == "imbalance_mr":
                    sig = decision.signal
                    r = await client.place_order(args.symbol, sig.side, sig.price, sig.size)
                    print(f"[{_elapsed(start)}] IMB | {sig.side} @{sig.price} | {sig.reason} | {r.get('code','?')}")

                elif decision.strategy == "sweep_breakout":
                    sig = decision.signal
                    r = await client.place_order(args.symbol, sig.side, sig.price, sig.size)
                    print(f"[{_elapsed(start)}] BURST | {sig.side} @{sig.price} | {sig.reason} | {r.get('code','?')}")

            except Exception as e:
                print(f"[{_elapsed(start)}] error: {e}")

            await asyncio.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nStopping...")
        await client.cancel_all(args.symbol)
        print(f"Orders cancelled. Stats: {json.dumps(router.stats)}")


def _elapsed(start):
    e = int(time.time() - start)
    return f"{e//60}:{e%60:02d}"


if __name__ == "__main__":
    asyncio.run(main())
