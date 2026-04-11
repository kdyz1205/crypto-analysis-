"""High-frequency market maker for Bitget USDT-Futures.

Places bid/ask orders around mid price, profits from spread.
Designed for $100 capital, conservative sizing.

Usage: python run_market_maker.py [--symbol HYPEUSDT] [--spread 0.15] [--size 5]
"""

import asyncio
import os
import sys
import time
import argparse
from decimal import Decimal, ROUND_DOWN

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass

import httpx

# ── Config ───────────────────────────────────────────────────────────────

BITGET_REST = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"


def parse_args():
    p = argparse.ArgumentParser(description="Bitget Market Maker")
    p.add_argument("--symbol", default="HYPEUSDT", help="Trading pair")
    p.add_argument("--spread", type=float, default=0.15, help="Spread percentage per side (0.15 = 0.15%%)")
    p.add_argument("--size", type=float, default=5.0, help="Order size in USDT")
    p.add_argument("--max-position", type=float, default=20.0, help="Max position USDT")
    p.add_argument("--interval", type=float, default=3.0, help="Loop interval seconds")
    p.add_argument("--mode", default="demo", choices=["demo", "live"], help="demo or live")
    return p.parse_args()


class BitgetMMClient:
    """Minimal Bitget client for market making."""

    def __init__(self, mode="demo"):
        import base64, hashlib, hmac
        self.api_key = os.environ.get("BITGET_API_KEY", "")
        self.api_secret = os.environ.get("BITGET_SECRET_KEY", "") or os.environ.get("BITGET_SECRET", "")
        self.passphrase = os.environ.get("BITGET_PASSPHRASE", "")
        self.mode = mode
        self.base = BITGET_REST
        self._hmac = hmac
        self._hashlib = hashlib
        self._base64 = base64

    def _sign(self, timestamp, method, path, body=""):
        message = f"{timestamp}{method}{path}{body}"
        signature = self._base64.b64encode(
            self._hmac.new(
                self.api_secret.encode("utf-8"),
                message.encode("utf-8"),
                self._hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        return signature

    def _headers(self, method, path, body=""):
        ts = str(int(time.time() * 1000))
        sig = self._sign(ts, method, path, body)
        h = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": sig,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US",
        }
        if self.mode == "demo":
            h["paptrading"] = "1"
        return h

    async def get_ticker(self, symbol: str) -> dict:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{self.base}/api/v2/mix/market/ticker", params={
                "productType": PRODUCT_TYPE, "symbol": symbol,
            })
            data = r.json()
            if data.get("code") == "00000" and data.get("data"):
                rows = data["data"] if isinstance(data["data"], list) else [data["data"]]
                return rows[0] if rows else {}
            return {}

    async def get_positions(self, symbol: str) -> list:
        path = "/api/v2/mix/position/single-position"
        params = f"productType={PRODUCT_TYPE}&symbol={symbol}&marginCoin={MARGIN_COIN}"
        full_path = f"{path}?{params}"
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{self.base}{full_path}", headers=self._headers("GET", full_path))
            data = r.json()
            if data.get("code") == "00000":
                return data.get("data", []) or []
            return []

    async def get_open_orders(self, symbol: str) -> list:
        path = "/api/v2/mix/order/orders-pending"
        params = f"productType={PRODUCT_TYPE}&symbol={symbol}"
        full_path = f"{path}?{params}"
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{self.base}{full_path}", headers=self._headers("GET", full_path))
            data = r.json()
            if data.get("code") == "00000":
                return data.get("data", {}).get("entrustedList", []) or []
            return []

    async def place_order(self, symbol: str, side: str, price: float, size: float) -> dict:
        import json
        path = "/api/v2/mix/order/place-order"
        body = json.dumps({
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "marginMode": "crossed",
            "marginCoin": MARGIN_COIN,
            "side": side,  # "buy" or "sell"
            "orderType": "limit",
            "price": str(price),
            "size": str(size),
            "tradeSide": "open",
            "force": "gtc",
        })
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(f"{self.base}{path}", content=body, headers=self._headers("POST", path, body))
            return r.json()

    async def cancel_all(self, symbol: str) -> dict:
        import json
        path = "/api/v2/mix/order/cancel-all-orders"
        body = json.dumps({
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "marginCoin": MARGIN_COIN,
        })
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(f"{self.base}{path}", content=body, headers=self._headers("POST", path, body))
            return r.json()


class MarketMaker:
    """Core market making logic."""

    def __init__(self, client: BitgetMMClient, symbol: str, spread_pct: float, size_usdt: float, max_pos: float):
        self.client = client
        self.symbol = symbol
        self.spread = spread_pct / 100.0  # e.g. 0.15% → 0.0015
        self.size_usdt = size_usdt
        self.max_pos = max_pos
        self.trades = 0
        self.pnl = 0.0
        self.start_time = time.time()

    async def run_once(self):
        """One iteration of market making."""
        # 1. Get current price
        ticker = await self.client.get_ticker(self.symbol)
        if not ticker:
            print("[mm] no ticker data")
            return

        last_price = float(ticker.get("lastPr", 0))
        bid1 = float(ticker.get("bidPr", last_price))
        ask1 = float(ticker.get("askPr", last_price))

        if last_price <= 0:
            print("[mm] invalid price")
            return

        mid = (bid1 + ask1) / 2

        # 2. Check position
        positions = await self.client.get_positions(self.symbol)
        net_pos = 0.0
        for p in positions:
            if isinstance(p, dict):
                total = float(p.get("total", 0) or 0)
                side = p.get("holdSide", "")
                if side == "long":
                    net_pos += total
                elif side == "short":
                    net_pos -= total

        # 3. Cancel existing orders
        await self.client.cancel_all(self.symbol)

        # 4. Calculate order prices
        half_spread = mid * self.spread
        buy_price = round(mid - half_spread, 4)
        sell_price = round(mid + half_spread, 4)

        # 5. Calculate size (in contracts)
        # For USDT-futures, size is usually in base currency units
        size = round(self.size_usdt / mid, 4)
        if size <= 0:
            size = 0.1  # minimum

        # 6. Inventory management: skew prices based on position
        pos_value = abs(net_pos * mid)
        if pos_value > self.max_pos:
            # Too much exposure — only place reducing orders
            if net_pos > 0:
                # Long heavy — only place sell
                result = await self.client.place_order(self.symbol, "sell", sell_price, size)
                self._log("SELL_ONLY", sell_price, size, net_pos, result)
                return
            else:
                # Short heavy — only place buy
                result = await self.client.place_order(self.symbol, "buy", buy_price, size)
                self._log("BUY_ONLY", buy_price, size, net_pos, result)
                return

        # 7. Place both sides
        buy_result = await self.client.place_order(self.symbol, "buy", buy_price, size)
        sell_result = await self.client.place_order(self.symbol, "sell", sell_price, size)

        self._log("QUOTE", f"bid={buy_price} ask={sell_price}", size, net_pos,
                  f"buy={buy_result.get('code','?')} sell={sell_result.get('code','?')}")

    def _log(self, action, price, size, pos, result):
        elapsed = time.time() - self.start_time
        mins = int(elapsed / 60)
        print(f"[mm {mins}m] {action} {self.symbol} price={price} size={size} pos={pos:.4f} | {result}")


async def main():
    args = parse_args()

    client = BitgetMMClient(mode=args.mode)
    if not client.api_key:
        print("ERROR: BITGET_API_KEY not set")
        return

    mm = MarketMaker(client, args.symbol, args.spread, args.size, args.max_position)

    print(f"{'='*50}")
    print(f"  Market Maker — {args.symbol}")
    print(f"  Mode: {args.mode}")
    print(f"  Spread: {args.spread}%")
    print(f"  Size: ${args.size} per side")
    print(f"  Max position: ${args.max_position}")
    print(f"  Interval: {args.interval}s")
    print(f"{'='*50}")

    # Test connection
    ticker = await client.get_ticker(args.symbol)
    if ticker:
        print(f"  Price: ${ticker.get('lastPr', '?')}")
        print(f"  Bid: ${ticker.get('bidPr', '?')} Ask: ${ticker.get('askPr', '?')}")
    else:
        print("  ERROR: Cannot get ticker. Check API keys and symbol.")
        return

    print(f"\n  Starting market making... Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                await mm.run_once()
            except Exception as e:
                print(f"[mm] error: {e}")
            await asyncio.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n[mm] Stopping... cancelling all orders...")
        await client.cancel_all(args.symbol)
        print(f"[mm] Done. Trades: {mm.trades}")


if __name__ == "__main__":
    asyncio.run(main())
