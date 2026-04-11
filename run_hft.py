"""HFT System — Market Making + Order Book Imbalance + Regime Router

Strategies:
1. Queue Imbalance Mean Reversion (盘口失衡反转)
2. Sweep Breakout Follower (微型突破跟随)
3. Inventory-Skewed Market Making (偏斜做市)
5. Regime Router (状态路由)

Usage: python run_hft.py --symbol HYPEUSDT --mode live
"""

import asyncio
import json
import os
import sys
import time
import argparse
from dataclasses import dataclass
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass

import httpx

BITGET_REST = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"


# ── Regime Classification ────────────────────────────────────────────────

class Regime(Enum):
    MEAN_REVERT = "mean_revert"      # 均值回归 — 用策略1
    BURST = "burst"                   # 爆发突破 — 用策略2
    STABLE_SPREAD = "stable_spread"   # 稳定做市 — 用策略3
    TOXIC = "toxic"                   # 有毒流量 — 不交易


@dataclass
class MarketState:
    mid_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_bps: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    imbalance: float = 0.0         # -1 to 1 (positive = bid heavy)
    last_price: float = 0.0
    volume_1m: float = 0.0
    price_change_1m: float = 0.0
    regime: Regime = Regime.STABLE_SPREAD
    net_position: float = 0.0
    position_value: float = 0.0


# ── Bitget Client ────────────────────────────────────────────────────────

class HFTClient:
    def __init__(self, mode="live"):
        import base64, hashlib, hmac
        self.key = os.environ.get("BITGET_API_KEY", "")
        self.secret = os.environ.get("BITGET_SECRET_KEY", "") or os.environ.get("BITGET_SECRET", "")
        self.passphrase = os.environ.get("BITGET_PASSPHRASE", "")
        self.mode = mode
        self._b64, self._hmac, self._hash = base64, hmac, hashlib

    def _sign(self, ts, method, path, body=""):
        msg = f"{ts}{method}{path}{body}"
        sig = self._b64.b64encode(self._hmac.new(self.secret.encode(), msg.encode(), self._hash.sha256).digest()).decode()
        h = {"ACCESS-KEY": self.key, "ACCESS-SIGN": sig, "ACCESS-TIMESTAMP": ts,
             "ACCESS-PASSPHRASE": self.passphrase, "Content-Type": "application/json", "locale": "en-US"}
        if self.mode == "demo": h["paptrading"] = "1"
        return h

    async def ticker(self, sym):
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{BITGET_REST}/api/v2/mix/market/ticker", params={"productType": PRODUCT_TYPE, "symbol": sym})
            d = r.json()
            if d.get("code") == "00000" and d.get("data"):
                rows = d["data"] if isinstance(d["data"], list) else [d["data"]]
                return rows[0] if rows else None
        return None

    async def orderbook(self, sym, limit=5):
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{BITGET_REST}/api/v2/mix/market/merge-depth",
                           params={"productType": PRODUCT_TYPE, "symbol": sym, "limit": str(limit)})
            d = r.json()
            if d.get("code") == "00000" and d.get("data"):
                return d["data"]
        return None

    async def positions(self, sym):
        path = f"/api/v2/mix/position/single-position?productType={PRODUCT_TYPE}&symbol={sym}&marginCoin={MARGIN_COIN}"
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{BITGET_REST}{path}", headers=self._sign(str(int(time.time()*1000)), "GET", path))
            d = r.json()
            return d.get("data", []) or [] if d.get("code") == "00000" else []

    async def place(self, sym, side, price, size):
        path = "/api/v2/mix/order/place-order"
        body = json.dumps({"symbol": sym, "productType": PRODUCT_TYPE, "marginMode": "crossed",
                          "marginCoin": MARGIN_COIN, "side": side, "orderType": "limit",
                          "price": str(price), "size": str(size), "tradeSide": "open", "force": "gtc"})
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.post(f"{BITGET_REST}{path}", content=body, headers=self._sign(str(int(time.time()*1000)), "POST", path, body))
            return r.json()

    async def cancel_all(self, sym):
        path = "/api/v2/mix/order/cancel-all-orders"
        body = json.dumps({"symbol": sym, "productType": PRODUCT_TYPE, "marginCoin": MARGIN_COIN})
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.post(f"{BITGET_REST}{path}", content=body, headers=self._sign(str(int(time.time()*1000)), "POST", path, body))
            return r.json()

    async def close_position(self, sym, side, size):
        path = "/api/v2/mix/order/place-order"
        close_side = "sell" if side == "long" else "buy"
        body = json.dumps({"symbol": sym, "productType": PRODUCT_TYPE, "marginMode": "crossed",
                          "marginCoin": MARGIN_COIN, "side": close_side, "orderType": "market",
                          "size": str(size), "tradeSide": "close"})
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.post(f"{BITGET_REST}{path}", content=body, headers=self._sign(str(int(time.time()*1000)), "POST", path, body))
            return r.json()


# ── Market State Builder ─────────────────────────────────────────────────

class StateBuilder:
    def __init__(self):
        self.price_history = []  # last 60 prices (1 per second)
        self.spread_history = []

    async def build(self, client: HFTClient, symbol: str) -> MarketState:
        state = MarketState()

        # Ticker
        tk = await client.ticker(symbol)
        if not tk: return state
        state.last_price = float(tk.get("lastPr", 0))
        state.bid = float(tk.get("bidPr", 0))
        state.ask = float(tk.get("askPr", 0))
        state.mid_price = (state.bid + state.ask) / 2 if state.bid and state.ask else state.last_price
        state.spread_bps = ((state.ask - state.bid) / state.mid_price * 10000) if state.mid_price > 0 else 0

        # Order book
        ob = await client.orderbook(symbol, 5)
        if ob:
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            total_bid = sum(float(b[1]) for b in bids[:5]) if bids else 0
            total_ask = sum(float(a[1]) for a in asks[:5]) if asks else 0
            state.bid_size = total_bid
            state.ask_size = total_ask
            total = total_bid + total_ask
            state.imbalance = (total_bid - total_ask) / total if total > 0 else 0

        # Price history
        self.price_history.append(state.last_price)
        if len(self.price_history) > 60: self.price_history = self.price_history[-60:]
        self.spread_history.append(state.spread_bps)
        if len(self.spread_history) > 60: self.spread_history = self.spread_history[-60:]

        if len(self.price_history) >= 10:
            state.price_change_1m = (state.last_price - self.price_history[0]) / self.price_history[0] * 100

        # Position
        positions = await client.positions(symbol)
        for p in positions:
            if isinstance(p, dict):
                total = float(p.get("total", 0) or 0)
                side = p.get("holdSide", "")
                if side == "long": state.net_position += total
                elif side == "short": state.net_position -= total
        state.position_value = abs(state.net_position * state.mid_price)

        # Classify regime
        state.regime = self._classify_regime(state)
        return state

    def _classify_regime(self, s: MarketState) -> Regime:
        # High imbalance + narrow spread = mean reversion opportunity
        if abs(s.imbalance) > 0.6 and s.spread_bps < 5:
            return Regime.MEAN_REVERT

        # Large price move in last minute = burst/toxic
        if abs(s.price_change_1m) > 0.3:
            return Regime.TOXIC  # too volatile, stay out

        # Wide spread = not worth making market
        if s.spread_bps > 10:
            return Regime.TOXIC

        # Normal conditions = market make
        return Regime.STABLE_SPREAD


# ── Strategy Implementations ─────────────────────────────────────────────

class Strategy1_Imbalance:
    """Queue Imbalance Mean Reversion — fade the imbalance."""

    async def execute(self, client: HFTClient, sym: str, state: MarketState, cfg: dict) -> str:
        if abs(state.imbalance) < 0.5:
            return "imbalance too weak"

        size = round(cfg["size_usdt"] / state.mid_price, 4)
        if size <= 0: return "size too small"

        # Fade the imbalance: if bid-heavy (positive), expect short-term dip → sell
        if state.imbalance > 0.5:
            price = round(state.ask - state.mid_price * 0.0001, 4)  # slightly inside ask
            r = await client.place(sym, "sell", price, size)
            return f"SELL@{price} imb={state.imbalance:.2f} {r.get('code','?')}"
        else:
            price = round(state.bid + state.mid_price * 0.0001, 4)
            r = await client.place(sym, "buy", price, size)
            return f"BUY@{price} imb={state.imbalance:.2f} {r.get('code','?')}"


class Strategy3_MarketMake:
    """Inventory-Skewed Market Making — quote both sides, skew by position."""

    async def execute(self, client: HFTClient, sym: str, state: MarketState, cfg: dict) -> str:
        spread = cfg["spread_pct"] / 100.0
        size = round(cfg["size_usdt"] / state.mid_price, 4)
        if size <= 0: return "size too small"

        half_spread = state.mid_price * spread

        # Inventory skew: if long, lower bid & lower ask (encourage selling)
        skew = 0.0
        if state.position_value > cfg["max_pos_usdt"] * 0.5:
            skew = -half_spread * 0.3 if state.net_position > 0 else half_spread * 0.3

        buy_price = round(state.mid_price - half_spread + skew, 4)
        sell_price = round(state.mid_price + half_spread + skew, 4)

        # Cancel existing, place new
        await client.cancel_all(sym)

        # Don't place if position too large
        if state.position_value > cfg["max_pos_usdt"]:
            # Only place reducing side
            if state.net_position > 0:
                r = await client.place(sym, "sell", sell_price, size)
                return f"SELL_ONLY@{sell_price} pos={state.net_position:.4f} {r.get('code','?')}"
            else:
                r = await client.place(sym, "buy", buy_price, size)
                return f"BUY_ONLY@{buy_price} pos={state.net_position:.4f} {r.get('code','?')}"

        br = await client.place(sym, "buy", buy_price, size)
        sr = await client.place(sym, "sell", sell_price, size)
        return f"QUOTE bid={buy_price} ask={sell_price} sz={size} pos={state.net_position:.4f} b={br.get('code','?')} s={sr.get('code','?')}"


# ── Main Loop ────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="HYPEUSDT")
    parser.add_argument("--mode", default="live", choices=["demo", "live"])
    parser.add_argument("--spread", type=float, default=0.12, help="Spread %% per side")
    parser.add_argument("--size", type=float, default=6.0, help="Order size USDT")
    parser.add_argument("--max-pos", type=float, default=15.0, help="Max position USDT")
    parser.add_argument("--interval", type=float, default=5.0, help="Loop seconds")
    args = parser.parse_args()

    client = HFTClient(mode=args.mode)
    if not client.key:
        print("ERROR: BITGET_API_KEY not set"); return

    builder = StateBuilder()
    s1 = Strategy1_Imbalance()
    s3 = Strategy3_MarketMake()
    cfg = {"spread_pct": args.spread, "size_usdt": args.size, "max_pos_usdt": args.max_pos}

    print(f"{'='*55}")
    print(f"  HFT System — {args.symbol} [{args.mode}]")
    print(f"  Spread: {args.spread}% | Size: ${args.size} | MaxPos: ${args.max_pos}")
    print(f"  Strategies: Imbalance + MarketMake + RegimeRouter")
    print(f"{'='*55}")

    # Test
    tk = await client.ticker(args.symbol)
    if tk:
        print(f"  Price: ${tk.get('lastPr')} Bid: ${tk.get('bidPr')} Ask: ${tk.get('askPr')}")
    else:
        print("  ERROR: no ticker"); return

    ob = await client.orderbook(args.symbol, 5)
    if ob:
        bids = ob.get("bids", [])[:3]
        asks = ob.get("asks", [])[:3]
        print(f"  Book: bids={bids} asks={asks}")

    print(f"\n  Running... Ctrl+C to stop\n")

    trades = 0
    start = time.time()

    try:
        while True:
            try:
                state = await builder.build(client, args.symbol)
                elapsed = int(time.time() - start)
                mins = elapsed // 60
                secs = elapsed % 60

                if state.regime == Regime.TOXIC:
                    await client.cancel_all(args.symbol)
                    print(f"[{mins}:{secs:02d}] TOXIC — no trade | price={state.last_price} chg={state.price_change_1m:.2f}% spread={state.spread_bps:.1f}bps")

                elif state.regime == Regime.MEAN_REVERT:
                    result = await s1.execute(client, args.symbol, state, cfg)
                    trades += 1
                    print(f"[{mins}:{secs:02d}] IMBALANCE | {result}")

                elif state.regime == Regime.STABLE_SPREAD:
                    result = await s3.execute(client, args.symbol, state, cfg)
                    trades += 1
                    print(f"[{mins}:{secs:02d}] MM | {result}")

                else:
                    print(f"[{mins}:{secs:02d}] {state.regime.value} | idle")

            except Exception as e:
                print(f"[err] {e}")

            await asyncio.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nStopping... cancelling orders...")
        await client.cancel_all(args.symbol)
        # Close any open position
        if abs(builder.price_history[-1] if builder.price_history else 0) > 0:
            positions = await client.positions(args.symbol)
            for p in positions:
                if isinstance(p, dict) and float(p.get("total", 0) or 0) > 0:
                    await client.close_position(args.symbol, p.get("holdSide", "long"), p.get("total", "0"))
                    print(f"  Closed {p.get('holdSide')} {p.get('total')}")
        print(f"Done. Trades: {trades} Runtime: {int(time.time()-start)}s")


if __name__ == "__main__":
    asyncio.run(main())
