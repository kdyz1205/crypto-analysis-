"""
Orderbook Service — manages websocket connections and provides real-time features.

Starts a background task that maintains L2 orderbook for actively traded symbols.
Exposes get_features(symbol) for other components (runner, strategies) to query.

Usage:
    from server.hft.orderbook_service import get_features, start_service, get_status
    features = get_features("BTCUSDT")  # MarketFeatures or None
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from .data_feed.ws_feed import BitgetWebSocketFeed
from .data_feed.book_builder import BookBuilder
from .data_feed.features import MarketFeatures, compute_features

# ─── State ───
_feeds: dict[str, BitgetWebSocketFeed] = {}
_builders: dict[str, BookBuilder] = {}
_features: dict[str, MarketFeatures] = {}
_task: asyncio.Task | None = None
_running = False


def get_features(symbol: str) -> MarketFeatures | None:
    """Get latest computed features for a symbol. Returns None if no data."""
    return _features.get(symbol.upper())


def get_all_features() -> dict[str, dict]:
    """Get features for all tracked symbols as dicts."""
    out = {}
    for sym, f in _features.items():
        out[sym] = {
            "mid": f.mid, "spread_bps": f.spread_bps,
            "imbalance_3": f.imbalance_3, "imbalance_5": f.imbalance_5,
            "microprice": f.microprice, "cancel_pressure": f.cancel_pressure,
            "depth_vacuum_ask": f.depth_vacuum_ask, "depth_vacuum_bid": f.depth_vacuum_bid,
            "realized_vol": f.realized_vol_5s, "compression": f.compression,
            "regime": f.regime, "toxicity": f.toxicity,
            "mr_score": f.mr_score, "burst_score": f.burst_score,
        }
    return out


def get_status() -> dict:
    """Service status for API/UI."""
    return {
        "running": _running,
        "symbols_tracked": list(_feeds.keys()),
        "features_available": list(_features.keys()),
        "stats": {sym: feed.stats for sym, feed in _feeds.items()},
    }


async def _on_book_update(symbol: str, data: dict):
    """Called by websocket feed on every orderbook update."""
    builder = _builders.get(symbol)
    if not builder:
        return

    # Parse Bitget book data
    bids = [(float(b[0]), float(b[1])) for b in (data.get("bids") or [])]
    asks = [(float(a[0]), float(a[1])) for a in (data.get("asks") or [])]

    if bids or asks:
        builder.update_snapshot(bids, asks)
        _features[symbol] = compute_features(builder)


async def _on_trade_update(symbol: str, data: dict):
    """Called on every trade print — can be used for flow analysis."""
    # For now just track in features; extend later for order fill speed etc.
    pass


async def start_service(symbols: list[str] | None = None):
    """Start websocket feeds for given symbols. Called at server startup."""
    global _task, _running

    if _running:
        return

    if symbols is None:
        # Default: top symbols the runner is likely to trade
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "DOGEUSDT"]

    _running = True

    for sym in symbols:
        sym = sym.upper()
        _builders[sym] = BookBuilder()

        async def make_book_cb(s=sym):
            async def cb(data):
                await _on_book_update(s, data)
            return cb

        async def make_trade_cb(s=sym):
            async def cb(data):
                await _on_trade_update(s, data)
            return cb

        book_cb = await make_book_cb(sym)
        trade_cb = await make_trade_cb(sym)

        feed = BitgetWebSocketFeed(
            symbol=sym,
            on_book=book_cb,
            on_trade=trade_cb,
        )
        _feeds[sym] = feed

    # Start all feeds as background tasks
    for sym, feed in _feeds.items():
        asyncio.create_task(feed.start())
        print(f"[orderbook] started feed for {sym}", flush=True)

    print(f"[orderbook] service started, tracking {list(_feeds.keys())}", flush=True)


async def stop_service():
    """Stop all websocket feeds."""
    global _running
    _running = False
    for sym, feed in _feeds.items():
        await feed.stop()
    _feeds.clear()
    _builders.clear()
    _features.clear()
    print("[orderbook] service stopped", flush=True)
