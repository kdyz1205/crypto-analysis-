"""
OKX Trading Interface — Paper & Live modes.

Paper mode: simulates fills at market price (no real orders).
Live mode: uses OKX REST API v5 with HMAC-SHA256 signing.

The agent ALWAYS starts in paper mode. Live mode requires explicit API keys.
"""

import time
import hmac
import hashlib
import base64
import json
import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np


OKX_REST_BASE = "https://www.okx.com"


@dataclass
class Position:
    symbol: str
    side: str  # 'long' | 'short'
    size: float  # in USD notional
    entry_price: float
    entry_time: float  # unix ts
    unrealized_pnl: float = 0.0
    peak_pnl: float = 0.0  # for trailing stop


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl_pct: float
    pnl_usd: float
    entry_time: float
    exit_time: float
    reason: str  # 'TP' | 'SL' | 'EMA_BREAK' | 'TRAILING' | 'SIGNAL'


@dataclass
class RiskLimits:
    """Hard risk limits — the agent's survival rules."""
    max_position_pct: float = 0.05      # max 5% of equity per position
    max_total_exposure_pct: float = 0.15  # max 15% total exposure
    max_daily_loss_pct: float = 0.02     # stop trading if daily loss > 2%
    max_drawdown_pct: float = 0.05       # emergency shutdown at 5% drawdown
    max_positions: int = 3               # max concurrent positions
    cooldown_seconds: int = 3600         # 1 hour between trades on same symbol


@dataclass
class AgentState:
    """Persistent agent state — the agent's memory."""
    mode: str = "paper"  # 'paper' | 'live'
    equity: float = 10000.0  # starting paper equity
    peak_equity: float = 10000.0
    cash: float = 10000.0
    positions: dict = field(default_factory=dict)  # symbol -> Position
    trade_history: list = field(default_factory=list)  # TradeRecord list
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_daily_reset: float = 0.0  # unix ts of last daily reset
    generation: int = 0  # strategy evolution generation
    total_trades: int = 0
    total_pnl_usd: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    is_alive: bool = True  # False = emergency shutdown
    shutdown_reason: str = ""
    last_trade_time: dict = field(default_factory=dict)  # symbol -> unix ts
    # Strategy params that evolve
    strategy_params: dict = field(default_factory=lambda: {
        "mfi_period": 14,
        "ma_fast": 8,
        "ema_span": 21,
        "ma_slow": 55,
        "atr_period": 14,
        "atr_sl_mult": 2.5,
        "bb_period": 21,
        "bb_std": 2.5,
        "rsi_low": 40,
        "rsi_high": 70,
        "mfi_threshold": 50,
        "trailing_trigger_atr": 1.0,  # activate trailing after 1×ATR profit
    })

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "cash": round(self.cash, 2),
            "positions": {k: {
                "symbol": v.symbol, "side": v.side, "size": round(v.size, 2),
                "entry_price": v.entry_price, "unrealized_pnl": round(v.unrealized_pnl, 2),
            } for k, v in self.positions.items()},
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "generation": self.generation,
            "total_trades": self.total_trades,
            "total_pnl_usd": round(self.total_pnl_usd, 2),
            "win_rate": round(self.win_count / max(self.total_trades, 1) * 100, 1),
            "is_alive": self.is_alive,
            "shutdown_reason": self.shutdown_reason,
            "strategy_params": self.strategy_params,
            "recent_trades": [
                {
                    "symbol": t.symbol, "side": t.side, "pnl_pct": round(t.pnl_pct, 2),
                    "pnl_usd": round(t.pnl_usd, 2), "reason": t.reason,
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                }
                for t in self.trade_history[-20:]
            ],
        }


class OKXTrader:
    """
    OKX trading interface. Paper mode simulates fills; live mode uses API.
    """

    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.state = AgentState()
        self.risk = RiskLimits()
        self._price_cache: dict[str, float] = {}
        self._price_cache_time: dict[str, float] = {}

    # ── Price data ────────────────────────────────────────────────────────

    async def get_price(self, symbol: str) -> float | None:
        """Get current mark price from OKX."""
        now = time.time()
        # Cache for 5 seconds
        if symbol in self._price_cache and now - self._price_cache_time.get(symbol, 0) < 5:
            return self._price_cache[symbol]
        try:
            base = symbol.upper().replace("USDT", "")
            inst_id = f"{base}-USDT-SWAP"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{OKX_REST_BASE}/api/v5/market/ticker", params={"instId": inst_id})
                data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                price = float(data["data"][0]["last"])
                self._price_cache[symbol] = price
                self._price_cache_time[symbol] = now
                return price
        except Exception as e:
            print(f"Price fetch error for {symbol}: {e}")
        return self._price_cache.get(symbol)

    # ── Risk checks ───────────────────────────────────────────────────────

    def check_daily_reset(self):
        """Reset daily counters at midnight UTC."""
        now = time.time()
        if now - self.state.last_daily_reset > 86400:
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.last_daily_reset = now

    def can_trade(self, symbol: str) -> tuple[bool, str]:
        """Check all risk limits before allowing a trade."""
        if not self.state.is_alive:
            return False, f"Agent shutdown: {self.state.shutdown_reason}"

        self.check_daily_reset()

        # Daily loss limit
        if self.state.daily_pnl < -self.risk.max_daily_loss_pct * self.state.peak_equity:
            self.state.is_alive = False
            self.state.shutdown_reason = f"Daily loss limit hit: {self.state.daily_pnl:.2f}"
            return False, self.state.shutdown_reason

        # Drawdown limit
        dd = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
        if dd > self.risk.max_drawdown_pct:
            self.state.is_alive = False
            self.state.shutdown_reason = f"Max drawdown hit: {dd*100:.1f}%"
            return False, self.state.shutdown_reason

        # Max positions
        if len(self.state.positions) >= self.risk.max_positions and symbol not in self.state.positions:
            return False, f"Max positions ({self.risk.max_positions}) reached"

        # Total exposure
        total_exposure = sum(p.size for p in self.state.positions.values())
        if total_exposure > self.risk.max_total_exposure_pct * self.state.equity:
            return False, "Max total exposure reached"

        # Cooldown
        last = self.state.last_trade_time.get(symbol, 0)
        if time.time() - last < self.risk.cooldown_seconds:
            remaining = int(self.risk.cooldown_seconds - (time.time() - last))
            return False, f"Cooldown: {remaining}s remaining for {symbol}"

        return True, "OK"

    # ── Paper trading ─────────────────────────────────────────────────────

    async def open_position(self, symbol: str, side: str, size_usd: float) -> dict:
        """Open a position (paper or live)."""
        can, reason = self.can_trade(symbol)
        if not can:
            return {"ok": False, "reason": reason}

        # Enforce position size limit
        max_size = self.state.equity * self.risk.max_position_pct
        size_usd = min(size_usd, max_size)

        price = await self.get_price(symbol)
        if price is None:
            return {"ok": False, "reason": f"Cannot get price for {symbol}"}

        if self.state.mode == "paper":
            pos = Position(
                symbol=symbol, side=side, size=size_usd,
                entry_price=price, entry_time=time.time(),
            )
            self.state.positions[symbol] = pos
            self.state.cash -= size_usd
            self.state.last_trade_time[symbol] = time.time()
            return {"ok": True, "price": price, "size": size_usd, "side": side}
        else:
            # Live mode: place market order via OKX API
            return await self._place_order_live(symbol, side, size_usd, price)

    async def close_position(self, symbol: str, reason: str = "SIGNAL") -> dict:
        """Close an existing position."""
        if symbol not in self.state.positions:
            return {"ok": False, "reason": f"No position for {symbol}"}

        pos = self.state.positions[symbol]
        price = await self.get_price(symbol)
        if price is None:
            return {"ok": False, "reason": f"Cannot get price for {symbol}"}

        if pos.side == "long":
            pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - price) / pos.entry_price * 100
        pnl_usd = pos.size * pnl_pct / 100

        # Record trade
        record = TradeRecord(
            symbol=symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=price,
            size=pos.size, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
            entry_time=pos.entry_time, exit_time=time.time(),
            reason=reason,
        )
        self.state.trade_history.append(record)
        self.state.total_trades += 1
        self.state.total_pnl_usd += pnl_usd
        self.state.daily_pnl += pnl_usd
        self.state.daily_trades += 1
        if pnl_usd > 0:
            self.state.win_count += 1
        else:
            self.state.loss_count += 1

        # Update equity
        self.state.cash += pos.size + pnl_usd
        self.state.equity = self.state.cash + sum(p.size for p in self.state.positions.values() if p.symbol != symbol)
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

        del self.state.positions[symbol]
        self.state.last_trade_time[symbol] = time.time()

        if self.state.mode == "live":
            await self._close_order_live(symbol)

        return {"ok": True, "pnl_pct": round(pnl_pct, 2), "pnl_usd": round(pnl_usd, 2), "reason": reason}

    async def update_positions(self):
        """Update unrealized PnL for all positions."""
        for symbol, pos in list(self.state.positions.items()):
            price = await self.get_price(symbol)
            if price is None:
                continue
            if pos.side == "long":
                pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price * 100
            else:
                pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price * 100
            pos.peak_pnl = max(pos.peak_pnl, pos.unrealized_pnl)

        # Update total equity
        total_unrealized = sum(
            p.size * p.unrealized_pnl / 100
            for p in self.state.positions.values()
        )
        self.state.equity = self.state.cash + sum(p.size for p in self.state.positions.values()) + total_unrealized
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

    # ── Live trading (OKX API v5) ─────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = timestamp + method + path + body
        mac = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    async def _place_order_live(self, symbol: str, side: str, size_usd: float, price: float) -> dict:
        """Place a market order on OKX."""
        if not self.api_key:
            return {"ok": False, "reason": "No API key configured"}

        base = symbol.upper().replace("USDT", "")
        inst_id = f"{base}-USDT-SWAP"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        path = "/api/v5/trade/order"

        body = json.dumps({
            "instId": inst_id,
            "tdMode": "cross",
            "side": "buy" if side == "long" else "sell",
            "ordType": "market",
            "sz": str(round(size_usd / price, 6)),
        })

        signature = self._sign(timestamp, "POST", path, body)
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{OKX_REST_BASE}{path}", headers=headers, content=body)
                data = resp.json()
            if data.get("code") == "0":
                return {"ok": True, "orderId": data["data"][0]["ordId"], "price": price}
            return {"ok": False, "reason": data.get("msg", "Unknown OKX error")}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    async def _close_order_live(self, symbol: str) -> dict:
        """Close position on OKX (place opposite market order)."""
        # Simplified: in production, query actual position and close
        return {"ok": True, "note": "live close placeholder"}

    # ── Agent revival ─────────────────────────────────────────────────────

    def revive(self):
        """Revive the agent after shutdown (manual intervention)."""
        self.state.is_alive = True
        self.state.shutdown_reason = ""
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.last_daily_reset = time.time()
