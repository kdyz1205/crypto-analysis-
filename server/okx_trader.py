"""
OKX Trading Interface — Paper & Live modes.

Paper mode: simulates fills at market price (no real orders).
Live mode: uses OKX REST API v5 with HMAC-SHA256 signing.

The agent ALWAYS starts in paper mode. Live mode requires explicit API keys.
"""

import os
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

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

OKX_REST_BASE = "https://www.okx.com"
# Set to True to use OKX demo/simulated trading (paper with real API)
OKX_DEMO_TRADING = os.environ.get("OKX_DEMO_TRADING", "false").lower() == "true"


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
    # V6 strategy params (4-layer filtered trend following)
    strategy_params: dict = field(default_factory=lambda: {
        # Moving average lengths
        "ma5_len": 5,
        "ma8_len": 8,
        "ema21_len": 21,
        "ma55_len": 55,
        # Bollinger Bands
        "bb_length": 21,
        "bb_std_dev": 2.5,
        # Layer 2: fanning distance thresholds (max % gap between adjacent MAs)
        "dist_ma5_ma8": 1.5,
        "dist_ma8_ema21": 2.5,
        "dist_ema21_ma55": 4.0,
        # Layer 3: slope momentum (min absolute slope over N bars, in %)
        "slope_len": 3,
        "slope_threshold": 0.1,
        # ATR for sizing
        "atr_period": 14,
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
        self.api_key = api_key or os.environ.get("OKX_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("OKX_SECRET", "") or os.environ.get("OKX_API_SECRET", "")
        self.passphrase = passphrase or os.environ.get("OKX_PASSPHRASE", "")
        self.state = AgentState()
        self.risk = RiskLimits()
        self._price_cache: dict[str, float] = {}
        self._price_cache_time: dict[str, float] = {}
        # Auto-detect: if keys present, log it
        if self.api_key:
            print(f"[OKX] API key loaded (live trading available)")

    def has_api_keys(self) -> bool:
        return bool(self.api_key and self.api_secret and self.passphrase)

    def set_api_keys(self, api_key: str, api_secret: str, passphrase: str):
        """Set OKX API keys at runtime (from frontend config)."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        print(f"[OKX] API keys updated. Live trading {'ready' if self.has_api_keys() else 'incomplete'}")

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
        import datetime as _dt
        now = time.time()
        now_utc = _dt.datetime.utcfromtimestamp(now)
        last_utc = _dt.datetime.utcfromtimestamp(self.state.last_daily_reset) if self.state.last_daily_reset > 0 else None
        if last_utc is None or now_utc.date() > last_utc.date():
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
        if self.state.peak_equity <= 0:
            self.state.peak_equity = max(self.state.equity, 1.0)
        dd = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
        if dd > self.risk.max_drawdown_pct:
            self.state.is_alive = False
            self.state.shutdown_reason = f"Max drawdown hit: {dd*100:.1f}%"
            return False, self.state.shutdown_reason

        # Max positions
        if len(self.state.positions) >= self.risk.max_positions and symbol not in self.state.positions:
            return False, f"Max positions ({self.risk.max_positions}) reached"

        # Total exposure (checked again with new position size in open_position)
        total_exposure = sum(p.size for p in self.state.positions.values())
        max_exposure = self.risk.max_total_exposure_pct * self.state.equity
        if total_exposure >= max_exposure:
            return False, f"Max total exposure reached ({total_exposure:.0f}/{max_exposure:.0f})"

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

        # Enforce total exposure limit (including this new position)
        current_exposure = sum(p.size for p in self.state.positions.values())
        max_exposure = self.risk.max_total_exposure_pct * self.state.equity
        remaining = max_exposure - current_exposure
        if remaining <= 0:
            return {"ok": False, "reason": "Max total exposure reached"}
        size_usd = min(size_usd, remaining)

        # Reject zero/tiny trades
        if size_usd < 1.0:
            return {"ok": False, "reason": f"Position size too small: ${size_usd:.2f}"}

        price = await self.get_price(symbol)
        if price is None:
            return {"ok": False, "reason": f"Cannot get price for {symbol}"}

        if self.state.mode == "paper":
            if self.state.cash < size_usd:
                return {"ok": False, "reason": f"Insufficient cash: {self.state.cash:.2f} < {size_usd:.2f}"}
            pos = Position(
                symbol=symbol, side=side, size=size_usd,
                entry_price=price, entry_time=time.time(),
            )
            self.state.positions[symbol] = pos
            self.state.cash -= size_usd
            # Update equity immediately so next trade uses correct value
            self.state.equity = self.state.cash + sum(p.size for p in self.state.positions.values())
            self.state.last_trade_time[symbol] = time.time()
            return {"ok": True, "price": price, "size": size_usd, "side": side}
        else:
            # Live mode: place market order via OKX API
            result = await self._place_order_live(symbol, side, size_usd, price)
            if result.get("ok"):
                # Sync position to local state after confirmed exchange order
                pos = Position(
                    symbol=symbol, side=side, size=size_usd,
                    entry_price=price, entry_time=time.time(),
                )
                self.state.positions[symbol] = pos
                self.state.cash -= size_usd
                self.state.equity = self.state.cash + sum(p.size for p in self.state.positions.values())
                self.state.last_trade_time[symbol] = time.time()
            return result

    async def close_position(self, symbol: str, reason: str = "SIGNAL") -> dict:
        """Close an existing position."""
        if symbol not in self.state.positions:
            return {"ok": False, "reason": f"No position for {symbol}"}

        pos = self.state.positions[symbol]
        price = await self.get_price(symbol)
        if price is None:
            return {"ok": False, "reason": f"Cannot get price for {symbol}"}

        if pos.entry_price <= 0:
            return {"ok": False, "reason": f"Invalid entry price for {symbol}"}
        if pos.side == "long":
            pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - price) / pos.entry_price * 100
        pnl_usd = pos.size * pnl_pct / 100

        # In live mode, close on exchange FIRST before recording trade
        if self.state.mode == "live":
            live_result = await self._close_order_live(symbol, pos.side)
            if not live_result.get("ok"):
                return {"ok": False, "reason": f"Exchange close failed: {live_result.get('reason')}"}

        # Record trade only after confirmed close
        record = TradeRecord(
            symbol=symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=price,
            size=pos.size, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
            entry_time=pos.entry_time, exit_time=time.time(),
            reason=reason,
        )
        self.state.trade_history.append(record)
        # Cap history to prevent unbounded memory growth
        if len(self.state.trade_history) > 500:
            self.state.trade_history = self.state.trade_history[-500:]
        self.state.total_trades += 1
        self.state.total_pnl_usd += pnl_usd
        self.state.daily_pnl += pnl_usd
        self.state.daily_trades += 1
        if pnl_usd > 0:
            self.state.win_count += 1
        elif pnl_usd < 0:
            self.state.loss_count += 1

        # Remove position, then update equity
        del self.state.positions[symbol]
        self.state.cash += pos.size + pnl_usd
        self.state.equity = self.state.cash + sum(p.size for p in self.state.positions.values())
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

        self.state.last_trade_time[symbol] = time.time()

        return {"ok": True, "price": price, "pnl_pct": round(pnl_pct, 2), "pnl_usd": round(pnl_usd, 2), "reason": reason, "exit_price": price}

    async def update_positions(self):
        """Update unrealized PnL for all positions."""
        for symbol, pos in list(self.state.positions.items()):
            price = await self.get_price(symbol)
            if price is None or pos.entry_price <= 0:
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

    def _make_headers(self, timestamp: str, method: str, path: str, body: str = "") -> dict:
        """Build authenticated OKX headers with optional demo trading flag."""
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": self._sign(timestamp, method, path, body),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        if OKX_DEMO_TRADING:
            headers["x-simulated-trading"] = "1"
        return headers

    def _inst_id(self, symbol: str) -> str:
        base = symbol.upper().replace("USDT", "")
        return f"{base}-USDT-SWAP"

    async def _okx_request(self, method: str, path: str, body: str = "") -> dict:
        """Make authenticated OKX API request."""
        if not self.has_api_keys():
            return {"code": "-1", "msg": "No API keys configured"}
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        headers = self._make_headers(timestamp, method, path, body)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                if method == "GET":
                    resp = await client.get(f"{OKX_REST_BASE}{path}", headers=headers)
                else:
                    resp = await client.post(f"{OKX_REST_BASE}{path}", headers=headers, content=body)
                if resp.status_code >= 400:
                    print(f"[OKX] HTTP {resp.status_code} on {method} {path}")
                    return {"code": "-1", "msg": f"HTTP {resp.status_code}: {resp.text[:200]}"}
                return resp.json()
        except Exception as e:
            return {"code": "-1", "msg": str(e)}

    async def get_account_balance(self) -> dict:
        """Get OKX account balance to verify API keys work."""
        data = await self._okx_request("GET", "/api/v5/account/balance")
        if data.get("code") == "0" and data.get("data") and len(data["data"]) > 0:
            acct = data["data"][0]
            details = acct.get("details", [])
            usdt = next((d for d in details if d.get("ccy") == "USDT"), None)
            return {
                "ok": True,
                "total_equity": float(acct.get("totalEq", 0)),
                "usdt_available": float(usdt.get("availBal", 0)) if usdt else 0,
            }
        return {"ok": False, "reason": data.get("msg", "Unknown error")}

    async def _get_contract_size(self, inst_id: str) -> float:
        """Get contract size for an instrument (e.g., BTC-USDT-SWAP ctVal=0.01)."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{OKX_REST_BASE}/api/v5/public/instruments",
                    params={"instType": "SWAP", "instId": inst_id}
                )
                data = resp.json()
            if data.get("code") == "0" and data.get("data") and len(data["data"]) > 0:
                return float(data["data"][0].get("ctVal", 1))
        except Exception as e:
            print(f"[OKX] Failed to get contract size for {inst_id}: {e}")
        return 1.0

    async def _place_order_live(self, symbol: str, side: str, size_usd: float, price: float) -> dict:
        """Place a market order on OKX."""
        if not self.has_api_keys():
            return {"ok": False, "reason": "No API keys configured"}

        inst_id = self._inst_id(symbol)
        # Calculate contract quantity: sz = number of contracts
        ct_val = await self._get_contract_size(inst_id)
        if price <= 0 or ct_val <= 0:
            return {"ok": False, "reason": f"Invalid price ({price}) or contract value ({ct_val})"}
        n_contracts = max(1, int(size_usd / (price * ct_val)))

        path = "/api/v5/trade/order"
        body = json.dumps({
            "instId": inst_id,
            "tdMode": "cross",
            "side": "buy" if side == "long" else "sell",
            "posSide": "long" if side == "long" else "short",
            "ordType": "market",
            "sz": str(n_contracts),
        })

        data = await self._okx_request("POST", path, body)
        if data.get("code") == "0" and data.get("data") and len(data["data"]) > 0:
            ord_id = data["data"][0].get("ordId", "")
            print(f"[OKX LIVE] Order placed: {side} {inst_id} x{n_contracts} ordId={ord_id}")
            return {"ok": True, "orderId": ord_id, "price": price, "contracts": n_contracts}
        msg = data.get("msg", "")
        if not msg and data.get("data") and len(data["data"]) > 0:
            msg = data["data"][0].get("sMsg", "Unknown error")
        print(f"[OKX LIVE] Order failed: {msg}")
        return {"ok": False, "reason": msg}

    async def _close_order_live(self, symbol: str, side: str = "") -> dict:
        """Close position on OKX by closing via the close-position API."""
        if not self.has_api_keys():
            return {"ok": False, "reason": "No API keys configured"}

        inst_id = self._inst_id(symbol)
        # Use the close-position endpoint for simplicity
        path = "/api/v5/trade/close-position"
        close_body = {
            "instId": inst_id,
            "mgnMode": "cross",
        }
        if side:
            close_body["posSide"] = side  # required for hedge mode
        body = json.dumps(close_body)

        data = await self._okx_request("POST", path, body)
        if data.get("code") == "0":
            print(f"[OKX LIVE] Position closed: {inst_id}")
            return {"ok": True}
        msg = data.get("msg", "Unknown error")
        print(f"[OKX LIVE] Close failed: {msg}, trying market order fallback")

        # Fallback: query position and place opposite market order
        pos_data = await self._okx_request("GET", f"/api/v5/account/positions?instId={inst_id}")
        if pos_data.get("code") == "0" and pos_data.get("data"):
            pos = pos_data["data"][0]
            pos_amt = abs(float(pos.get("pos", 0)))
            if pos_amt > 0:
                close_side = "sell" if float(pos.get("pos", 0)) > 0 else "buy"
                pos_side = "long" if float(pos.get("pos", 0)) > 0 else "short"
                close_body = json.dumps({
                    "instId": inst_id,
                    "tdMode": "cross",
                    "side": close_side,
                    "posSide": pos_side,
                    "ordType": "market",
                    "sz": str(pos_amt),
                    "reduceOnly": "true",
                })
                close_data = await self._okx_request("POST", "/api/v5/trade/order", close_body)
                if close_data.get("code") == "0":
                    print(f"[OKX LIVE] Position closed via market order: {inst_id}")
                    return {"ok": True}

        return {"ok": False, "reason": msg}

    # ── Agent revival ─────────────────────────────────────────────────────

    def revive(self):
        """Revive the agent after shutdown (manual intervention)."""
        self.state.is_alive = True
        self.state.shutdown_reason = ""
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.last_daily_reset = time.time()
