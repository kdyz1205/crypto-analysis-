"""
Agent Brain — The autonomous trading intelligence.

Responsibilities:
1. Generate signals from OHLCV data (using the optimized strategy)
2. Manage positions (trailing stops, EMA exits)
3. Self-evolve: after every N trades, evaluate performance and mutate strategy params
4. Stay alive: hard risk limits, conservative sizing

The brain runs as a background loop inside the FastAPI app.
"""

import time
import asyncio
import random
import math
import json
from pathlib import Path
from typing import Any

import numpy as np

from .okx_trader import OKXTrader, AgentState, RiskLimits, Position
from .backtest_service import (
    _mfi, _sma, _ema, _atr, _rsi, _bb_upper, BacktestParams, run_backtest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_STATE_FILE = PROJECT_ROOT / "autoresearch" / "agent_state.json"
EVOLUTION_LOG = PROJECT_ROOT / "autoresearch" / "evolution_log.tsv"

# Symbols to monitor
WATCH_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"]
SIGNAL_INTERVAL = "4h"  # timeframe for signal generation
TICK_INTERVAL_SEC = 60  # check every 60 seconds

# Evolution: mutate params after this many trades
EVOLVE_EVERY_N_TRADES = 10
MIN_TRADES_FOR_EVAL = 5


class AgentBrain:
    """The autonomous trading agent."""

    def __init__(self, trader: OKXTrader | None = None):
        self.trader = trader or OKXTrader()
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_signals: dict[str, dict] = {}  # symbol -> signal info
        self._load_state()

    # ── State persistence ─────────────────────────────────────────────────

    def _load_state(self):
        """Load agent state from disk."""
        if AGENT_STATE_FILE.exists():
            try:
                data = json.loads(AGENT_STATE_FILE.read_text())
                s = self.trader.state
                s.mode = data.get("mode", "paper")
                s.equity = data.get("equity", 10000.0)
                s.peak_equity = data.get("peak_equity", 10000.0)
                s.cash = data.get("cash", 10000.0)
                s.generation = data.get("generation", 0)
                s.total_trades = data.get("total_trades", 0)
                s.total_pnl_usd = data.get("total_pnl_usd", 0.0)
                s.win_count = data.get("win_count", 0)
                s.loss_count = data.get("loss_count", 0)
                s.is_alive = data.get("is_alive", True)
                s.shutdown_reason = data.get("shutdown_reason", "")
                if "strategy_params" in data:
                    s.strategy_params.update(data["strategy_params"])
                print(f"[Agent] Loaded state: gen={s.generation} equity={s.equity:.2f} trades={s.total_trades}")
            except Exception as e:
                print(f"[Agent] Failed to load state: {e}")

    def _save_state(self):
        """Save agent state to disk."""
        try:
            AGENT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "mode": self.trader.state.mode,
                "equity": self.trader.state.equity,
                "peak_equity": self.trader.state.peak_equity,
                "cash": self.trader.state.cash,
                "generation": self.trader.state.generation,
                "total_trades": self.trader.state.total_trades,
                "total_pnl_usd": self.trader.state.total_pnl_usd,
                "win_count": self.trader.state.win_count,
                "loss_count": self.trader.state.loss_count,
                "is_alive": self.trader.state.is_alive,
                "shutdown_reason": self.trader.state.shutdown_reason,
                "strategy_params": self.trader.state.strategy_params,
            }
            AGENT_STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Agent] Failed to save state: {e}")

    # ── Signal generation ─────────────────────────────────────────────────

    async def generate_signal(self, symbol: str) -> dict | None:
        """
        Analyze current market data for a symbol and return a signal.
        Returns: {'action': 'long'|'short'|'close'|None, 'confidence': float, 'reason': str}
        """
        try:
            from .data_service import get_ohlcv_with_df
            df, _ = await get_ohlcv_with_df(symbol, SIGNAL_INTERVAL, days=90)
            if df is None or df.is_empty() or len(df) < 60:
                return None
        except Exception as e:
            print(f"[Agent] Data error for {symbol}: {e}")
            return None

        params = self.trader.state.strategy_params
        close = df["close"].to_numpy().astype(float)
        high = df["high"].to_numpy().astype(float)
        low = df["low"].to_numpy().astype(float)
        volume = df["volume"].to_numpy().astype(float)

        mfi = _mfi(high, low, close, volume, params["mfi_period"])
        ma_fast = _sma(close, params["ma_fast"])
        ema = _ema(close, params["ema_span"])
        ma_slow = _sma(close, params["ma_slow"])
        atr = _atr(high, low, close, params["atr_period"])
        rsi = _rsi(close, 14)
        bb_up = _bb_upper(close, params["bb_period"], params["bb_std"])

        i = len(close) - 1  # latest bar
        if any(np.isnan(x[i]) for x in [mfi, ma_fast, ema, ma_slow, atr, rsi, bb_up]):
            return None

        price = close[i]

        # Check if we have an existing position
        if symbol in self.trader.state.positions:
            pos = self.trader.state.positions[symbol]
            # Check trailing stop / EMA breakdown
            if pos.side == "long":
                if close[i] < ema[i]:
                    return {"action": "close", "confidence": 0.8, "reason": "EMA breakdown"}
                if pos.unrealized_pnl < -3.0:  # -3% hard stop
                    return {"action": "close", "confidence": 1.0, "reason": "Hard stop -3%"}
            return None  # hold position

        # Entry signals
        rsi_ok = params["rsi_low"] < rsi[i] < params["rsi_high"]
        ma_stack_long = (price > ma_fast[i] and ma_fast[i] > ema[i] and ema[i] > ma_slow[i])
        mfi_bull = mfi[i] > params["mfi_threshold"]

        # Confidence: how many bars of MA stacking (lookback 5)
        stack_count = 0
        for j in range(max(0, i - 5), i + 1):
            if (not np.isnan(ma_fast[j]) and not np.isnan(ema[j]) and not np.isnan(ma_slow[j])
                    and close[j] > ma_fast[j] > ema[j] > ma_slow[j]):
                stack_count += 1
        confidence = min(stack_count / 5.0, 1.0)

        # Volume confirmation: current vol > average vol
        vol_avg = np.mean(volume[max(0, i-20):i]) if i > 20 else volume[i]
        vol_confirm = volume[i] > vol_avg * 0.8

        if ma_stack_long and mfi_bull and rsi_ok and vol_confirm and confidence >= 0.4:
            return {
                "action": "long",
                "confidence": round(confidence, 2),
                "reason": f"MA stacking ({stack_count}/5) + MFI={mfi[i]:.0f} RSI={rsi[i]:.0f}",
                "sl": ema[i] - params["atr_sl_mult"] * atr[i],
                "tp": bb_up[i],
            }

        return None

    # ── Position management ───────────────────────────────────────────────

    async def manage_positions(self):
        """Check all open positions for exits."""
        for symbol in list(self.trader.state.positions.keys()):
            signal = await self.generate_signal(symbol)
            if signal and signal["action"] == "close":
                result = await self.trader.close_position(symbol, signal["reason"])
                if result["ok"]:
                    print(f"[Agent] Closed {symbol}: {signal['reason']} PnL={result['pnl_pct']}%")
                    self._save_state()

    # ── Evolution: self-improving strategy ─────────────────────────────────

    def _evaluate_recent_performance(self) -> dict:
        """Evaluate recent trade performance for evolution decisions."""
        trades = self.trader.state.trade_history[-EVOLVE_EVERY_N_TRADES:]
        if len(trades) < MIN_TRADES_FOR_EVAL:
            return {"ready": False}

        pnls = [t.pnl_pct for t in trades]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg_pnl = np.mean(pnls)
        sharpe = np.mean(pnls) / max(np.std(pnls), 1e-8)

        return {
            "ready": True,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "num_trades": len(trades),
        }

    def evolve(self):
        """
        Mutate strategy parameters based on recent performance.
        Conservative mutations: small % changes to avoid catastrophic shifts.
        """
        perf = self._evaluate_recent_performance()
        if not perf["ready"]:
            return

        params = self.trader.state.strategy_params
        old_params = dict(params)

        # Mutation strength: smaller when performing well, larger when performing poorly
        if perf["sharpe"] > 0:
            mutation_rate = 0.05  # 5% mutation when profitable
        else:
            mutation_rate = 0.10  # 10% mutation when losing

        # Mutate numeric params
        int_params = ["mfi_period", "ma_fast", "ema_span", "ma_slow", "atr_period", "bb_period"]
        float_params = ["atr_sl_mult", "bb_std", "rsi_low", "rsi_high", "mfi_threshold", "trailing_trigger_atr"]

        # Pick 1-2 params to mutate (not everything at once)
        all_params = int_params + float_params
        n_mutations = random.randint(1, 2)
        to_mutate = random.sample(all_params, min(n_mutations, len(all_params)))

        for key in to_mutate:
            val = params[key]
            delta = val * mutation_rate * random.choice([-1, 1])
            new_val = val + delta

            # Enforce bounds
            bounds = {
                "mfi_period": (5, 30), "ma_fast": (3, 20), "ema_span": (10, 40),
                "ma_slow": (30, 90), "atr_period": (5, 25), "bb_period": (10, 35),
                "atr_sl_mult": (0.5, 5.0), "bb_std": (1.5, 4.0),
                "rsi_low": (20, 50), "rsi_high": (55, 85),
                "mfi_threshold": (30, 70), "trailing_trigger_atr": (0.5, 3.0),
            }
            lo, hi = bounds.get(key, (0, 100))
            new_val = max(lo, min(hi, new_val))

            if key in int_params:
                new_val = int(round(new_val))

            params[key] = new_val

        # Enforce MA ordering: ma_fast < ema_span < ma_slow
        if params["ma_fast"] >= params["ema_span"]:
            params["ema_span"] = params["ma_fast"] + 5
        if params["ema_span"] >= params["ma_slow"]:
            params["ma_slow"] = params["ema_span"] + 10

        self.trader.state.generation += 1

        # Log evolution
        try:
            EVOLUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
            if not EVOLUTION_LOG.exists():
                EVOLUTION_LOG.write_text("generation\tsharpe\twin_rate\tmutated\told_values\tnew_values\n")
            with open(EVOLUTION_LOG, "a") as f:
                mutated_str = ",".join(to_mutate)
                old_str = ",".join(f"{k}={old_params[k]}" for k in to_mutate)
                new_str = ",".join(f"{k}={params[k]}" for k in to_mutate)
                f.write(f"{self.trader.state.generation}\t{perf['sharpe']:.4f}\t{perf['win_rate']:.1f}\t{mutated_str}\t{old_str}\t{new_str}\n")
        except Exception:
            pass

        print(f"[Agent] Evolution gen={self.trader.state.generation}: mutated {to_mutate}")
        self._save_state()

    # ── Main loop ─────────────────────────────────────────────────────────

    async def tick(self):
        """One iteration of the agent loop."""
        if not self.trader.state.is_alive:
            return

        self.trader.check_daily_reset()

        # Update existing positions
        await self.trader.update_positions()
        await self.manage_positions()

        # Look for new entry signals
        for symbol in WATCH_SYMBOLS:
            if symbol in self.trader.state.positions:
                continue  # already have a position

            signal = await self.generate_signal(symbol)
            if signal is None or signal["action"] is None:
                continue

            if signal["action"] == "long" and signal["confidence"] >= 0.6:
                # Size: confidence × max_position_pct × equity
                size = signal["confidence"] * self.trader.risk.max_position_pct * self.trader.state.equity
                result = await self.trader.open_position(symbol, "long", size)
                if result["ok"]:
                    print(f"[Agent] Opened LONG {symbol} @ {result['price']} size=${size:.0f} conf={signal['confidence']}")
                    self._last_signals[symbol] = signal
                    self._save_state()

        # Check if it's time to evolve
        if (self.trader.state.total_trades > 0 and
                self.trader.state.total_trades % EVOLVE_EVERY_N_TRADES == 0):
            self.evolve()

    async def run_loop(self):
        """Background loop: tick every TICK_INTERVAL_SEC."""
        self._running = True
        print(f"[Agent] Started. Mode={self.trader.state.mode} Equity=${self.trader.state.equity:.2f} Gen={self.trader.state.generation}")

        while self._running:
            try:
                await self.tick()
            except Exception as e:
                print(f"[Agent] Tick error: {e}")
            await asyncio.sleep(TICK_INTERVAL_SEC)

    def start(self):
        """Start the agent background loop."""
        if self._task and not self._task.done():
            return  # already running
        self._task = asyncio.create_task(self.run_loop())

    def stop(self):
        """Stop the agent."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._save_state()
        print("[Agent] Stopped.")

    def get_status(self) -> dict:
        """Get current agent status for the UI."""
        return {
            **self.trader.state.to_dict(),
            "running": self._running,
            "watch_symbols": WATCH_SYMBOLS,
            "signal_interval": SIGNAL_INTERVAL,
            "last_signals": {
                k: {"action": v.get("action"), "confidence": v.get("confidence"), "reason": v.get("reason")}
                for k, v in self._last_signals.items()
            },
        }
