"""
Agent Brain — V6 Self-Evolving Trading Harness with 举一反三 (Learn-by-Analogy).

Closed-loop phases: OBSERVE → LEARN → RESEARCH → REASON → REFINE → ACT → CHECKPOINT → REFLECT

Layers:
1. Trend ordering: P > MA5 > MA8 > EMA21 > MA55 (long) or mirror (short)
2. Fanning distance: adjacent MA gaps within thresholds (not too spread)
3. Slope momentum: all MAs sloping in trend direction
4. BB position: price < BB_upper (long) or price > BB_lower (short)
5. Volume confirmation: reject low-volume breakouts

Pre-trade checklist (institutional-grade):
- Data completeness, signal validity, strategy enabled
- Risk limits, SL sanity, duplicate suppression
- Consecutive loss protection, conflicting position check

举一反三 (Learn-by-Analogy):
- Lessons learned from one symbol are generalized and applied across all symbols
- Market regime detection adapts strategy behavior globally
- Pattern failure recognition persists across cycles
"""

import os
import time
import asyncio
import random
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import httpx
import numpy as np

from .okx_trader import OKXTrader, AgentState, RiskLimits, Position
from .backtest_service import _sma, _ema, _atr, _bb_upper


def _bb_lower(close, period, std_mult):
    """Bollinger Band lower."""
    sma = _sma(close, period)
    n = len(close)
    bb = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        std = np.std(window)
        bb[i] = sma[i] - std_mult * std
    return bb


def _slope(arr, length, i):
    """Percentage slope of arr over `length` bars ending at index i."""
    if i < length or np.isnan(arr[i]) or np.isnan(arr[i - length]):
        return 0.0
    prev = arr[i - length]
    if prev == 0:
        return 0.0
    return (arr[i] - prev) / prev * 100


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_STATE_FILE = PROJECT_ROOT / "agent_state.json"
EVOLUTION_LOG = PROJECT_ROOT / "evolution_log.tsv"
TRADE_AUDIT_LOG = PROJECT_ROOT / "trade_audit.jsonl"
LESSONS_FILE = PROJECT_ROOT / "lessons_ledger.json"

# Symbols to monitor
WATCH_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"]
SIGNAL_INTERVAL = "4h"
TICK_INTERVAL_SEC = 60

# Evolution
EVOLVE_EVERY_N_TRADES = 10
MIN_TRADES_FOR_EVAL = 5

# Signal suppression: don't fire same signal within this window (seconds)
SIGNAL_DEDUP_WINDOW_SEC = 14400  # 4 hours


class LessonsLedger:
    """
    举一反三 — Learn from one trade/pattern and generalize across all symbols.

    Categories:
    - regime: market regime observations (trending/ranging/volatile)
    - pattern: pattern failures or successes
    - risk: risk management insights
    - param: parameter optimization insights
    """

    MAX_LESSONS = 50

    def __init__(self):
        self.lessons: list[dict] = []
        self.market_regime: str = "unknown"
        self.regime_confidence: float = 0.0
        self.cycle: int = 0
        self._load()

    def _load(self):
        if LESSONS_FILE.exists():
            try:
                data = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
                self.lessons = data.get("lessons", [])
                self.market_regime = data.get("market_regime", "unknown")
                self.regime_confidence = data.get("regime_confidence", 0.0)
                self.cycle = data.get("cycle", 0)
            except Exception as e:
                logging.warning(f"Failed to load lessons ledger: {e}")

    def save(self):
        try:
            LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            LESSONS_FILE.write_text(json.dumps({
                "lessons": self.lessons[-self.MAX_LESSONS:],
                "market_regime": self.market_regime,
                "regime_confidence": self.regime_confidence,
                "cycle": self.cycle,
            }, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            print(f"[Lessons] Save failed: {e}")

    def add(self, category: str, lesson: str, symbol: str = "ALL", data: dict | None = None):
        """Add a lesson. 举一反三: lessons tagged 'ALL' apply to every symbol."""
        entry = {
            "cycle": self.cycle,
            "time": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "symbol": symbol,
            "lesson": lesson,
            "data": data or {},
        }
        self.lessons.append(entry)
        if len(self.lessons) > self.MAX_LESSONS:
            self.lessons = self.lessons[-self.MAX_LESSONS:]
        print(f"[举一反三] {category}: {lesson}")

    def get_applicable(self, symbol: str) -> list[dict]:
        """Get lessons applicable to a symbol (its own + ALL)."""
        return [l for l in self.lessons if l["symbol"] in (symbol, "ALL")]

    def has_recent_warning(self, category: str, lookback: int = 5) -> bool:
        """Check if there's a warning in the last N lessons."""
        recent = self.lessons[-lookback:] if self.lessons else []
        return any(l["category"] == category for l in recent)

    def detect_regime(self, close: np.ndarray, atr: np.ndarray) -> str:
        """Detect market regime from price action."""
        if len(close) < 60:
            return "unknown"

        # Trend strength: compare MA20 slope
        ma20 = _sma(close, 20)
        ma50 = _sma(close, 50)
        i = len(close) - 1

        if np.isnan(ma20[i]) or np.isnan(ma50[i]):
            return "unknown"

        slope_20 = _slope(ma20, 10, i)
        atr_val = atr[i] if not np.isnan(atr[i]) else 0
        atr_pct = atr_val / close[i] * 100 if close[i] > 0 else 0

        if abs(slope_20) > 0.5 and atr_pct < 4.0:
            regime = "trending"
            confidence = min(abs(slope_20) / 2.0, 1.0)
        elif atr_pct > 4.0:
            regime = "volatile"
            confidence = min(atr_pct / 8.0, 1.0)
        elif abs(slope_20) < 0.2:
            regime = "ranging"
            confidence = 1.0 - abs(slope_20) / 0.2
        else:
            regime = "mixed"
            confidence = 0.5

        self.market_regime = regime
        self.regime_confidence = round(confidence, 2)
        return regime

    def learn_from_trade(self, symbol: str, side: str, pnl_pct: float,
                         entry_price: float, exit_price: float,
                         regime: str, vol_regime: str):
        """举一反三: Learn from a closed trade and generalize the insight."""
        if pnl_pct > 2.0:
            self.add("pattern", f"{side} in {regime}/{vol_regime} market yielded +{pnl_pct:.1f}% — favorable setup",
                     symbol="ALL", data={"regime": regime, "vol": vol_regime, "side": side, "pnl": pnl_pct})
        elif pnl_pct < -1.5:
            self.add("risk", f"{side} in {regime}/{vol_regime} lost {pnl_pct:.1f}% — avoid this regime for {side}",
                     symbol="ALL", data={"regime": regime, "vol": vol_regime, "side": side, "pnl": pnl_pct})

        # Cross-symbol generalization
        losses_in_regime = [l for l in self.lessons[-20:]
                           if l["category"] == "risk"
                           and l.get("data", {}).get("regime") == regime
                           and l.get("data", {}).get("side") == side]
        if len(losses_in_regime) >= 3:
            self.add("regime", f"3+ losses in {regime} market for {side} — consider pausing {side} entries globally",
                     symbol="ALL", data={"regime": regime, "side": side, "loss_count": len(losses_in_regime)})

    def should_skip_regime(self, side: str) -> bool:
        """Check if recent lessons warn against trading this side in current regime."""
        recent_warnings = [l for l in self.lessons[-10:]
                          if l["category"] == "regime"
                          and l.get("data", {}).get("side") == side
                          and l.get("data", {}).get("regime") == self.market_regime]
        return len(recent_warnings) > 0

    def get_summary(self) -> dict:
        """Return summary for UI display."""
        cats = {}
        for l in self.lessons:
            c = l["category"]
            cats[c] = cats.get(c, 0) + 1

        return {
            "cycle": self.cycle,
            "total_lessons": len(self.lessons),
            "by_category": cats,
            "market_regime": self.market_regime,
            "regime_confidence": self.regime_confidence,
            "recent": self.lessons[-5:] if self.lessons else [],
        }


def _audit_log(event: dict):
    """Append a structured JSON log entry to the trade audit log."""
    try:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        TRADE_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:
        print(f"[Audit] Log write failed: {e}")


class PreTradeChecklist:
    """Institutional-grade pre-trade validation. Returns (pass, reasons) tuple."""

    @staticmethod
    def validate(agent, symbol: str, signal: dict) -> tuple[bool, list[str]]:
        failures = []

        # 1. Data completeness — signal must have required fields
        required = ["action", "confidence", "reason", "sl", "tp", "price"]
        for field in required:
            if field not in signal or signal[field] is None:
                failures.append(f"Missing field: {field}")

        # 2. Signal validity — confidence threshold
        if signal.get("confidence", 0) < 0.6:
            failures.append(f"Low confidence: {signal.get('confidence', 0):.2f} < 0.60")

        # 3. Strategy enabled
        if not agent.trader.state.is_alive:
            failures.append(f"Agent shutdown: {agent.trader.state.shutdown_reason}")

        # 4. Risk limits
        can, reason = agent.trader.can_trade(symbol)
        if not can:
            failures.append(f"Risk check failed: {reason}")

        # 5. Stop loss must exist, be reasonable, and be directionally correct
        price = signal.get("price", 0)
        sl = signal.get("sl", 0)
        action = signal.get("action", "")
        if action in ("long", "short") and (not sl or sl <= 0):
            failures.append("Missing or zero stop loss for entry signal")
        elif price > 0 and sl > 0:
            sl_dist_pct = abs(price - sl) / price * 100
            if sl_dist_pct > 10:
                failures.append(f"SL too far: {sl_dist_pct:.1f}% (max 10%)")
            if sl_dist_pct < 0.1:
                failures.append(f"SL too tight: {sl_dist_pct:.3f}% (min 0.1%)")
            # SL must be below price for longs, above price for shorts
            if action == "long" and sl >= price:
                failures.append(f"SL above price for long: SL={sl:.6f} >= price={price:.6f}")
            if action == "short" and sl <= price:
                failures.append(f"SL below price for short: SL={sl:.6f} <= price={price:.6f}")

        # 6. Duplicate signal suppression
        last_sig = agent._last_signals.get(symbol)
        if last_sig and last_sig.get("action") == signal.get("action"):
            last_time = last_sig.get("_ts", 0)
            if time.time() - last_time < SIGNAL_DEDUP_WINDOW_SEC:
                failures.append(f"Duplicate signal suppressed (same {signal['action']} within {SIGNAL_DEDUP_WINDOW_SEC}s)")

        # 7. No conflicting position — any open position blocks a new entry signal
        if symbol in agent.trader.state.positions:
            pos = agent.trader.state.positions[symbol]
            sig_action = signal.get("action")
            if sig_action in ("long", "short"):
                failures.append(f"Already in {pos.side} position for {symbol} — close it before entering {sig_action}")

        # 8. Consecutive loss protection
        recent = agent.trader.state.trade_history[-5:]
        consec_losses = 0
        for t in reversed(recent):
            if t.pnl_usd < 0:
                consec_losses += 1
            else:
                break
        if consec_losses >= 3:
            failures.append(f"Consecutive loss protection: {consec_losses} losses in a row")

        passed = len(failures) == 0
        return passed, failures


class AgentBrain:
    """Self-evolving trading agent — V6 strategy + 举一反三 lessons + closed-loop harness."""

    def __init__(self, trader: OKXTrader | None = None):
        self.trader = trader or OKXTrader()
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_signals: dict[str, dict] = {}
        self._signal_history: list[dict] = []  # all signals for UI review
        self.lessons = LessonsLedger()
        self._cycle_phase: str = "idle"  # current harness phase for UI
        self._last_evolved_at: int = 0
        self._telegram_config: dict | None = None
        self._load_state()
        self._load_telegram_config()

    # ── Telegram notifications ────────────────────────────────────────────

    def _load_telegram_config(self):
        """Load Telegram config from environment variables."""
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if bot_token and chat_id:
            self._telegram_config = {
                "bot_token": bot_token,
                "chat_id": chat_id,
                "notify_signals": True,
                "notify_fills": True,
                "notify_errors": True,
                "notify_daily": True,
            }
            print(f"[Telegram] Config loaded from env: chat_id={chat_id}")

    async def _notify_telegram(self, message: str, category: str = "general"):
        """Send a notification to Telegram if configured."""
        cfg = self._telegram_config
        if not cfg:
            return
        # Check if this category is enabled
        cat_map = {
            "signal": "notify_signals",
            "fill": "notify_fills",
            "error": "notify_errors",
            "daily": "notify_daily",
        }
        flag = cat_map.get(category, None)
        if flag and not cfg.get(flag, True):
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage",
                    json={
                        "chat_id": cfg["chat_id"],
                        "text": message,
                        "parse_mode": "HTML",
                    },
                )
        except Exception as e:
            print(f"[Telegram] Send failed: {e}")

    # ── State persistence ─────────────────────────────────────────────────

    def _load_state(self):
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

    # ── V6 Signal generation ──────────────────────────────────────────────

    async def generate_signal(self, symbol: str) -> dict | None:
        """
        V6 four-layer filtered trend following signal.
        Returns: {'action': 'long'|'short'|'close'|None, 'confidence': float, 'reason': str}
        """
        try:
            from .data_service import get_ohlcv_with_df
            # Need at least 55+ bars for MA55; fetch 365 days for 4h = ~2190 bars
            fetch_days = {"5m": 7, "15m": 21, "1h": 90, "4h": 365, "1d": 365 * 3}.get(SIGNAL_INTERVAL, 90)
            df, _ = await get_ohlcv_with_df(symbol, SIGNAL_INTERVAL, days=fetch_days)
            if df is None or df.is_empty() or len(df) < 60:
                print(f"[Agent] {symbol}: insufficient data ({len(df) if df is not None else 0} bars)")
                return None
        except Exception as e:
            print(f"[Agent] Data error for {symbol}: {e}")
            return None

        p = self.trader.state.strategy_params
        close = df["close"].to_numpy().astype(float)
        high = df["high"].to_numpy().astype(float)
        low = df["low"].to_numpy().astype(float)
        vol = df["volume"].to_numpy().astype(float) if "volume" in df.columns else None

        # Detect market regime (updates lessons ledger)
        atr_temp = _atr(high, low, close, 14)
        self.lessons.detect_regime(close, atr_temp)

        # Compute indicators
        ma5 = _sma(close, p["ma5_len"])
        ma8 = _sma(close, p["ma8_len"])
        ema21 = _ema(close, p["ema21_len"])
        ma55 = _sma(close, p["ma55_len"])
        bb_up = _bb_upper(close, p["bb_length"], p["bb_std_dev"])
        bb_lo = _bb_lower(close, p["bb_length"], p["bb_std_dev"])
        atr = _atr(high, low, close, p["atr_period"])

        i = len(close) - 1  # latest bar
        indicators = [ma5, ma8, ema21, ma55, bb_up, bb_lo, atr]
        if any(np.isnan(x[i]) for x in indicators):
            print(f"[Agent] {symbol}: NaN indicators at bar {i}")
            return None

        price = close[i]
        if np.isnan(price) or price <= 0:
            return None
        slope_len = p["slope_len"]
        slope_thresh = p["slope_threshold"]

        # ── ATR-normalized adaptive thresholds ──
        atr_pct = atr[i] / price * 100  # ATR as percentage of price
        atr_dist_scale = max(1.0, atr_pct / 2.0)
        atr_slope_scale = max(1.0, atr_pct / 1.5)

        adapted_dist_5_8 = p["dist_ma5_ma8"] * atr_dist_scale
        adapted_dist_8_21 = p["dist_ma8_ema21"] * atr_dist_scale
        adapted_dist_21_55 = p["dist_ema21_ma55"] * atr_dist_scale
        adapted_slope_thresh = slope_thresh * atr_slope_scale

        if atr_pct < 1.5:
            volatility_regime = "low"
        elif atr_pct < 4.0:
            volatility_regime = "normal"
        else:
            volatility_regime = "high"

        # ── Check existing position exits ──
        if symbol in self.trader.state.positions:
            pos = self.trader.state.positions[symbol]
            if pos.side == "long":
                # Long SL: price < MA55
                if price < ma55[i]:
                    return {"action": "close", "confidence": 1.0, "reason": "Long SL: price < MA55", "volatility_regime": volatility_regime}
                # Long TP: price >= BB_upper
                if price >= bb_up[i]:
                    return {"action": "close", "confidence": 0.9, "reason": "Long TP: price hit BB_upper", "volatility_regime": volatility_regime}
            elif pos.side == "short":
                # Short SL: price > MA55
                if price > ma55[i]:
                    return {"action": "close", "confidence": 1.0, "reason": "Short SL: price > MA55", "volatility_regime": volatility_regime}
                # Short TP: price <= BB_lower
                if price <= bb_lo[i]:
                    return {"action": "close", "confidence": 0.9, "reason": "Short TP: price hit BB_lower", "volatility_regime": volatility_regime}
            return None  # hold

        # ── Layer 1: Trend ordering ──
        long_order = (price > ma5[i] > ma8[i] > ema21[i] > ma55[i])
        short_order = (price < ma5[i] < ma8[i] < ema21[i] < ma55[i])

        if not long_order and not short_order:
            print(f"[Agent] {symbol}: L1 fail — P={price:.2f} MA5={ma5[i]:.2f} MA8={ma8[i]:.2f} EMA21={ema21[i]:.2f} MA55={ma55[i]:.2f}")
            return None

        # ── Layer 2: Fanning distance (MAs not too spread apart) ──
        def pct_dist(a, b):
            if np.isnan(a) or np.isnan(b):
                return 999.0  # force fail
            return abs(a - b) / max(abs(b), 1e-10) * 100

        dist_5_8 = pct_dist(ma5[i], ma8[i])
        dist_8_21 = pct_dist(ma8[i], ema21[i])
        dist_21_55 = pct_dist(ema21[i], ma55[i])

        fan_ok = (
            dist_5_8 < adapted_dist_5_8
            and dist_8_21 < adapted_dist_8_21
            and dist_21_55 < adapted_dist_21_55
        )
        if not fan_ok:
            return None

        # ── Layer 3: Slope momentum (all MAs trending in the right direction) ──
        s_ma5 = _slope(ma5, slope_len, i)
        s_ma8 = _slope(ma8, slope_len, i)
        s_ema21 = _slope(ema21, slope_len, i)
        s_ma55 = _slope(ma55, slope_len, i)

        if long_order:
            slopes_ok = all(s > adapted_slope_thresh for s in [s_ma5, s_ma8, s_ema21, s_ma55])
        else:
            slopes_ok = all(s < -adapted_slope_thresh for s in [s_ma5, s_ma8, s_ema21, s_ma55])

        if not slopes_ok:
            return None

        # ── Layer 4: BB position filter ──
        if long_order and price >= bb_up[i]:
            return None  # already at upper band, too late
        if short_order and price <= bb_lo[i]:
            return None  # already at lower band, too late

        # ── Confidence: how many bars of consistent ordering (lookback 5) ──
        stack_count = 0
        for j in range(max(0, i - 5), i + 1):
            if any(np.isnan(x[j]) for x in [ma5, ma8, ema21, ma55]):
                continue
            if long_order and close[j] > ma5[j] > ma8[j] > ema21[j] > ma55[j]:
                stack_count += 1
            elif short_order and close[j] < ma5[j] < ma8[j] < ema21[j] < ma55[j]:
                stack_count += 1
        confidence = min(stack_count / 5.0, 1.0)

        # Boost confidence when expanding volatility aligns with trending entry
        if volatility_regime in ("normal", "high"):
            confidence = min(confidence + 0.1, 1.0)

        # ── Layer 5: Volume confirmation ──
        if vol is not None and len(vol) > 20:
            vol_slice = vol[max(0, i - 20):i] if i >= 20 else vol[:i+1]
            vol_ma = np.nanmean(vol_slice) if len(vol_slice) > 0 else 0
            if vol_ma > 0 and not np.isnan(vol_ma) and not np.isnan(vol[i]):
                vol_ratio = vol[i] / vol_ma
                if vol_ratio < 0.5:
                    # Very low volume — likely fake breakout
                    return None
                if vol_ratio > 1.5:
                    confidence = min(confidence + 0.1, 1.0)  # High volume = strong confirmation

        if confidence < 0.4:
            return None

        # Build signal
        if long_order:
            return {
                "action": "long",
                "confidence": round(confidence, 2),
                "reason": (
                    f"V6 Long: ordering OK, dist={dist_5_8:.1f}/{dist_8_21:.1f}/{dist_21_55:.1f}%, "
                    f"slopes={s_ma5:.2f}/{s_ma8:.2f}/{s_ema21:.2f}/{s_ma55:.2f}%, "
                    f"vol={volatility_regime} atr={atr_pct:.2f}%"
                ),
                "sl": round(float(ma55[i]), 6),
                "tp": round(float(bb_up[i]), 6),
                "price": round(float(price), 6),
                "volatility_regime": volatility_regime,
            }
        else:
            return {
                "action": "short",
                "confidence": round(confidence, 2),
                "reason": (
                    f"V6 Short: ordering OK, dist={dist_5_8:.1f}/{dist_8_21:.1f}/{dist_21_55:.1f}%, "
                    f"slopes={s_ma5:.2f}/{s_ma8:.2f}/{s_ema21:.2f}/{s_ma55:.2f}%, "
                    f"vol={volatility_regime} atr={atr_pct:.2f}%"
                ),
                "sl": round(float(ma55[i]), 6),
                "tp": round(float(bb_lo[i]), 6),
                "price": round(float(price), 6),
                "volatility_regime": volatility_regime,
            }

    # ── Position management ───────────────────────────────────────────────

    async def manage_positions(self):
        """Check all open positions for V6 exits. On close, feed lessons ledger."""
        for symbol in list(self.trader.state.positions.keys()):
            pos = self.trader.state.positions[symbol]
            signal = await self.generate_signal(symbol)
            if signal and signal["action"] == "close":
                result = await self.trader.close_position(symbol, signal["reason"])
                if result["ok"]:
                    pnl_pct = result.get('pnl_pct', 0)
                    pnl_usd = result.get('pnl_usd', 0)
                    print(f"[Agent] Closed {symbol}: {signal['reason']} PnL={pnl_pct}%")

                    # Telegram notification for position close
                    emoji = "🟢" if pnl_usd >= 0 else "🔴"
                    await self._notify_telegram(
                        f"{emoji} <b>平仓 {symbol}</b>\n"
                        f"原因: {signal['reason']}\n"
                        f"盈亏: {pnl_pct:+.2f}% (${pnl_usd:+.2f})\n"
                        f"💰 权益: ${self.trader.state.equity:.2f}",
                        category="fill",
                    )

                    # 举一反三: Learn from this trade
                    self.lessons.learn_from_trade(
                        symbol=symbol,
                        side=pos.side,
                        pnl_pct=pnl_pct,
                        entry_price=pos.entry_price,
                        exit_price=result.get("exit_price", 0),
                        regime=self.lessons.market_regime,
                        vol_regime=signal.get("volatility_regime", "unknown"),
                    )

                    _audit_log({
                        "event": "close_position",
                        "symbol": symbol,
                        "reason": signal["reason"],
                        "pnl_pct": pnl_pct,
                        "pnl_usd": result.get("pnl_usd", 0),
                        "exit_price": result.get("exit_price", 0),
                        "mode": self.trader.state.mode,
                        "equity_after": self.trader.state.equity,
                        "market_regime": self.lessons.market_regime,
                    })
                    self._save_state()
                    self.lessons.save()

    # ── Evolution: self-improving V6 params ───────────────────────────────

    def _evaluate_recent_performance(self) -> dict:
        trades = self.trader.state.trade_history[-EVOLVE_EVERY_N_TRADES:]
        if len(trades) < MIN_TRADES_FOR_EVAL:
            return {"ready": False}

        pnls = [t.pnl_pct for t in trades]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0.0
        sharpe = avg_pnl / max(std_pnl, 1e-8)

        return {
            "ready": True,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "num_trades": len(trades),
        }

    def evolve(self):
        """Mutate V6 strategy parameters based on recent performance."""
        perf = self._evaluate_recent_performance()
        if not perf["ready"]:
            return

        params = self.trader.state.strategy_params
        old_params = dict(params)

        mutation_rate = 0.05 if perf["sharpe"] > 0 else 0.10

        int_params = ["ma5_len", "ma8_len", "ema21_len", "ma55_len", "bb_length", "atr_period", "slope_len"]
        float_params = ["bb_std_dev", "dist_ma5_ma8", "dist_ma8_ema21", "dist_ema21_ma55", "slope_threshold"]

        all_params = int_params + float_params
        n_mutations = random.randint(1, 2)
        to_mutate = random.sample(all_params, min(n_mutations, len(all_params)))

        bounds = {
            "ma5_len": (3, 8), "ma8_len": (6, 12), "ema21_len": (15, 30),
            "ma55_len": (40, 80), "bb_length": (15, 30), "bb_std_dev": (1.5, 4.0),
            "dist_ma5_ma8": (0.5, 3.0), "dist_ma8_ema21": (1.0, 5.0),
            "dist_ema21_ma55": (2.0, 8.0), "slope_len": (2, 5),
            "slope_threshold": (0.02, 0.5), "atr_period": (7, 21),
        }

        for key in to_mutate:
            val = params[key]
            delta = val * mutation_rate * random.choice([-1, 1])
            new_val = val + delta

            lo, hi = bounds.get(key, (0, 100))
            new_val = max(lo, min(hi, new_val))

            if key in int_params:
                new_val = int(round(new_val))

            params[key] = new_val

        # Enforce MA ordering: ma5 < ma8 < ema21 < ma55
        if params["ma5_len"] >= params["ma8_len"]:
            params["ma8_len"] = params["ma5_len"] + 2
        if params["ma8_len"] >= params["ema21_len"]:
            params["ema21_len"] = params["ma8_len"] + 5
        if params["ema21_len"] >= params["ma55_len"]:
            params["ma55_len"] = params["ema21_len"] + 15

        self.trader.state.generation += 1

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
        if not self.trader.state.is_alive:
            return

        self.lessons.cycle += 1
        self._cycle_phase = "observe"

        self.trader.check_daily_reset()

        # ── PHASE: OBSERVE — update positions ──
        await self.trader.update_positions()

        # ── PHASE: MANAGE — check exits ──
        self._cycle_phase = "manage"
        await self.manage_positions()

        # ── PHASE: LEARN — extract lessons from recent trades ──
        self._cycle_phase = "learn"
        self._learn_from_recent()

        # Check consecutive losses — pause after 3 consecutive losses
        recent = self.trader.state.trade_history[-3:]
        if len(recent) >= 3 and all(t.pnl_usd < 0 for t in recent):
            if not self.lessons.has_recent_warning("risk", lookback=3):
                print(f"[Agent] 3 consecutive losses — pausing new entries until a win")
                self.lessons.add("risk", "3 consecutive losses — auto-paused", symbol="ALL")
        else:
            # ── PHASE: SCAN — generate signals for all symbols ──
            self._cycle_phase = "scan"
            for symbol in WATCH_SYMBOLS:
                if symbol in self.trader.state.positions:
                    continue

                signal = await self.generate_signal(symbol)
                if signal is None or signal.get("action") not in ("long", "short"):
                    if signal:
                        self._last_signals[symbol] = signal
                    continue

                # Stamp signal with timestamp for dedup tracking
                signal["_ts"] = time.time()
                signal["market_regime"] = self.lessons.market_regime

                # ── 举一反三: Check if lessons warn against this regime/side ──
                if self.lessons.should_skip_regime(signal["action"]):
                    reason = f"Lessons warn: {signal['action']} risky in {self.lessons.market_regime} regime"
                    print(f"[Agent] {symbol} 举一反三 BLOCKED: {reason}")
                    _audit_log({
                        "event": "signal_blocked_by_lesson",
                        "symbol": symbol,
                        "signal": signal.get("action"),
                        "reason": reason,
                        "regime": self.lessons.market_regime,
                    })
                    self._last_signals[symbol] = {**signal, "blocked": True, "block_reasons": [reason]}
                    continue

                # ── Pre-trade checklist (institutional-grade) ──
                self._cycle_phase = "validate"
                passed, failures = PreTradeChecklist.validate(self, symbol, signal)

                if not passed:
                    print(f"[Agent] {symbol} pre-trade BLOCKED: {'; '.join(failures)}")
                    _audit_log({
                        "event": "signal_blocked",
                        "symbol": symbol,
                        "signal": signal.get("action"),
                        "confidence": signal.get("confidence"),
                        "reason": signal.get("reason"),
                        "failures": failures,
                        "mode": self.trader.state.mode,
                    })
                    self._last_signals[symbol] = {**signal, "blocked": True, "block_reasons": failures}
                    continue

                # ── PHASE: ACT — Execute trade ──
                self._cycle_phase = "execute"
                size = signal["confidence"] * self.trader.risk.max_position_pct * self.trader.state.equity
                equity_before = self.trader.state.equity
                result = await self.trader.open_position(symbol, signal["action"], size)
                if result["ok"]:
                    print(f"[Agent] Opened {signal['action'].upper()} {symbol} @ {result['price']} size=${size:.0f} conf={signal['confidence']}")

                    # Telegram notification for new position
                    direction = "📈 做多" if signal["action"] == "long" else "📉 做空"
                    await self._notify_telegram(
                        f"{direction} <b>{symbol}</b>\n"
                        f"价格: ${result['price']}\n"
                        f"仓位: ${size:.0f} | 信心: {signal['confidence']:.0%}\n"
                        f"止损: ${signal.get('sl', 0):.2f} | 止盈: ${signal.get('tp', 0):.2f}\n"
                        f"策略: {signal.get('reason', '')[:80]}",
                        category="fill",
                    )

                    _audit_log({
                        "event": "open_position",
                        "symbol": symbol,
                        "side": signal["action"],
                        "price": result["price"],
                        "size_usd": size,
                        "confidence": signal["confidence"],
                        "reason": signal["reason"],
                        "sl": signal.get("sl"),
                        "tp": signal.get("tp"),
                        "volatility_regime": signal.get("volatility_regime"),
                        "market_regime": self.lessons.market_regime,
                        "mode": self.trader.state.mode,
                        "equity_before": equity_before,
                        "pre_trade_checks": "all_passed",
                    })
                    self._last_signals[symbol] = signal
                    self._signal_history.append({
                        "symbol": symbol,
                        "signal": signal,
                        "time": time.time(),
                        "executed": True,
                    })
                    self._save_state()
                else:
                    print(f"[Agent] {symbol} execution failed: {result.get('reason')}")
                    _audit_log({
                        "event": "execution_failed",
                        "symbol": symbol,
                        "reason": result.get("reason"),
                        "signal": signal.get("action"),
                    })

        # ── PHASE: EVOLVE — mutate params if due ──
        self._cycle_phase = "evolve"
        if (self.trader.state.total_trades > 0 and
                self.trader.state.total_trades % EVOLVE_EVERY_N_TRADES == 0 and
                self.trader.state.total_trades != self._last_evolved_at):
            self.evolve()
            self._last_evolved_at = self.trader.state.total_trades

        # ── PHASE: CHECKPOINT ──
        self._cycle_phase = "checkpoint"
        self.lessons.save()
        self._cycle_phase = "idle"

    def _learn_from_recent(self):
        """Extract generalizable lessons from recent trade history."""
        trades = self.trader.state.trade_history
        if len(trades) < 5:
            return

        recent = trades[-10:]
        wins = sum(1 for t in recent if t.pnl_pct > 0)
        losses = len(recent) - wins
        winrate = wins / len(recent) * 100

        # Detect win/loss streaks
        streak = 0
        streak_dir = None
        for t in reversed(recent):
            if streak_dir is None:
                streak_dir = "win" if t.pnl_pct > 0 else "loss"
                streak = 1
            elif (t.pnl_pct > 0) == (streak_dir == "win"):
                streak += 1
            else:
                break

        if streak >= 4 and streak_dir == "win":
            self.lessons.add("pattern", f"Win streak {streak} — current params working well, avoid mutation",
                           symbol="ALL", data={"streak": streak, "winrate": winrate})
        elif streak >= 3 and streak_dir == "loss":
            self.lessons.add("risk", f"Loss streak {streak} — consider reducing position size or pausing",
                           symbol="ALL", data={"streak": streak, "winrate": winrate})

        # Check if recent average PnL is drifting negative
        avg_pnl = np.mean([t.pnl_pct for t in recent])
        if avg_pnl < -0.5 and len(recent) >= 8:
            self.lessons.add("param", f"Avg PnL drifting negative ({avg_pnl:.2f}%) — params may need recalibration",
                           symbol="ALL")

    async def _send_daily_summary(self):
        """Send daily equity summary via Telegram."""
        s = self.trader.state
        positions_str = ""
        for sym, pos in s.positions.items():
            positions_str += f"\n  • {sym} {pos.side} PnL: ${pos.unrealized_pnl:+.2f}"
        if not positions_str:
            positions_str = "\n  无持仓"

        win_rate = (s.win_count / max(s.total_trades, 1)) * 100
        await self._notify_telegram(
            f"📊 <b>每日报告</b>\n"
            f"模式: {'模拟盘' if s.mode == 'paper' else '实盘'}\n"
            f"💰 权益: ${s.equity:.2f}\n"
            f"📈 今日盈亏: ${s.daily_pnl:+.2f}\n"
            f"总盈亏: ${s.total_pnl_usd:+.2f}\n"
            f"交易次数: {s.total_trades} | 胜率: {win_rate:.0f}%\n"
            f"持仓:{positions_str}\n"
            f"市场状态: {self.lessons.market_regime}",
            category="daily",
        )

    async def run_loop(self):
        self._running = True
        print(f"[Agent] Started V6 strategy. Mode={self.trader.state.mode} Equity=${self.trader.state.equity:.2f} Gen={self.trader.state.generation}")

        # Send startup notification
        await self._notify_telegram(
            f"🚀 <b>交易机器人已启动</b>\n"
            f"模式: {'模拟盘' if self.trader.state.mode == 'paper' else '实盘'}\n"
            f"初始资金: ${self.trader.state.equity:.2f}\n"
            f"监控币种: {', '.join(WATCH_SYMBOLS)}\n"
            f"策略: V6 趋势跟踪 (Gen {self.trader.state.generation})\n"
            f"扫描间隔: {TICK_INTERVAL_SEC}秒",
            category="fill",
        )

        daily_summary_hour = -1  # track last daily summary hour

        while self._running:
            try:
                await self.tick()

                # Send daily summary once per day (at hour 8 UTC)
                current_hour = datetime.now(timezone.utc).hour
                if current_hour == 8 and daily_summary_hour != 8:
                    daily_summary_hour = 8
                    await self._send_daily_summary()
                elif current_hour != 8:
                    daily_summary_hour = current_hour
            except Exception as e:
                print(f"[Agent] Tick error: {e}")
                await self._notify_telegram(f"⚠️ Agent error: {e}", category="error")
            await asyncio.sleep(TICK_INTERVAL_SEC)

    def start(self):
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self.run_loop())

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._save_state()
        print("[Agent] Stopped.")

    def get_status(self) -> dict:
        return {
            **self.trader.state.to_dict(),
            "running": self._running,
            "cycle_phase": self._cycle_phase,
            "watch_symbols": WATCH_SYMBOLS,
            "signal_interval": SIGNAL_INTERVAL,
            "tick_interval_sec": TICK_INTERVAL_SEC,
            "last_signals": {
                k: {
                    "action": v.get("action"),
                    "confidence": v.get("confidence"),
                    "reason": v.get("reason"),
                    "blocked": v.get("blocked", False),
                    "block_reasons": v.get("block_reasons", []),
                    "market_regime": v.get("market_regime", ""),
                }
                for k, v in self._last_signals.items()
            },
            "risk_limits": {
                "max_position_pct": self.trader.risk.max_position_pct,
                "max_total_exposure_pct": self.trader.risk.max_total_exposure_pct,
                "max_daily_loss_pct": self.trader.risk.max_daily_loss_pct,
                "max_drawdown_pct": self.trader.risk.max_drawdown_pct,
                "max_positions": self.trader.risk.max_positions,
                "cooldown_seconds": self.trader.risk.cooldown_seconds,
            },
            "harness": self.lessons.get_summary(),
        }
