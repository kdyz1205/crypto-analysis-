"""Factor Engine — pure-function factor discovery, evaluation, iteration, composition.

No side effects. No global state. Every function takes input → returns output.
Designed to be called by: evolution engine, TG bot, AI chat, or scheduled tasks.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class Factor:
    """A single trading factor with parameters."""
    factor_id: str
    name: str
    category: str  # "momentum" | "trend" | "volatility" | "volume" | "structure" | "composite"
    indicator: str  # e.g. "rsi", "macd", "bb_width", "zone_score"
    params: dict[str, Any] = field(default_factory=dict)  # e.g. {"period": 14, "threshold": 30}
    condition: str = "gt"  # "gt" | "lt" | "cross_above" | "cross_below" | "between"
    threshold: float = 0.0
    weight: float = 1.0
    description: str = ""


@dataclass
class FactorResult:
    """Result of evaluating a factor or factor combination."""
    factor_ids: list[str]
    symbol: str
    timeframe: str
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_rr: float = 0.0
    score: float = 0.0
    tested_at: float = 0.0
    params_hash: str = ""


# ── Factor Pool (known factors) ──────────────────────────────────────────

FACTOR_POOL: list[Factor] = [
    # Momentum
    Factor("rsi_oversold", "RSI 超卖", "momentum", "rsi", {"period": 14}, "lt", 30, description="RSI < 30 超卖做多"),
    Factor("rsi_overbought", "RSI 超买", "momentum", "rsi", {"period": 14}, "gt", 70, description="RSI > 70 超买做空"),
    Factor("rsi6_oversold", "RSI6 超卖", "momentum", "rsi", {"period": 6}, "lt", 25, description="短周期RSI超卖"),
    Factor("stoch_oversold", "Stoch 超卖", "momentum", "stoch_k", {"period": 14}, "lt", 20),
    Factor("stoch_overbought", "Stoch 超买", "momentum", "stoch_k", {"period": 14}, "gt", 80),
    Factor("macd_bull", "MACD 金叉", "momentum", "macd_hist", {}, "gt", 0, description="MACD柱 > 0"),
    Factor("macd_bear", "MACD 死叉", "momentum", "macd_hist", {}, "lt", 0),
    Factor("roc_positive", "ROC 正", "momentum", "roc", {"period": 12}, "gt", 0),
    Factor("williams_oversold", "Williams超卖", "momentum", "williams_r", {"period": 14}, "lt", -80),

    # Trend
    Factor("adx_strong", "ADX 强趋势", "trend", "adx", {"period": 14}, "gt", 25, description="ADX > 25 有趋势"),
    Factor("adx_weak", "ADX 弱趋势", "trend", "adx", {"period": 14}, "lt", 20, description="ADX < 20 震荡"),
    Factor("ema_bull", "EMA 多头排列", "trend", "ema_alignment", {}, "gt", 0.5),
    Factor("cci_oversold", "CCI 超卖", "trend", "cci", {"period": 20}, "lt", -100),
    Factor("cci_overbought", "CCI 超买", "trend", "cci", {"period": 20}, "gt", 100),

    # Volatility
    Factor("bb_squeeze", "BB 压缩", "volatility", "bb_width", {"period": 20}, "lt", 0.02, description="布林带收窄"),
    Factor("bb_expand", "BB 扩张", "volatility", "bb_width", {"period": 20}, "gt", 0.05),
    Factor("atr_high", "ATR 高波动", "volatility", "atr", {"period": 14}, "gt", 0, description="ATR高于均值"),
    Factor("kc_squeeze", "KC 压缩", "volatility", "squeeze", {}, "gt", 0.5, description="BB在KC内"),

    # Volume
    Factor("volume_surge", "成交量激增", "volume", "volume_ratio", {}, "gt", 1.5, description="量>1.5倍均值"),
    Factor("mfi_oversold", "MFI 超卖", "volume", "mfi", {"period": 14}, "lt", 20),
    Factor("mfi_overbought", "MFI 超买", "volume", "mfi", {"period": 14}, "gt", 80),
    Factor("obv_rising", "OBV 上升", "volume", "obv_slope", {}, "gt", 0),

    # Structure (from SR engine)
    Factor("near_support", "接近支撑", "structure", "dist_to_support_atr", {}, "lt", 0.5),
    Factor("near_resistance", "接近阻力", "structure", "dist_to_resistance_atr", {}, "lt", 0.5),
    Factor("zone_strong", "Zone 高分", "structure", "zone_score", {}, "gt", 0.6),
    Factor("trend_up", "上升趋势", "structure", "trend_strength_up", {}, "gt", 0.5),
    Factor("trend_down", "下降趋势", "structure", "trend_strength_down", {}, "gt", 0.5),
]

FACTOR_BY_ID = {f.factor_id: f for f in FACTOR_POOL}


# ── Factor Engine (pure functions) ───────────────────────────────────────

class FactorEngine:
    """Stateless factor engine. All methods are pure functions."""

    @staticmethod
    def list_factors() -> list[dict]:
        """List all available factors."""
        return [asdict(f) for f in FACTOR_POOL]

    @staticmethod
    def get_factor(factor_id: str) -> Factor | None:
        return FACTOR_BY_ID.get(factor_id)

    @staticmethod
    def generate_candidates(n: int = 10) -> list[list[str]]:
        """Generate N random factor combinations to test."""
        candidates = []
        for _ in range(n):
            # Pick 2-4 random factors from different categories
            categories = random.sample(list({f.category for f in FACTOR_POOL}), min(3, len({f.category for f in FACTOR_POOL})))
            combo = []
            for cat in categories:
                pool = [f for f in FACTOR_POOL if f.category == cat]
                combo.append(random.choice(pool).factor_id)
            candidates.append(combo)
        return candidates

    @staticmethod
    def evaluate_factor(
        df: pd.DataFrame,
        factor_ids: list[str],
        *,
        symbol: str = "",
        timeframe: str = "",
        risk_per_trade: float = 0.01,
        rr_target: float = 2.0,
    ) -> FactorResult:
        """Evaluate a factor combination on historical data. Pure function."""
        from ..strategy.indicators import compute_indicator

        factors = [FACTOR_BY_ID[fid] for fid in factor_ids if fid in FACTOR_BY_ID]
        if not factors:
            return FactorResult(factor_ids=factor_ids, symbol=symbol, timeframe=timeframe)

        n = len(df)
        if n < 50:
            return FactorResult(factor_ids=factor_ids, symbol=symbol, timeframe=timeframe)

        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values

        # Compute ATR for position sizing
        atr = pd.Series(np.zeros(n))
        for i in range(1, n):
            atr.iloc[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = atr.rolling(14).mean().fillna(atr.iloc[:14].mean())

        # Compute each factor's indicator
        signals = np.ones(n, dtype=bool)  # start with all True, AND each factor
        for factor in factors:
            try:
                if factor.indicator == "volume_ratio":
                    vol = df["volume"].astype(float)
                    avg_vol = vol.rolling(20).mean()
                    values = (vol / avg_vol.clip(lower=1)).values
                elif factor.indicator in ("dist_to_support_atr", "dist_to_resistance_atr", "zone_score", "trend_strength_up", "trend_strength_down", "ema_alignment", "squeeze", "obv_slope"):
                    # Structure factors need SR engine — skip for now, always True
                    continue
                else:
                    values = compute_indicator(df, factor.indicator, **factor.params).values

                # Apply condition
                if factor.condition == "gt":
                    mask = values > factor.threshold
                elif factor.condition == "lt":
                    mask = values < factor.threshold
                else:
                    mask = np.ones(n, dtype=bool)

                signals &= mask
            except Exception:
                continue

        # Simulate trades at signal bars
        trades = []
        equity = 10000.0
        in_trade = False
        entry = stop = tp = 0.0

        for i in range(20, n - 5):
            if in_trade:
                if low[i] <= stop:
                    pnl = -(equity * risk_per_trade)
                    equity += pnl
                    trades.append({"pnl": pnl, "won": False, "rr": 0})
                    in_trade = False
                elif high[i] >= tp:
                    pnl = equity * risk_per_trade * rr_target
                    equity += pnl
                    trades.append({"pnl": pnl, "won": True, "rr": rr_target})
                    in_trade = False
            elif signals[i] and not in_trade:
                entry = close[i]
                atr_val = float(atr.iloc[i])
                if atr_val <= 0:
                    continue
                stop = entry - atr_val * 1.5
                tp = entry + atr_val * 1.5 * rr_target
                in_trade = True

            if equity <= 0:
                break

        if len(trades) < 3:
            return FactorResult(factor_ids=factor_ids, symbol=symbol, timeframe=timeframe, total_trades=len(trades))

        wins = sum(1 for t in trades if t["won"])
        total_return = (equity - 10000) / 10000 * 100
        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0) or 0.01
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)) or 0.01
        pf = gross_profit / gross_loss
        returns = [t["pnl"] / 10000 for t in trades]
        sharpe = (np.mean(returns) / max(np.std(returns), 1e-10)) * np.sqrt(len(trades))
        peak, max_dd, eq = 10000.0, 0.0, 10000.0
        for t in trades:
            eq += t["pnl"]; peak = max(peak, eq); max_dd = max(max_dd, (peak - eq) / peak * 100)

        score = 0.25 * min(sharpe / 2, 1) + 0.25 * min(pf / 3, 1) + 0.20 * (wins / len(trades)) + 0.15 * min(rr_target / 5, 1) + 0.15 * max(0, 1 - max_dd / 20)

        return FactorResult(
            factor_ids=factor_ids, symbol=symbol, timeframe=timeframe,
            total_trades=len(trades), win_rate=round(wins / len(trades) * 100, 1),
            profit_factor=round(pf, 2), sharpe_ratio=round(sharpe, 2),
            total_return_pct=round(total_return, 2), max_drawdown_pct=round(max_dd, 2),
            avg_rr=round(rr_target, 2), score=round(score, 4), tested_at=time.time(),
            params_hash=hashlib.md5(json.dumps(factor_ids, sort_keys=True).encode()).hexdigest()[:8],
        )

    @staticmethod
    def compose_strategy(
        factor_ids: list[str],
        name: str,
        symbol: str = "BTCUSDT",
        timeframe: str = "4h",
    ) -> dict:
        """Compose factors into a named strategy definition."""
        factors = [asdict(FACTOR_BY_ID[fid]) for fid in factor_ids if fid in FACTOR_BY_ID]
        return {
            "name": name,
            "factors": factors,
            "factor_ids": factor_ids,
            "symbol": symbol,
            "timeframe": timeframe,
            "created_at": time.time(),
        }
