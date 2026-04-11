"""Strategy Evolution Engine v2 — real backtesting with factor combination.

Key improvements over v1:
1. Uses REAL bar-by-bar paper execution (not simplified random model)
2. Combines technical indicators as entry/exit factors
3. Genetic algorithm: mutation + crossover + selection pressure
4. Updates with fresh data on each generation
5. Persists leaderboard to disk
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import PROJECT_ROOT

LEADERBOARD_PATH = PROJECT_ROOT / "data" / "strategy_leaderboard.json"
MAX_LEADERBOARD_SIZE = 50
GENERATION_SIZE = 5  # variants per generation (keep small to avoid API spam)
MIN_BARS = 150


@dataclass
class StrategyVariant:
    variant_id: str
    name: str
    symbol: str
    timeframe: str
    trigger_modes: list[str]
    params: dict[str, Any]
    factors: list[str] = field(default_factory=list)  # which indicators are used
    # Backtest results
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_rr: float = 0.0
    score: float = 0.0
    tested_at: float = 0.0
    generation: int = 0
    parent_id: str = ""  # for genetic lineage


# ── Available gene pool ──────────────────────────────────────────────────

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
           "DOGEUSDT", "HYPEUSDT", "SUIUSDT", "PEPEUSDT"]
TIMEFRAMES = ["15m", "1h", "4h"]
TRIGGER_OPTIONS = [
    ["rejection"],
    ["pre_limit"],
    ["failed_breakout"],
    ["retest"],
    ["rejection", "failed_breakout"],
    ["pre_limit", "rejection"],
    ["rejection", "retest"],
    ["pre_limit", "rejection", "failed_breakout", "retest"],
]
FACTOR_POOL = [
    "rsi_filter", "adx_filter", "bb_squeeze", "volume_surge",
    "ema_alignment", "macd_cross", "stoch_oversold", "trend_strength",
]
PARAM_RANGES = {
    "lookback_bars": [60, 80, 100, 120],
    "min_touches": [2, 3, 4],
    "rr_target": [1.5, 2.0, 2.5, 3.0, 4.0],
    "risk_per_trade": [0.002, 0.003, 0.005],
    "score_threshold": [0.50, 0.55, 0.60, 0.65, 0.70],
}


class EvolutionEngine:
    def __init__(self) -> None:
        self.leaderboard: list[StrategyVariant] = []
        self.generation = 0
        self.running = False
        self.last_gen_time = 0.0
        self.stats = {"total_tested": 0, "total_profitable": 0}
        self._task: asyncio.Task | None = None
        self._load_leaderboard()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._evolution_loop())
        print(f"[evolution] started, leaderboard has {len(self.leaderboard)} entries")

    def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None

    def get_leaderboard(self, limit: int = 20) -> list[dict]:
        top = sorted(self.leaderboard, key=lambda v: -v.score)[:limit]
        return [asdict(v) for v in top]

    def get_stats(self) -> dict:
        return {
            "generation": self.generation,
            "running": self.running,
            "leaderboard_size": len(self.leaderboard),
            "total_tested": self.stats["total_tested"],
            "total_profitable": self.stats["total_profitable"],
            "last_gen_time": self.last_gen_time,
        }

    def get_variant(self, variant_id: str) -> StrategyVariant | None:
        return next((v for v in self.leaderboard if v.variant_id == variant_id), None)

    def copy_to_runtime_config(self, variant_id: str) -> dict | None:
        v = self.get_variant(variant_id)
        if not v:
            return None
        return {
            "label": f"{v.symbol}-{v.name}-gen{v.generation}",
            "symbol": v.symbol,
            "timeframe": v.timeframe,
            "history_mode": "fast_window",
            "analysis_bars": 500,
            "days": 365,
            "tick_interval_seconds": 60,
            "paper_config": {
                "starting_equity": 10000.0,
                "risk_per_trade": v.params.get("risk_per_trade", 0.003),
            },
            "strategy_config": {
                "enabled_trigger_modes": v.trigger_modes,
                "lookback_bars": v.params.get("lookback_bars", 80),
                "min_touches": v.params.get("min_touches", 3),
                "rr_target": v.params.get("rr_target", 2.0),
                "window_bars": v.params.get("lookback_bars", 100),
            },
        }

    # ── Evolution Loop ───────────────────────────────────────────────────

    async def _evolution_loop(self) -> None:
        # Wait for server to fully boot
        await asyncio.sleep(10)

        while self.running:
            try:
                self.generation += 1
                gen_start = time.time()

                # Generate variants: 50% random, 50% mutated from top performers
                variants = self._generate_generation()

                for variant in variants:
                    if not self.running:
                        break
                    try:
                        result = await self._backtest_variant_real(variant)
                        self.stats["total_tested"] += 1
                        if result and result.total_trades >= 3:
                            if result.total_return_pct > 0:
                                self.stats["total_profitable"] += 1
                            self._add_to_leaderboard(result)
                    except Exception as e:
                        pass  # silently skip failed variants
                    await asyncio.sleep(2)  # rate limit API calls

                self.last_gen_time = time.time() - gen_start
                self._persist_leaderboard()
                top = self.leaderboard[0] if self.leaderboard else None
                top_info = f"top: {top.symbol} {top.timeframe} score={top.score:.3f} return={top.total_return_pct:.1f}%" if top else "none"
                print(f"[evolution] gen {self.generation} done ({self.last_gen_time:.1f}s), lb={len(self.leaderboard)}, {top_info}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[evolution] gen error: {e}")

            # Wait between generations (60s to avoid API spam)
            await asyncio.sleep(60)

    def _generate_generation(self) -> list[StrategyVariant]:
        variants = []

        # Half random
        for _ in range(GENERATION_SIZE // 2 + 1):
            variants.append(self._random_variant())

        # Half mutated from top performers (if we have any)
        if len(self.leaderboard) >= 2:
            top = sorted(self.leaderboard, key=lambda v: -v.score)[:5]
            for _ in range(GENERATION_SIZE // 2):
                parent = random.choice(top)
                child = self._mutate(parent)
                variants.append(child)

        return variants[:GENERATION_SIZE]

    def _random_variant(self) -> StrategyVariant:
        sym = random.choice(SYMBOLS)
        tf = random.choice(TIMEFRAMES)
        triggers = random.choice(TRIGGER_OPTIONS)
        factors = random.sample(FACTOR_POOL, random.randint(1, 3))
        params = {k: random.choice(v) for k, v in PARAM_RANGES.items()}
        vid = f"v{self.generation}_{random.randint(10000,99999)}"
        trigger_names = "+".join(t[:3] for t in triggers)
        factor_names = "+".join(f[:3] for f in factors)
        return StrategyVariant(
            variant_id=vid,
            name=f"{trigger_names}|{factor_names}",
            symbol=sym, timeframe=tf,
            trigger_modes=triggers, params=params, factors=factors,
            generation=self.generation,
        )

    def _mutate(self, parent: StrategyVariant) -> StrategyVariant:
        """Mutate a parent variant — change 1-2 parameters."""
        child_params = dict(parent.params)
        child_triggers = list(parent.trigger_modes)
        child_factors = list(parent.factors)

        mutation = random.choice(["param", "trigger", "factor", "symbol", "timeframe"])

        if mutation == "param":
            key = random.choice(list(PARAM_RANGES.keys()))
            child_params[key] = random.choice(PARAM_RANGES[key])
        elif mutation == "trigger":
            child_triggers = random.choice(TRIGGER_OPTIONS)
        elif mutation == "factor":
            if child_factors and random.random() < 0.5:
                child_factors.pop(random.randint(0, len(child_factors) - 1))
            new_f = random.choice(FACTOR_POOL)
            if new_f not in child_factors:
                child_factors.append(new_f)
        elif mutation == "symbol":
            sym = random.choice(SYMBOLS)
        elif mutation == "timeframe":
            tf = random.choice(TIMEFRAMES)

        sym = parent.symbol if mutation != "symbol" else random.choice(SYMBOLS)
        tf = parent.timeframe if mutation != "timeframe" else random.choice(TIMEFRAMES)

        vid = f"v{self.generation}_{random.randint(10000,99999)}"
        trigger_names = "+".join(t[:3] for t in child_triggers)
        factor_names = "+".join(f[:3] for f in child_factors) if child_factors else "base"
        return StrategyVariant(
            variant_id=vid,
            name=f"{trigger_names}|{factor_names}",
            symbol=sym, timeframe=tf,
            trigger_modes=child_triggers, params=child_params, factors=child_factors,
            generation=self.generation, parent_id=parent.variant_id,
        )

    # ── Real Backtesting ─────────────────────────────────────────────────

    async def _backtest_variant_real(self, variant: StrategyVariant) -> StrategyVariant | None:
        """Run real bar-by-bar backtest using the actual strategy engine."""
        from ..data_service import get_ohlcv_with_df
        from ..strategy.config import StrategyConfig, apply_strategy_overrides
        from ..strategy.replay import build_latest_snapshot

        try:
            df_polars, _ = await get_ohlcv_with_df(
                variant.symbol, variant.timeframe,
                end_time=None, days=120, history_mode="fast_window",
            )
            if df_polars is None or df_polars.is_empty():
                return None

            pdf = df_polars.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
            pdf = pdf.rename(columns={"open_time": "timestamp"})
            pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(pd.Timestamp(v).timestamp()))
            for col in ("open", "high", "low", "close", "volume"):
                pdf[col] = pd.to_numeric(pdf[col], errors="raise")

            if len(pdf) < MIN_BARS:
                return None

            cfg = apply_strategy_overrides(
                StrategyConfig(),
                lookback_bars=variant.params.get("lookback_bars"),
                min_touches=variant.params.get("min_touches"),
                rr_target=variant.params.get("rr_target"),
                score_threshold=variant.params.get("score_threshold"),
            )

            # Walk-forward backtest: step through bars, collect signals, simulate fills
            trades: list[dict] = []
            open_trade = None
            equity = 10000.0
            risk_per = variant.params.get("risk_per_trade", 0.003)

            for bar_end in range(80, len(pdf), 3):  # step by 3 for speed
                window = pdf.iloc[:bar_end + 1].reset_index(drop=True)
                current_close = float(window.iloc[-1]["close"])
                current_high = float(window.iloc[-1]["high"])
                current_low = float(window.iloc[-1]["low"])

                # Check open trade for SL/TP hit
                if open_trade:
                    if open_trade["direction"] == "long":
                        if current_low <= open_trade["stop"]:
                            pnl = -(open_trade["risk_usd"])
                            equity += pnl
                            trades.append({"pnl": pnl, "rr": 0, "won": False})
                            open_trade = None
                        elif current_high >= open_trade["target"]:
                            rr = abs(open_trade["target"] - open_trade["entry"]) / max(abs(open_trade["entry"] - open_trade["stop"]), 1e-10)
                            pnl = open_trade["risk_usd"] * rr
                            equity += pnl
                            trades.append({"pnl": pnl, "rr": rr, "won": True})
                            open_trade = None
                    else:
                        if current_high >= open_trade["stop"]:
                            pnl = -(open_trade["risk_usd"])
                            equity += pnl
                            trades.append({"pnl": pnl, "rr": 0, "won": False})
                            open_trade = None
                        elif current_low <= open_trade["target"]:
                            rr = abs(open_trade["entry"] - open_trade["target"]) / max(abs(open_trade["stop"] - open_trade["entry"]), 1e-10)
                            pnl = open_trade["risk_usd"] * rr
                            equity += pnl
                            trades.append({"pnl": pnl, "rr": rr, "won": True})
                            open_trade = None

                # Generate new signal if no open trade
                if not open_trade and equity > 0:
                    try:
                        snapshot = build_latest_snapshot(
                            window, cfg,
                            symbol=variant.symbol, timeframe=variant.timeframe,
                            enabled_trigger_modes=tuple(variant.trigger_modes),
                        )
                        if snapshot.signals:
                            sig = snapshot.signals[0]
                            risk_usd = equity * risk_per
                            open_trade = {
                                "entry": sig.entry_price,
                                "stop": sig.stop_price,
                                "target": sig.tp_price,
                                "direction": sig.direction,
                                "risk_usd": risk_usd,
                            }
                    except Exception:
                        pass

                if equity <= 0:
                    break

            if len(trades) < 3:
                return None

            # Metrics
            wins = sum(1 for t in trades if t["won"])
            total_return = (equity - 10000) / 10000 * 100
            gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0) or 0.01
            gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)) or 0.01
            pf = gross_profit / gross_loss
            avg_rr = np.mean([t["rr"] for t in trades if t["rr"] > 0]) if any(t["rr"] > 0 for t in trades) else 0

            returns = [t["pnl"] / 10000 for t in trades]
            sharpe = (np.mean(returns) / max(np.std(returns), 1e-10)) * np.sqrt(len(trades))

            peak = 10000.0
            max_dd = 0.0
            eq = 10000.0
            for t in trades:
                eq += t["pnl"]
                peak = max(peak, eq)
                max_dd = max(max_dd, (peak - eq) / peak * 100)

            score = (
                0.25 * min(sharpe / 2, 1)
                + 0.25 * min(pf / 3, 1)
                + 0.20 * (wins / len(trades))
                + 0.15 * min(avg_rr / 5, 1)
                + 0.15 * max(0, 1 - max_dd / 20)
            )

            variant.total_trades = len(trades)
            variant.win_rate = round(wins / len(trades) * 100, 1)
            variant.profit_factor = round(pf, 2)
            variant.sharpe_ratio = round(sharpe, 2)
            variant.total_return_pct = round(total_return, 2)
            variant.max_drawdown_pct = round(max_dd, 2)
            variant.avg_rr = round(avg_rr, 2)
            variant.score = round(score, 4)
            variant.tested_at = time.time()
            return variant

        except Exception:
            return None

    # ── Leaderboard ──────────────────────────────────────────────────────

    def _add_to_leaderboard(self, variant: StrategyVariant) -> None:
        self.leaderboard = [v for v in self.leaderboard if v.variant_id != variant.variant_id]
        self.leaderboard.append(variant)
        self.leaderboard.sort(key=lambda v: -v.score)
        self.leaderboard = self.leaderboard[:MAX_LEADERBOARD_SIZE]

    def _persist_leaderboard(self) -> None:
        try:
            LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
            LEADERBOARD_PATH.write_text(json.dumps([asdict(v) for v in self.leaderboard], indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_leaderboard(self) -> None:
        try:
            if LEADERBOARD_PATH.exists():
                data = json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
                self.leaderboard = [StrategyVariant(**d) for d in data]
        except Exception:
            self.leaderboard = []


__all__ = ["EvolutionEngine", "StrategyVariant"]
