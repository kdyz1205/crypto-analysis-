"""Evaluator: runs a variant across a symbol/TF matrix.

Per-setup tracing (bug-fixed):
- Harness tags every Trade with (line_id, setup_touch_number).
- Evaluator groups trades by (line_id, setup_touch_number) and produces one
  SetupTrace per setup (1 fade + optional 1 flip).
- Aggregation reports overall metrics AND per-touch-number breakdowns so
  the reflection agent can see "at which touch number does edge appear".

Fitness formula (FIXED across evolution):
  fitness = win_rate × avg_total_R × sqrt(n_setups_triggered)
  invalid (hard penalty) if n_setups_triggered < min_total_triggered
  penalty if total_R < max_allowed_dd_R
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from importlib import import_module
from pathlib import Path
from typing import Callable
from collections import defaultdict
import math

import pandas as pd

from server.strategy.evolved.base import EvolvedLine, sanity_check
from .data_cache import load_candles, split_train_test
from .harness import HarnessParams, run_backtest, Trade
from .trace import SetupTrace, write_traces_jsonl, summarize_traces


DetectFn = Callable[[pd.DataFrame, str, str], list[EvolvedLine]]


@dataclass(slots=True)
class EvalConfig:
    symbols: list[str]
    timeframes: list[str]
    days: int = 730
    train_fraction: float = 0.7
    harness: HarnessParams = field(default_factory=HarnessParams)
    min_total_triggered: int = 20
    max_allowed_dd_R: float = -30.0


@dataclass(slots=True)
class EvalResult:
    variant: str
    train: dict
    test: dict
    fitness_train: float
    fitness_test: float
    per_slice: list[dict]
    traces: list[SetupTrace] = field(default_factory=list)

    def to_summary(self) -> dict:
        return {
            "variant": self.variant,
            "fitness_train": round(self.fitness_train, 4),
            "fitness_test": round(self.fitness_test, 4),
            "train_total_R": round(self.train.get("total_R", 0.0), 2),
            "test_total_R": round(self.test.get("total_R", 0.0), 2),
            "train_win_rate": round(self.train.get("fade_win_rate", 0.0), 3),
            "test_win_rate": round(self.test.get("fade_win_rate", 0.0), 3),
            "train_triggered": self.train.get("n_setups_triggered", 0),
            "test_triggered": self.test.get("n_setups_triggered", 0),
        }


# ────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ────────────────────────────────────────────────────────────────
def _slope_pct_per_bar(line: EvolvedLine) -> float:
    span = line.end_index - line.start_index
    if span <= 0 or line.start_price <= 0:
        return 0.0
    return ((line.end_price - line.start_price) / line.start_price) / span


def _vol_regime(atr_pct: float) -> str:
    if atr_pct < 0.005:
        return "low"
    if atr_pct > 0.02:
        return "high"
    return "normal"


# ────────────────────────────────────────────────────────────────
# Trade → SetupTrace alignment (BUG-FIXED: uses line_id grouping)
# ────────────────────────────────────────────────────────────────
def _build_traces(
    variant: str,
    symbol: str,
    timeframe: str,
    split: str,
    candles: pd.DataFrame,
    lines: list[EvolvedLine],
    trades: list[Trade],
    atr_series,
    lookback_bars: int,
) -> list[SetupTrace]:
    """One SetupTrace per (line_id, setup_touch_number) group in trades.

    The harness discovers touches ORGANICALLY from price action (no look-ahead),
    so the number of setups per line is not known in advance. We simply
    collect every (line, setup_touch_n) group found in trades and emit a trace.

    We also emit a "not_triggered" trace for every line that produced ZERO
    trades, so variants can be fairly penalized for emitting many lines that
    never actually get traded.
    """
    grouped: dict[tuple[int, int], dict[str, Trade]] = defaultdict(dict)
    for tr in trades:
        if tr.line_id < 0:
            continue
        key = (tr.line_id, tr.setup_touch_number)
        grouped[key][tr.leg] = tr

    traces: list[SetupTrace] = []
    n_total = len(candles)
    close_series = candles["close"].astype(float).values

    # Lines that produced at least one trade — emit one trace per setup.
    lines_with_trades: set[int] = {k[0] for k in grouped.keys()}

    for (line_idx, setup_touch_n), bucket in sorted(grouped.items()):
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]
        fade = bucket.get("fade")
        flip = bucket.get("flip")

        anchor_bar = fade.entry_index if fade else line.end_index
        bi = min(max(anchor_bar, 0), n_total - 1)
        atr_at = float(atr_series[bi])
        price_at = float(close_series[bi])
        atr_pct = atr_at / price_at if price_at > 0 else 0.0
        span = line.end_index - line.start_index
        span_pct = span / max(lookback_bars, 1)

        trace = SetupTrace(
            variant=variant,
            symbol=symbol,
            timeframe=timeframe,
            split=split,
            line_id=line_idx,
            setup_touch_number=setup_touch_n,
            side=line.side,
            span_bars=span,
            span_pct_of_available=round(span_pct, 4),
            total_touch_count=line.touch_count,
            slope_pct_per_bar=round(_slope_pct_per_bar(line), 6),
            anchor_prominence=0.0,
            anchor_bar=anchor_bar,
            total_bars=n_total,
            atr_at_anchor=round(atr_at, 6),
            price_at_anchor=round(price_at, 6),
            vol_regime=_vol_regime(atr_pct),
            fade_triggered=(fade is not None),
            fade_result=(fade.reason if fade else "not_triggered"),
            fade_R=round(fade.R, 4) if fade else 0.0,
        )
        if fade is not None:
            trace.fade_bars_to_trigger = max(0, fade.entry_index - line.end_index)
            trace.fade_bars_held = fade.exit_index - fade.entry_index
        if flip is not None:
            trace.flip_triggered = True
            trace.flip_result = flip.reason
            trace.flip_R = round(flip.R, 4)
            trace.flip_bars_held = flip.exit_index - flip.entry_index
        trace.total_R = round(trace.fade_R + trace.flip_R, 4)
        traces.append(trace)

    # Lines that produced NO trades — emit a "not_triggered" placeholder so
    # variants that spam many unused lines are fairly counted.
    for line_idx, line in enumerate(lines):
        if line_idx in lines_with_trades:
            continue
        bi = min(max(line.end_index, 0), n_total - 1)
        atr_at = float(atr_series[bi])
        price_at = float(close_series[bi])
        atr_pct = atr_at / price_at if price_at > 0 else 0.0
        span = line.end_index - line.start_index

        traces.append(SetupTrace(
            variant=variant, symbol=symbol, timeframe=timeframe, split=split,
            line_id=line_idx, setup_touch_number=0,  # 0 = never traded
            side=line.side,
            span_bars=span,
            span_pct_of_available=round(span / max(lookback_bars, 1), 4),
            total_touch_count=line.touch_count,
            slope_pct_per_bar=round(_slope_pct_per_bar(line), 6),
            anchor_prominence=0.0,
            anchor_bar=line.end_index,
            total_bars=n_total,
            atr_at_anchor=round(atr_at, 6),
            price_at_anchor=round(price_at, 6),
            vol_regime=_vol_regime(atr_pct),
            fade_triggered=False,
            fade_result="not_triggered",
            fade_R=0.0,
        ))

    return traces


# ────────────────────────────────────────────────────────────────
# Per-slice runner
# ────────────────────────────────────────────────────────────────
def _eval_slice(
    detect_fn: DetectFn,
    variant_name: str,
    symbol: str,
    tf: str,
    candles: pd.DataFrame,
    split_name: str,
    harness: HarnessParams,
) -> tuple[list[SetupTrace], dict]:
    empty = {
        "n_setups": 0, "n_setups_triggered": 0, "trigger_rate": 0.0,
        "fade_win_rate": 0.0, "total_R": 0.0, "avg_total_R": 0.0,
        "n_flip": 0, "flip_win_rate": 0.0, "by_setup_touch": {},
    }
    if candles is None or len(candles) < 60:
        return [], empty

    try:
        lines = detect_fn(candles, tf, symbol)
    except Exception as e:
        return [], {**empty, "error": f"detect_fn raised: {e}"}

    ok, reason = sanity_check(lines, len(candles))
    if not ok:
        return [], {**empty, "error": f"sanity: {reason}"}

    result = run_backtest(candles, lines, harness)

    from .harness import _atr_series
    atr_vals = _atr_series(candles).values
    lookback_bars = min(len(candles), 500)

    traces = _build_traces(
        variant_name, symbol, tf, split_name, candles, list(lines),
        result.trades, atr_vals, lookback_bars,
    )
    return traces, summarize_traces(traces)


# ────────────────────────────────────────────────────────────────
# Variant evaluation
# ────────────────────────────────────────────────────────────────
def evaluate_variant(
    variant_module_path: str,
    config: EvalConfig,
    output_dir: Path | None = None,
) -> EvalResult:
    variant_name = variant_module_path.rsplit(".", 1)[-1]
    mod = import_module(variant_module_path)
    detect_fn: DetectFn = getattr(mod, "detect_lines")

    per_slice: list[dict] = []
    all_train_traces: list[SetupTrace] = []
    all_test_traces: list[SetupTrace] = []

    for symbol in config.symbols:
        for tf in config.timeframes:
            candles = load_candles(symbol, tf, config.days)
            if candles is None or len(candles) < 200:
                per_slice.append({"symbol": symbol, "tf": tf, "error": "no data"})
                continue

            train, test = split_train_test(candles, config.train_fraction)

            train_traces, train_summary = _eval_slice(
                detect_fn, variant_name, symbol, tf, train, "train", config.harness,
            )
            test_traces, test_summary = _eval_slice(
                detect_fn, variant_name, symbol, tf, test, "test", config.harness,
            )

            all_train_traces.extend(train_traces)
            all_test_traces.extend(test_traces)

            per_slice.append({
                "symbol": symbol, "tf": tf,
                "train": train_summary, "test": test_summary,
            })

    # Global aggregates (all traces, per-touch-number slice included)
    train_metrics = summarize_traces(all_train_traces)
    test_metrics = summarize_traces(all_test_traces)

    fitness_train = _fitness(train_metrics, config)
    fitness_test = _fitness(test_metrics, config)

    all_traces = all_train_traces + all_test_traces
    result = EvalResult(
        variant=variant_name,
        train=train_metrics,
        test=test_metrics,
        fitness_train=fitness_train,
        fitness_test=fitness_test,
        per_slice=per_slice,
        traces=all_traces,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_traces_jsonl(output_dir / f"{variant_name}_traces.jsonl", all_traces)
        import json
        with (output_dir / f"{variant_name}_summary.json").open("w", encoding="utf-8") as f:
            json.dump({
                "variant": variant_name,
                "train": train_metrics,
                "test": test_metrics,
                "fitness_train": fitness_train,
                "fitness_test": fitness_test,
                "per_slice": per_slice,
            }, f, indent=2, default=str)

    return result


# ────────────────────────────────────────────────────────────────
# Fitness
# ────────────────────────────────────────────────────────────────
def _fitness(m: dict, config: EvalConfig) -> float:
    n = m.get("n_setups_triggered", 0)
    wr = m.get("fade_win_rate", 0.0)
    avg_R = m.get("avg_total_R", 0.0)
    if n < config.min_total_triggered:
        # Small gradient so larger-but-still-insufficient samples rank higher,
        # but capped strictly below zero so no "underpowered" variant can
        # ever beat a legit one.
        return min(-100.0 + n / 2.0, -1.0)
    raw = wr * avg_R * math.sqrt(n)
    total_R = m.get("total_R", 0.0)
    if total_R < config.max_allowed_dd_R:
        raw -= (config.max_allowed_dd_R - total_R) * 0.1
    return raw
