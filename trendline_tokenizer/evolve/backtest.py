"""Lightweight bar-by-bar backtest: given (line, trade_config), simulate
the user's entry/SL/TP rule and return realized_r.

Matches the semantics used in data/user_drawing_outcomes.jsonl so the
outcomes produced here are comparable to the user's stored outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..schemas.trendline import TrendlineRecord


@dataclass
class BacktestOutcome:
    line_id: str
    config: dict[str, Any]
    filled: bool = False
    direction: str | None = None
    entry_bar: int | None = None
    entry_price: float | None = None
    exit_bar: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    realized_r: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0

    def as_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "config": self.config,
            "filled": self.filled,
            "direction": self.direction,
            "entry_bar": self.entry_bar,
            "entry_price": self.entry_price,
            "exit_bar": self.exit_bar,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "realized_r": self.realized_r,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "bars_held": self.bars_held,
        }


def _line_price_at(line: TrendlineRecord, bar_index: int) -> float:
    """Linear interpolation in price space is used for backtest-scale
    evaluation; log interp is for live placement only (matches legacy
    outcomes.jsonl which also used linear)."""
    dur = max(1, line.end_bar_index - line.start_bar_index)
    if bar_index <= line.start_bar_index:
        return line.start_price
    if bar_index >= line.end_bar_index:
        # extend the line forward along the same slope
        slope = (line.end_price - line.start_price) / dur
        return line.end_price + slope * (bar_index - line.end_bar_index)
    frac = (bar_index - line.start_bar_index) / dur
    return line.start_price + frac * (line.end_price - line.start_price)


def _derive_direction(line: TrendlineRecord) -> str:
    """For a support line we default to LONG; resistance → SHORT.
    Channels / wedges: use line.direction as a fallback heuristic."""
    if line.line_role == "support":
        return "long"
    if line.line_role == "resistance":
        return "short"
    return "long" if line.direction == "up" else "short"


def backtest_one(
    line: TrendlineRecord,
    ohlcv: pd.DataFrame,
    config: dict[str, Any],
    *,
    max_bars_forward: int = 80,
) -> BacktestOutcome:
    """Simulate ONE (line, config) pair. Walks bars forward from
    line.end_bar_index + 1, looks for a touch within buffer_pct, sets
    SL = line.opposite_side - sl_tick_pct, TP = rr * risk."""
    buffer_pct = float(config.get("buffer_pct") or 0.0)
    rr = float(config.get("rr") or 2.0)
    sl_tick_pct = float(config.get("sl_tick_pct") or 1e-5)
    direction = _derive_direction(line)
    out = BacktestOutcome(line_id=line.id, config=config, direction=direction)

    start = line.end_bar_index + 1
    end = min(len(ohlcv), start + max_bars_forward)
    if start >= end:
        return out

    entry_price = None
    sl_price = None
    tp_price = None
    entry_bar = None

    # Walk forward, look for line touch (high/low enters buffer band)
    for i in range(start, end):
        row = ohlcv.iloc[i]
        line_px = _line_price_at(line, i)
        hi = float(row["high"])
        lo = float(row["low"])

        if entry_price is None:
            # Entry: for long on support → touched line from above AND
            # bounced back (low < line < close). Simplified: price touched
            # line_px ± buffer band.
            band_lo = line_px * (1 - buffer_pct)
            band_hi = line_px * (1 + buffer_pct)
            if direction == "long" and lo <= band_hi and lo >= band_lo * (1 - 0.01):
                entry_price = line_px * (1 + buffer_pct)  # place buy slightly above line
                sl_price = line_px * (1 - buffer_pct - sl_tick_pct)
                risk = max(1e-9, entry_price - sl_price)
                tp_price = entry_price + rr * risk
                entry_bar = i
                out.filled = True
                out.entry_bar = i
                out.entry_price = entry_price
                continue
            if direction == "short" and hi >= band_lo and hi <= band_hi * (1 + 0.01):
                entry_price = line_px * (1 - buffer_pct)
                sl_price = line_px * (1 + buffer_pct + sl_tick_pct)
                risk = max(1e-9, sl_price - entry_price)
                tp_price = entry_price - rr * risk
                entry_bar = i
                out.filled = True
                out.entry_bar = i
                out.entry_price = entry_price
                continue
        else:
            # track MFE / MAE in R units
            risk = max(1e-9,
                       (entry_price - sl_price) if direction == "long"
                       else (sl_price - entry_price))
            if direction == "long":
                mfe = (hi - entry_price) / risk
                mae = (lo - entry_price) / risk
            else:
                mfe = (entry_price - lo) / risk
                mae = (entry_price - hi) / risk
            out.mfe_r = max(out.mfe_r, mfe)
            out.mae_r = min(out.mae_r, mae)

            # check SL / TP hit
            if direction == "long":
                if lo <= sl_price:
                    out.exit_reason = "stop"
                    out.exit_price = sl_price
                    out.exit_bar = i
                    out.realized_r = -1.0
                    out.bars_held = i - entry_bar
                    return out
                if hi >= tp_price:
                    out.exit_reason = "target"
                    out.exit_price = tp_price
                    out.exit_bar = i
                    out.realized_r = rr
                    out.bars_held = i - entry_bar
                    return out
            else:
                if hi >= sl_price:
                    out.exit_reason = "stop"
                    out.exit_price = sl_price
                    out.exit_bar = i
                    out.realized_r = -1.0
                    out.bars_held = i - entry_bar
                    return out
                if lo <= tp_price:
                    out.exit_reason = "target"
                    out.exit_price = tp_price
                    out.exit_bar = i
                    out.realized_r = rr
                    out.bars_held = i - entry_bar
                    return out

    # timeout
    if entry_price is not None:
        # mark-to-market at last bar
        last = ohlcv.iloc[end - 1]
        last_close = float(last["close"])
        risk = max(1e-9,
                   (entry_price - sl_price) if direction == "long"
                   else (sl_price - entry_price))
        if direction == "long":
            out.realized_r = (last_close - entry_price) / risk
        else:
            out.realized_r = (entry_price - last_close) / risk
        out.exit_reason = "timeout"
        out.exit_price = last_close
        out.exit_bar = end - 1
        out.bars_held = end - 1 - entry_bar
    return out


def backtest_lines(
    lines: list[TrendlineRecord],
    ohlcv_by_sym_tf: dict[tuple[str, str], pd.DataFrame],
    configs: list[dict[str, Any]],
    *,
    max_bars_forward: int = 80,
) -> list[BacktestOutcome]:
    """Cartesian product backtest: every line × every config."""
    outs: list[BacktestOutcome] = []
    for line in lines:
        df = ohlcv_by_sym_tf.get((line.symbol, line.timeframe))
        if df is None:
            continue
        for cfg in configs:
            outs.append(backtest_one(line, df, cfg, max_bars_forward=max_bars_forward))
    return outs
