"""Event replay labels for user-drawn trendlines.

The learner captures why the user drew a line. This module labels what
happened after that draw time using lower-timeframe candles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .drawing_learner import TF_SECONDS, _line_price_at_ts, _safe_float, _timestamp_seconds


@dataclass(frozen=True)
class ReplayConfig:
    buffer_pct: float = 0.0005
    rr: float = 8.0
    sl_tick_pct: float = 0.00001
    max_tf_bars: int = 20
    fee_bps: float = 0.0
    trailing_enabled: bool = True
    only_tighten_stop: bool = True


def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a pandas OHLCV frame with timestamp seconds."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    if "timestamp" not in out.columns and "open_time" in out.columns:
        out = out.rename(columns={"open_time": "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("ohlcv df must include timestamp or open_time")
    out["timestamp"] = out["timestamp"].map(_timestamp_seconds)
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    else:
        out["volume"] = 0.0
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def simulate_line_outcome(
    drawing: dict[str, Any],
    ohlcv_1m: pd.DataFrame,
    config: ReplayConfig,
) -> dict[str, Any]:
    """Replay a bounce trade from draw time onward.

    support -> long: entry at line * (1 + buffer), SL just below line.
    resistance -> short: entry at line * (1 - buffer), SL just above line.

    The unfilled entry order and the filled stop are updated at the start of
    each bar of the drawing's own timeframe, not every 1m candle.
    """
    df = normalize_ohlcv_df(ohlcv_1m)
    base = _base_result(drawing, config)
    if df.empty:
        return {**base, "status": "no_data", "exit_reason": "no_data"}

    symbol = str(drawing.get("symbol") or "").upper()
    timeframe = str(drawing.get("timeframe") or "")
    side = str(drawing.get("side") or drawing.get("kind") or "").lower()
    kind = side if side in ("support", "resistance") else "support"
    direction = "long" if kind == "support" else "short"
    tf_seconds = TF_SECONDS.get(timeframe, 3600)
    price_start = _safe_float(drawing.get("price_start"))
    price_end = _safe_float(drawing.get("price_end"))
    t_start = int(_safe_float(drawing.get("t_start")))
    t_end = int(_safe_float(drawing.get("t_end")))
    if price_start <= 0 or price_end <= 0 or t_end <= t_start:
        return {**base, "status": "invalid_line", "exit_reason": "invalid_line"}

    bars_between = (_timestamp_seconds(t_end) - _timestamp_seconds(t_start)) / max(float(tf_seconds), 1.0)
    if bars_between <= 0:
        return {**base, "status": "invalid_line", "exit_reason": "invalid_span"}
    slope_per_bar = (price_end - price_start) / bars_between

    replay_start_ts = _replay_start_ts(drawing)
    rows = df[df["timestamp"] > replay_start_ts].reset_index(drop=True)
    if rows.empty:
        return {**base, "status": "waiting_for_future_data", "exit_reason": "no_bars_after_draw"}

    deadline_ts = replay_start_ts + max(int(config.max_tf_bars), 1) * tf_seconds
    active_bucket: int | None = None
    active_entry = 0.0
    active_stop = 0.0
    entry_ts: int | None = None
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    initial_risk = 0.0
    mfe_r = 0.0
    mae_r = 0.0
    stop_updates = 0

    for row in rows.itertuples(index=False):
        ts = int(row.timestamp)
        bucket = (ts // tf_seconds) * tf_seconds
        if active_bucket != bucket:
            active_bucket = bucket
            line_price = _line_price_at_ts(
                price_start=price_start,
                slope_per_bar=slope_per_bar,
                t_start=t_start,
                target_ts=bucket,
                tf_seconds=tf_seconds,
            )
            next_entry, next_stop = _entry_and_stop(kind, line_price, config)
            if entry_ts is None:
                active_entry = next_entry
                active_stop = next_stop
            elif config.trailing_enabled:
                old_stop = stop_price
                if config.only_tighten_stop:
                    stop_price = max(stop_price, next_stop) if direction == "long" else min(stop_price, next_stop)
                else:
                    stop_price = next_stop
                if abs(stop_price - old_stop) > 1e-12:
                    stop_updates += 1

        high = float(row.high)
        low = float(row.low)
        close = float(row.close)

        if entry_ts is None:
            if ts > deadline_ts:
                return {**base, "status": "no_fill", "exit_reason": "expired_unfilled", "deadline_ts": deadline_ts}
            filled = low <= active_entry if direction == "long" else high >= active_entry
            if not filled:
                continue
            entry_ts = ts
            entry_price = active_entry
            stop_price = active_stop
            initial_risk = abs(entry_price - stop_price)
            if initial_risk <= 1e-12:
                return {**base, "status": "invalid_risk", "exit_reason": "zero_risk"}
            tp_price = entry_price + config.rr * initial_risk if direction == "long" else entry_price - config.rr * initial_risk

        mfe_r, mae_r = _update_excursion(direction, entry_price, initial_risk, high, low, mfe_r, mae_r)
        exit_price, reason = _exit_for_bar(direction, high, low, stop_price, tp_price)
        if reason:
            return _final_result(
                base=base,
                status="closed",
                exit_reason=reason,
                direction=direction,
                entry_ts=entry_ts,
                entry_price=entry_price,
                exit_ts=ts,
                exit_price=exit_price,
                initial_risk=initial_risk,
                mfe_r=mfe_r,
                mae_r=mae_r,
                stop_updates=stop_updates,
                fee_bps=config.fee_bps,
            )

    if entry_ts is None:
        return {**base, "status": "no_fill", "exit_reason": "end_of_data_unfilled"}
    last = rows.iloc[-1]
    return _final_result(
        base=base,
        status="open_at_end",
        exit_reason="mark_to_last_close",
        direction=direction,
        entry_ts=entry_ts,
        entry_price=entry_price,
        exit_ts=int(last["timestamp"]),
        exit_price=float(last["close"]),
        initial_risk=initial_risk,
        mfe_r=mfe_r,
        mae_r=mae_r,
        stop_updates=stop_updates,
        fee_bps=config.fee_bps,
    )


def _base_result(drawing: dict[str, Any], config: ReplayConfig) -> dict[str, Any]:
    return {
        "event": "user_drawing_outcome",
        "manual_line_id": drawing.get("manual_line_id"),
        "symbol": str(drawing.get("symbol") or "").upper(),
        "timeframe": str(drawing.get("timeframe") or ""),
        "side": str(drawing.get("side") or drawing.get("kind") or ""),
        "capture_ts": _replay_start_ts(drawing),
        "config": asdict(config),
    }


def _replay_start_ts(drawing: dict[str, Any]) -> int:
    for key in ("ts", "capture_ts", "updated_at", "created_at"):
        value = drawing.get(key)
        if value is not None:
            return _timestamp_seconds(value)
    return int(pd.Timestamp.now(tz="UTC").timestamp())


def _entry_and_stop(kind: str, line_price: float, config: ReplayConfig) -> tuple[float, float]:
    if kind == "support":
        return line_price * (1.0 + config.buffer_pct), line_price * (1.0 - config.sl_tick_pct)
    return line_price * (1.0 - config.buffer_pct), line_price * (1.0 + config.sl_tick_pct)


def _update_excursion(
    direction: str,
    entry: float,
    risk: float,
    high: float,
    low: float,
    mfe_r: float,
    mae_r: float,
) -> tuple[float, float]:
    if direction == "long":
        mfe_r = max(mfe_r, (high - entry) / risk)
        mae_r = min(mae_r, (low - entry) / risk)
    else:
        mfe_r = max(mfe_r, (entry - low) / risk)
        mae_r = min(mae_r, (entry - high) / risk)
    return float(mfe_r), float(mae_r)


def _exit_for_bar(direction: str, high: float, low: float, stop: float, tp: float) -> tuple[float, str | None]:
    if direction == "long":
        stop_hit = low <= stop
        tp_hit = high >= tp
        if stop_hit:
            return stop, "stop"
        if tp_hit:
            return tp, "tp"
    else:
        stop_hit = high >= stop
        tp_hit = low <= tp
        if stop_hit:
            return stop, "stop"
        if tp_hit:
            return tp, "tp"
    return 0.0, None


def _final_result(
    *,
    base: dict[str, Any],
    status: str,
    exit_reason: str,
    direction: str,
    entry_ts: int,
    entry_price: float,
    exit_ts: int,
    exit_price: float,
    initial_risk: float,
    mfe_r: float,
    mae_r: float,
    stop_updates: int,
    fee_bps: float,
) -> dict[str, Any]:
    gross = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
    gross_r = gross / initial_risk if initial_risk > 0 else 0.0
    gross_pct = gross / entry_price if entry_price > 0 else 0.0
    fee_pct = max(float(fee_bps), 0.0) / 10000.0 * 2.0
    risk_pct = initial_risk / entry_price if entry_price > 0 else 0.0
    net_pct = gross_pct - fee_pct
    net_r = net_pct / risk_pct if risk_pct > 0 else gross_r
    return {
        **base,
        "status": status,
        "direction": direction,
        "filled": True,
        "entry_ts": int(entry_ts),
        "entry_price": round(float(entry_price), 12),
        "exit_ts": int(exit_ts),
        "exit_price": round(float(exit_price), 12),
        "exit_reason": exit_reason,
        "initial_risk": round(float(initial_risk), 12),
        "realized_r_gross": round(float(gross_r), 6),
        "realized_r": round(float(net_r), 6),
        "realized_pct_gross": round(float(gross_pct), 8),
        "realized_pct": round(float(net_pct), 8),
        "mfe_r": round(float(mfe_r), 6),
        "mae_r": round(float(mae_r), 6),
        "tf_bars_held": round((int(exit_ts) - int(entry_ts)) / max(float(TF_SECONDS.get(base.get("timeframe"), 3600)), 1.0), 3),
        "minutes_held": round((int(exit_ts) - int(entry_ts)) / 60.0, 3),
        "walking_stop_updates": int(stop_updates),
        "label_trade_win": 1 if net_r > 0 else 0,
    }


__all__ = ["ReplayConfig", "normalize_ohlcv_df", "simulate_line_outcome"]
