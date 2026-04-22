"""Drawing Learner — captures market features when user draws a trendline.

When a user draws a line on the chart:
1. Compute slope, intercept from the two anchor points
2. Snapshot market features at the current bar (ATR, BB, ribbon, volume, etc.)
3. Save to data/user_drawings_ml.jsonl for PyTorch training

Later, compare user-drawn lines vs algorithm-found lines to learn the user's
selection criteria (what makes a "good" line in the user's eyes).
"""
from __future__ import annotations

import json
import time
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ML_DRAWINGS_FILE = Path("data/user_drawings_ml.jsonl")
OUTCOMES_FILE = Path("data/user_drawing_outcomes.jsonl")
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _timestamp_seconds(value) -> int:
    if isinstance(value, (int, float, np.integer, np.floating)):
        raw = float(value)
        return int(raw / 1000) if raw > 1e12 else int(raw)
    return int(pd.Timestamp(value).timestamp())


def _timestamp_series_seconds(df: pd.DataFrame) -> np.ndarray:
    if "timestamp" not in df.columns:
        return np.arange(len(df), dtype=float)
    return np.asarray([_timestamp_seconds(value) for value in df["timestamp"].tolist()], dtype=float)


def _atr_values(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) == 0:
        return np.asarray([], dtype=float)
    if len(close) == 1:
        return np.asarray([max(float(high[0] - low[0]), 0.0)], dtype=float)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    return pd.Series(np.concatenate([[np.nan], tr])).rolling(period).mean().bfill().fillna(0.0).values


def _rsi_at(close: np.ndarray, bar_idx: int, period: int = 14) -> float:
    if bar_idx < period or len(close) <= period:
        return 50.0
    window = close[max(0, bar_idx - period): bar_idx + 1]
    delta = np.diff(window)
    if len(delta) == 0:
        return 50.0
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_loss = float(np.mean(losses))
    if avg_loss <= 1e-12:
        return 100.0
    rs = float(np.mean(gains)) / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _line_price_at_ts(
    *,
    price_start: float,
    slope_per_bar: float,
    t_start: int,
    target_ts: int | float,
    tf_seconds: int,
) -> float:
    bars_from_start = (_timestamp_seconds(target_ts) - _timestamp_seconds(t_start)) / max(float(tf_seconds), 1.0)
    return float(price_start) + float(slope_per_bar) * bars_from_start


def _compute_features(df: pd.DataFrame, bar_idx: int) -> dict:
    """Compute market features at a specific bar index."""
    if bar_idx < 0 or bar_idx >= len(df):
        return {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # ATR
    atr_arr = _atr_values(high, low, close, 14)
    atr = float(atr_arr[bar_idx]) if np.isfinite(atr_arr[bar_idx]) else 0

    c = float(close[bar_idx])
    atr_pct = atr / c if c > 0 else 0

    # Bollinger Bands
    bb_ma = pd.Series(close).rolling(21).mean().values
    bb_sd = pd.Series(close).rolling(21).std().values
    bb_upper = bb_ma[bar_idx] + 2.1 * bb_sd[bar_idx] if np.isfinite(bb_sd[bar_idx]) else c
    bb_lower = bb_ma[bar_idx] - 2.1 * bb_sd[bar_idx] if np.isfinite(bb_sd[bar_idx]) else c
    bb_width = (bb_upper - bb_lower) / bb_ma[bar_idx] if bb_ma[bar_idx] > 0 else 0
    bb_pct = (c - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # EMA Ribbon
    periods = [5, 8, 21, 55]
    emas = {}
    for p in periods:
        emas[p] = pd.Series(close).ewm(span=p, adjust=False).mean().values[bar_idx]

    ribbon_bull = all(emas[periods[i]] > emas[periods[i + 1]] for i in range(len(periods) - 1))
    ribbon_bear = all(emas[periods[i]] < emas[periods[i + 1]] for i in range(len(periods) - 1))
    ribbon_score = 1.0 if ribbon_bull else (-1.0 if ribbon_bear else 0.0)
    ribbon_spread = abs(emas[periods[0]] - emas[periods[-1]]) / c if c > 0 else 0

    # Volume
    vol = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    vol_ma = pd.Series(vol).rolling(20).mean().values
    vol_ratio = float(vol[bar_idx] / vol_ma[bar_idx]) if vol_ma[bar_idx] > 0 else 1.0

    # Returns
    ret_1 = (close[bar_idx] / close[bar_idx - 1] - 1) if bar_idx > 0 else 0
    ret_4 = (close[bar_idx] / close[max(0, bar_idx - 4)] - 1) if bar_idx > 4 else 0
    ret_12 = (close[bar_idx] / close[max(0, bar_idx - 12)] - 1) if bar_idx > 12 else 0
    rsi = _rsi_at(close, bar_idx)
    if ribbon_score > 0 and ret_12 >= 0:
        trend_context = "uptrend"
    elif ribbon_score < 0 and ret_12 <= 0:
        trend_context = "downtrend"
    else:
        trend_context = "range"

    return {
        "close": round(c, 8),
        "atr": round(atr, 8),
        "atr_pct": round(atr_pct, 6),
        "bb_width": round(bb_width, 6),
        "bb_pct": round(bb_pct, 4),
        "ribbon_score": ribbon_score,
        "ribbon_spread": round(ribbon_spread, 6),
        "vol_ratio": round(vol_ratio, 4),
        "rsi": round(rsi, 4),
        "ret_1": round(ret_1, 6),
        "ret_4": round(ret_4, 6),
        "ret_12": round(ret_12, 6),
        "trend_context": trend_context,
    }


def _compute_line_interaction_features(
    *,
    df: pd.DataFrame,
    timeframe: str,
    side: str,
    price_start: float,
    price_end: float,
    t_start: int,
    t_end: int,
    slope_per_bar: float,
) -> dict:
    """Describe how price has interacted with the user's projected line.

    These are user-drawing features, not trade labels. They only use the
    candles loaded at capture time and summarize why this line may matter:
    touches, rejections, violations, line age, and distance to the line.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return {}

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(df.get("open", df["close"]), errors="coerce").to_numpy(dtype=float)
    timestamps = _timestamp_series_seconds(df)
    finite = np.isfinite(close) & np.isfinite(high) & np.isfinite(low) & np.isfinite(open_) & np.isfinite(timestamps)
    if not finite.any():
        return {}

    close = close[finite]
    high = high[finite]
    low = low[finite]
    open_ = open_[finite]
    timestamps = timestamps[finite]
    tf_seconds = TF_SECONDS.get(timeframe, 3600)
    line_prices = np.asarray([
        _line_price_at_ts(
            price_start=price_start,
            slope_per_bar=slope_per_bar,
            t_start=t_start,
            target_ts=ts,
            tf_seconds=tf_seconds,
        )
        for ts in timestamps
    ], dtype=float)

    atr = _atr_values(high, low, close, 14)
    fallback_tol = np.maximum(np.abs(close) * 0.001, 1e-12)
    tolerance = np.maximum(atr * 0.20, fallback_tol)
    near_tolerance = np.maximum(atr * 0.50, fallback_tol * 2.0)

    # Ignore candles before the first anchor unless the user explicitly drew
    # a left-extended line. The captured line meaning starts at anchor one.
    active_mask = timestamps >= _timestamp_seconds(t_start)
    touches = active_mask & (high >= line_prices - tolerance) & (low <= line_prices + tolerance)
    near_misses = active_mask & ~touches & (np.abs(close - line_prices) <= near_tolerance)

    kind = side if side in ("support", "resistance") else ("support" if price_end >= price_start else "resistance")
    if kind == "support":
        body_violation = active_mask & (np.minimum(open_, close) < line_prices - tolerance)
        rejected = touches & (close >= line_prices) & (low <= line_prices + tolerance)
        rejection_atr = np.maximum(close - line_prices, 0.0) / np.maximum(atr, 1e-12)
        wrong_side_close = close < line_prices
    else:
        body_violation = active_mask & (np.maximum(open_, close) > line_prices + tolerance)
        rejected = touches & (close <= line_prices) & (high >= line_prices - tolerance)
        rejection_atr = np.maximum(line_prices - close, 0.0) / np.maximum(atr, 1e-12)
        wrong_side_close = close > line_prices

    touch_indices = np.flatnonzero(touches)
    rejected_values = rejection_atr[rejected] if rejected.any() else np.asarray([], dtype=float)
    recent_window = min(len(close), 50)
    recent_touches = touches[-recent_window:] if recent_window else np.asarray([], dtype=bool)
    last_touch_age = int(len(close) - 1 - touch_indices[-1]) if len(touch_indices) else -1
    current_line_price = float(line_prices[-1])
    current_close = float(close[-1])
    current_atr = max(float(atr[-1]), 1e-12)
    line_span_bars = (_timestamp_seconds(t_end) - _timestamp_seconds(t_start)) / max(float(tf_seconds), 1.0)
    line_age_bars = (float(timestamps[-1]) - _timestamp_seconds(t_start)) / max(float(tf_seconds), 1.0)

    # This is same-TF confluence for now. True higher-TF confluence is added
    # by the label script when it loads multiple timeframes.
    htf_confluence = 0.0
    if kind == "support" and current_close >= current_line_price and not bool(wrong_side_close[-1]):
        htf_confluence = 0.35
    elif kind == "resistance" and current_close <= current_line_price and not bool(wrong_side_close[-1]):
        htf_confluence = 0.35

    return {
        "touch_count": int(touches.sum()),
        "recent_touch_count": int(recent_touches.sum()),
        "near_miss_count": int(near_misses.sum()),
        "body_violation_count": int(body_violation.sum()),
        "wick_rejection_count": int(rejected.sum()),
        "wick_rejection_ratio": round(float(rejected.sum() / max(int(touches.sum()), 1)), 6),
        "avg_rejection_atr": round(float(np.mean(rejected_values)) if len(rejected_values) else 0.0, 6),
        "max_rejection_atr": round(float(np.max(rejected_values)) if len(rejected_values) else 0.0, 6),
        "last_touch_age_bars": last_touch_age,
        "line_age_bars": round(float(line_age_bars), 3),
        "line_span_bars": round(float(line_span_bars), 3),
        "distance_to_line_atr": round(float((current_close - current_line_price) / current_atr), 6),
        "wrong_side_close": int(bool(wrong_side_close[-1])),
        "htf_confluence_score": round(htf_confluence, 4),
    }


def _compute_htf_confluence_features(htf_df: pd.DataFrame | None, kind: str) -> dict:
    if htf_df is None or htf_df.empty:
        return {}
    htf_market = _compute_features(htf_df, len(htf_df) - 1)
    ribbon = _safe_float(htf_market.get("ribbon_score"))
    context = str(htf_market.get("trend_context") or "range")
    if kind == "support":
        score = 1.0 if ribbon > 0 and context == "uptrend" else (0.5 if ribbon >= 0 else 0.0)
    else:
        score = 1.0 if ribbon < 0 and context == "downtrend" else (0.5 if ribbon <= 0 else 0.0)
    return {
        "htf_confluence_score": round(score, 4),
        "htf_ribbon_score": round(ribbon, 4),
        "htf_trend_context": context,
    }


def capture_user_drawing(
    symbol: str,
    timeframe: str,
    side: str,
    price_start: float,
    price_end: float,
    t_start: int,
    t_end: int,
    df: pd.DataFrame | None = None,
    htf_df: pd.DataFrame | None = None,
    manual_line_id: str | None = None,
    capture_stage: str = "created",
    reason: str | None = None,
) -> dict | None:
    """Capture a user drawing with market features for ML training.

    Called when a user creates a manual trendline on the chart.

    capture_stage: 'created' | 'updated' | 'deleted' | 'triggered' | 'closed'
    (+ '_basic' suffix when OHLCV features couldn't be loaded)
    reason: optional free-form tag, e.g. 'user_delete' / 'user_update'.
    """
    if t_end <= t_start or price_start <= 0 or price_end <= 0:
        return None

    # Compute slope and intercept (in bar-index space)
    # t_start/t_end could be seconds or milliseconds
    is_ms = t_start > 1e12  # timestamps > 1 trillion = milliseconds
    dt_seconds = (t_end - t_start) / 1000 if is_ms else float(t_end - t_start)
    bar_dur = TF_SECONDS.get(timeframe, 3600)
    bars_between = dt_seconds / bar_dur

    if bars_between < 1:
        return None

    slope = (price_end - price_start) / bars_between
    # Determine kind from side or slope
    kind = side if side in ("support", "resistance") else (
        "support" if slope >= 0 else "resistance"
    )

    anchor_distance_pct = (price_end - price_start) / price_start if price_start > 0 else 0

    record = {
        "event": "user_drawing",
        "manual_line_id": manual_line_id,
        "capture_stage": capture_stage,
        "ts": time.time(),
        "dt": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "kind": kind,
        "side": side,
        "price_start": price_start,
        "price_end": price_end,
        "t_start": t_start,
        "t_end": t_end,
        "slope": slope,
        "bars_between": round(bars_between, 1),
        "anchor_distance_pct": round(anchor_distance_pct, 6),
        "slope_per_bar": round(slope, 10),
    }
    if reason:
        record["reason"] = reason

    # Add market features if we have OHLCV data
    if df is not None and len(df) > 0:
        # Use the last bar as current market state
        features = _compute_features(df, len(df) - 1)
        interaction_features = _compute_line_interaction_features(
            df=df,
            timeframe=timeframe,
            side=side,
            price_start=price_start,
            price_end=price_end,
            t_start=t_start,
            t_end=t_end,
            slope_per_bar=slope,
        )
        features.update(interaction_features)
        htf_features = _compute_htf_confluence_features(htf_df, kind)
        features.update(htf_features)
        record["features"] = features

        # Compute projected line price at current bar
        if "timestamp" in df.columns:
            ts_value = df["timestamp"].iloc[-1]
            projected_price = _line_price_at_ts(
                price_start=price_start,
                slope_per_bar=slope,
                t_start=t_start,
                target_ts=ts_value,
                tf_seconds=bar_dur,
            )
            record["projected_price"] = round(projected_price, 8)
            dist_to_line = (features["close"] - projected_price) / features["close"] if features["close"] > 0 else 0
            record["dist_to_line_pct"] = round(dist_to_line, 6)
        for key in (
            "touch_count",
            "recent_touch_count",
            "body_violation_count",
            "wick_rejection_count",
            "distance_to_line_atr",
            "htf_confluence_score",
        ):
            if key in features:
                record[key] = features[key]

    # Save
    ML_DRAWINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ML_DRAWINGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    return record


def capture_position_closed_from_drawing(
    *,
    manual_line_id: str,
    symbol: str,
    timeframe: str,
    side: str,
    pnl_usd: float,
    pnl_pct: float,
    close_reason: str,
    bars_to_fill: int | None = None,
    bars_held: int | None = None,
    features_at_close: dict | None = None,
    **extra,
) -> dict:
    """Close the learning loop on a user-drawn (or conditional-derived) line.

    Called when a position that originated from a manual_line_id closes.
    Writes a 'position_closed_from_drawing' event so the user_drawings_ml.jsonl
    file has: user drew line → triggered → closed → outcome + features.
    """
    record = {
        "event": "position_closed_from_drawing",
        "ts": time.time(),
        "dt": datetime.now(timezone.utc).isoformat(),
        "manual_line_id": manual_line_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "close_reason": close_reason,
        "bars_to_fill": bars_to_fill,
        "bars_held": bars_held,
        "features_at_close": features_at_close or {},
        **extra,
    }
    ML_DRAWINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ML_DRAWINGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def capture_live_outcome(
    *,
    manual_line_id: str,
    symbol: str,
    timeframe: str,
    side: str,
    direction: str,
    entry_ts: int | float,
    entry_price: float,
    exit_ts: int | float,
    exit_price: float,
    exit_reason: str,
    qty: float,
    leverage: float,
    pnl_usd: float,
    pnl_pct: float,
    fee_pct_total: float = 0.12,          # 0.06% × 2 (Bitget taker, roundtrip)
    initial_stop_price: float | None = None,
    initial_tp_price: float | None = None,
    mfe_price: float | None = None,
    mae_price: float | None = None,
    exchange_order_id: str | None = None,
    conditional_id: str | None = None,
    **extra,
) -> dict:
    """Append a real-trade outcome record to user_drawing_outcomes.jsonl.

    Complements the batch-simulation outcomes from label_user_drawings.py —
    this is REAL fill data from the exchange. Distinguished via
    `source: "live"` so downstream analysis can filter.

    User 2026-04-21: asked for real-time outcome capture so the Excel
    "结果" sheet shows what ACTUALLY happened on the exchange, not just
    historical sims.

    Computes R ratio: R = pnl_usd / initial_risk_usd where initial_risk
    is |entry - initial_stop| × qty. If initial_stop is missing, R is
    None and downstream can skip the line.
    """
    # Compute initial risk + R-multiple
    initial_risk_usd = None
    realized_r = None
    if initial_stop_price is not None and entry_price > 0 and qty > 0:
        initial_risk_usd = abs(entry_price - float(initial_stop_price)) * qty
        if initial_risk_usd > 0:
            realized_r = pnl_usd / initial_risk_usd

    # Net-of-fee R: subtract roundtrip fee_pct of notional from pnl
    notional = entry_price * qty
    fee_usd = notional * (fee_pct_total / 100.0)
    pnl_net = pnl_usd - fee_usd
    realized_r_net = (pnl_net / initial_risk_usd) if (initial_risk_usd and initial_risk_usd > 0) else None

    # MFE / MAE as R (if price tracking available)
    mfe_r = None
    mae_r = None
    if initial_risk_usd and initial_risk_usd > 0 and qty > 0:
        if mfe_price is not None and entry_price > 0:
            mfe_pnl_raw = (float(mfe_price) - entry_price) * qty
            if direction == "short": mfe_pnl_raw = -mfe_pnl_raw
            mfe_r = mfe_pnl_raw / initial_risk_usd
        if mae_price is not None and entry_price > 0:
            mae_pnl_raw = (float(mae_price) - entry_price) * qty
            if direction == "short": mae_pnl_raw = -mae_pnl_raw
            mae_r = mae_pnl_raw / initial_risk_usd

    record = {
        "event": "user_drawing_outcome",
        "source": "live",                    # mark as real, not simulation
        "manual_line_id": manual_line_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "direction": direction,
        "capture_ts": int(time.time()),
        "config": {
            "fee_bps": fee_pct_total * 100,   # 0.12% → 12 bps
            "leverage": leverage,
        },
        "status": "closed",
        "filled": True,
        "entry_ts": int(_timestamp_seconds(entry_ts)),
        "entry_price": entry_price,
        "exit_ts": int(_timestamp_seconds(exit_ts)),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "initial_stop_price": initial_stop_price,
        "initial_tp_price": initial_tp_price,
        "initial_risk_usd": initial_risk_usd,
        "qty": qty,
        "notional_usd": notional,
        "pnl_usd_gross": pnl_usd,
        "pnl_usd_net": pnl_net,
        "pnl_pct": pnl_pct,
        "realized_r_gross": realized_r,
        "realized_r": realized_r_net,
        "mfe_price": mfe_price,
        "mae_price": mae_price,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "exchange_order_id": exchange_order_id,
        "conditional_id": conditional_id,
        **extra,
    }
    OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTCOMES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def capture_user_order_intent(
    *,
    manual_line_id: str,
    symbol: str,
    timeframe: str,
    side: str,
    direction: str,
    order_kind: str,
    line_price: float,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    tolerance_pct: float,
    stop_offset_pct: float,
    rr_target: float,
    size_usdt: float | None,
    exchange_order_id: str | None = None,
) -> dict:
    """Record that the user converted a drawing into an order.

    This is a stronger positive label than "line exists": it means the
    user judged this line good enough to trade.
    """
    record = {
        "event": "user_order_intent",
        "manual_line_id": manual_line_id,
        "ts": time.time(),
        "dt": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "direction": direction,
        "order_kind": order_kind,
        "line_price": line_price,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "tp_price": tp_price,
        "tolerance_pct": tolerance_pct,
        "stop_offset_pct": stop_offset_pct,
        "rr_target": rr_target,
        "size_usdt": size_usdt,
        "exchange_order_id": exchange_order_id,
        "label": 1,
        "label_reason": "user_placed_line_order",
    }
    ML_DRAWINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ML_DRAWINGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def get_user_drawings(last_n: int = 500) -> list[dict]:
    if not ML_DRAWINGS_FILE.exists():
        return []
    lines = ML_DRAWINGS_FILE.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines[-last_n:] if l.strip()]
