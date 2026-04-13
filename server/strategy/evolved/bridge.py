"""Bridge: convert evolved EvolvedLine output into the production
StrategyLineModel-compatible dict format expected by the frontend.

Allows swapping in an evolved detector via feature flag EVOLVED_TRENDLINES
without touching the snapshot pipeline.
"""

from __future__ import annotations

import os
from hashlib import sha1
from typing import Any

import pandas as pd

from .base import EvolvedLine


def _evolved_detector_enabled() -> bool:
    return os.getenv("EVOLVED_TRENDLINES", "0").lower() in ("1", "true", "yes")


def _get_active_detector():
    """Pick which evolved variant to use. Default: v2_trader (canon-based).

    Override via EVOLVED_VARIANT env var. If the tag isn't recognized,
    we raise ImportError so the caller falls back to production — we do
    NOT silently swap to a different detector.
    """
    tag = os.getenv("EVOLVED_VARIANT", "v2_trader").strip()
    if tag == "v0_baseline":
        return None  # let production path run
    try:
        if tag == "v2_trader":
            from .v2_trader import detect_lines
        elif tag == "v1a_filtered":
            from .v1a_filtered import detect_lines
        elif tag == "v1_clean":
            from .v1_clean import detect_lines
        elif tag == "v1b_recency":
            from .v1b_recency import detect_lines
        elif tag == "v1c_fadeonly":
            from .v1c_fadeonly import detect_lines
        else:
            # Unknown tag → explicit failure (caller falls back to production)
            raise ImportError(f"Unknown EVOLVED_VARIANT tag: {tag!r}")
        return detect_lines
    except ImportError as e:
        print(f"[evolved_bridge] detector load failed: {e}", flush=True)
        return None


def _stable_line_id(variant: str, symbol: str, side: str, p1: int, p2: int) -> str:
    return sha1(f"{variant}|{symbol}|{side}|{p1}|{p2}".encode()).hexdigest()[:16]


def _coerce_timestamps_to_seconds(candles: pd.DataFrame) -> list[int]:
    """Round 10 #6: candles['timestamp'] may be datetime64[ns] (~1e18) or
    seconds-since-epoch int (~1e9). Convert defensively to UNIX seconds.
    """
    ts = candles["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        return (ts.astype("int64") // 1_000_000_000).tolist()
    # Numeric — sample first value to detect ns vs s
    arr = ts.astype("int64").tolist()
    if not arr:
        return arr
    first = arr[0]
    if first > 10_000_000_000_000:  # > 10^13 → ms or ns
        if first > 10_000_000_000_000_000:  # ns
            return [int(v // 1_000_000_000) for v in arr]
        # ms
        return [int(v // 1_000) for v in arr]
    return arr


def _line_to_dict(
    line: EvolvedLine,
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    variant: str,
) -> dict[str, Any]:
    """Minimal StrategyLineModel-compatible dict produced from an EvolvedLine."""
    n = len(candles)
    timestamps = _coerce_timestamps_to_seconds(candles)

    a_idx = int(line.start_index)
    b_idx = int(line.end_index)
    a_t = int(timestamps[a_idx]) if 0 <= a_idx < n else 0
    b_t = int(timestamps[b_idx]) if 0 <= b_idx < n else 0

    span = max(1, b_idx - a_idx)
    slope = (line.end_price - line.start_price) / span
    intercept = line.start_price - slope * a_idx

    # Project current / next
    current_idx = n - 1
    proj_current = line.start_price + slope * (current_idx - a_idx)
    next_idx = current_idx + 1
    proj_next = line.start_price + slope * (next_idx - a_idx)
    current_time = int(timestamps[current_idx]) if n > 0 else 0
    # Approximate next bar time from recent step
    if n >= 2:
        step = int(timestamps[-1]) - int(timestamps[-2])
        next_time = current_time + step
    else:
        next_time = current_time

    touch_idx_list = [int(i) for i in (line.touch_indices or (a_idx, b_idx))]
    # De-dupe + sort
    touch_idx_list = sorted(set(touch_idx_list))

    is_active = line.touch_count >= 3
    state = "confirmed" if is_active else "candidate"

    return {
        "line_id": _stable_line_id(variant, symbol, line.side, a_idx, b_idx),
        "symbol": symbol,
        "timeframe": timeframe,
        "side": line.side,
        "source": f"evolved:{variant}",
        "state": state,
        "t_start": a_t,
        "t_end": b_t,
        "price_start": float(line.start_price),
        "price_end": float(line.end_price),
        "slope": float(slope),
        "intercept": float(intercept),
        "anchor_indices": [a_idx, b_idx],
        "anchor_prices": [float(line.start_price), float(line.end_price)],
        "anchor_timestamps": [a_t, b_t],
        "confirming_touch_indices": touch_idx_list,
        "bar_touch_indices": [],
        "touch_count": int(line.touch_count),
        "confirming_touch_count": int(line.touch_count),
        "bar_touch_count": 0,
        "line_score": float(line.score),
        "score_components": {},
        "projected_price_current": float(proj_current),
        "projected_price_next": float(proj_next),
        "projected_time_current": current_time,
        "projected_time_next": next_time,
        "is_active": is_active,
        "is_invalidated": False,
        "invalidation_reason": None,
        "invalidation_bar_index": None,
        "invalidation_timestamp": None,
        "display_rank": None,
        "display_class": "primary",
        "line_usability_score": float(line.score),
        "last_quality_touch_index": b_idx,
        "collapsed_invalidation_count": 0,
    }


def try_build_evolved_lines(
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> list[dict[str, Any]] | None:
    """Return a list of line dicts if evolved detector is enabled and
    produces output, else None (caller falls back to production).
    """
    if not _evolved_detector_enabled():
        return None
    detect = _get_active_detector()
    if detect is None:
        return None
    try:
        # Diagnostic: log candle shape
        print(f"[evolved_bridge] {symbol}/{timeframe}: input={len(candles)} bars cols={list(candles.columns)}", flush=True)
        lines = detect(candles, timeframe, symbol)
        print(f"[evolved_bridge] {symbol}/{timeframe}: detector returned {len(lines)} lines", flush=True)
    except Exception as e:
        import traceback
        print(f"[evolved_bridge] detect failed {symbol} {timeframe}: {e}\n{traceback.format_exc()}", flush=True)
        return None
    if not lines:
        return []
    variant = os.getenv("EVOLVED_VARIANT", "v1a_filtered")
    return [_line_to_dict(l, candles, symbol, timeframe, variant) for l in lines]
