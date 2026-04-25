from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from .config import StrategyConfig, calculate_atr
from .pivots import filter_confirmed_pivots
from .types import Pivot, Trendline, TrendlineDetectionResult, ensure_candles_df, project_price, stable_id


class _BarArrays(NamedTuple):
    """Pre-extracted numpy views of OHLC + ATR.

    2026-04-23 optimization: _evaluate_candidate_line used to call
    `df.iloc[bar_index]["col"]` ~15 times per bar × 500 bars × thousands of
    pivot pairs → 8+ seconds of snapshot latency. A pandas iloc lookup is
    slow (per-access dtype dispatch); a numpy float-array lookup is ~1000x
    faster. We extract once at build_candidate_lines and thread through."""
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    atr: np.ndarray


def _extract_bar_arrays(df, atr) -> _BarArrays:
    return _BarArrays(
        opens=df["open"].to_numpy(dtype=np.float64, copy=False),
        highs=df["high"].to_numpy(dtype=np.float64, copy=False),
        lows=df["low"].to_numpy(dtype=np.float64, copy=False),
        closes=df["close"].to_numpy(dtype=np.float64, copy=False),
        atr=(atr.to_numpy(dtype=np.float64, copy=False) if hasattr(atr, "to_numpy")
             else np.asarray(atr, dtype=np.float64)),
    )


def build_candidate_lines(
    candles,
    pivots: Sequence[Pivot],
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
) -> list[Trendline]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    atr = calculate_atr(df, cfg.atr_period)
    current_index = len(df) - 1
    arrays = _extract_bar_arrays(df, atr)

    from .scoring import score_lines

    selected: list[Trendline] = []
    for side in ("resistance", "support"):
        pivot_kind = "high" if side == "resistance" else "low"
        side_lines = _build_side_candidates(
            df,
            atr,
            arrays,
            pivots,
            side=side,
            pivot_kind=pivot_kind,
            config=cfg,
            symbol=symbol,
            timeframe=timeframe,
            current_index=current_index,
        )
        side_lines = _pre_score_prune_candidates(side_lines, cfg)
        side_lines = score_lines(df, side_lines, cfg)
        side_lines = _merge_similar_lines(df, side_lines, cfg, current_index=current_index)
        side_lines.sort(key=_line_sort_key)
        selected.extend(side_lines[: cfg.max_candidate_lines_per_side])
    return selected


def select_active_lines(lines: Sequence[Trendline], config: StrategyConfig | None = None) -> list[Trendline]:
    cfg = config or StrategyConfig()
    active: list[Trendline] = []
    for side in ("resistance", "support"):
        side_lines = [line for line in lines if line.side == side and line.state == "confirmed"]
        side_lines.sort(key=_line_sort_key)
        active.extend(side_lines[: cfg.max_active_lines_per_side])
    return active


def detect_trendlines(
    candles,
    pivots: Sequence[Pivot],
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
) -> TrendlineDetectionResult:
    cfg = config or StrategyConfig()
    candidate_lines = build_candidate_lines(candles, pivots, cfg, symbol=symbol, timeframe=timeframe)
    active_lines = select_active_lines(candidate_lines, cfg)
    return TrendlineDetectionResult(
        candidate_lines=tuple(candidate_lines),
        active_lines=tuple(active_lines),
    )


def _build_side_candidates(
    df,
    atr,
    arrays: _BarArrays,
    pivots: Sequence[Pivot],
    *,
    side: str,
    pivot_kind: str,
    config: StrategyConfig,
    symbol: str,
    timeframe: str,
    current_index: int,
) -> list[Trendline]:
    lookback_start = max(0, current_index - config.lookback_bars + 1)
    side_pivots = [
        pivot
        for pivot in filter_confirmed_pivots(pivots, kind=pivot_kind, up_to_index=current_index)
        if pivot.index >= lookback_start
    ]
    candidates: list[Trendline] = []

    for right_position in range(1, len(side_pivots)):
        right_pivot = side_pivots[right_position]
        left_candidates = side_pivots[max(0, right_position - config.max_anchor_combinations_per_pivot) : right_position]
        for left_pivot in left_candidates:
            candidate = _evaluate_candidate_line(
                arrays,
                side_pivots,
                left_pivot,
                right_pivot,
                side=side,
                config=config,
                symbol=symbol,
                timeframe=timeframe,
                current_index=current_index,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _evaluate_candidate_line(
    arrays: _BarArrays,
    pivots: Sequence[Pivot],
    left_pivot: Pivot,
    right_pivot: Pivot,
    *,
    side: str,
    config: StrategyConfig,
    symbol: str,
    timeframe: str,
    current_index: int,
) -> Trendline | None:
    slope = (right_pivot.price - left_pivot.price) / (right_pivot.index - left_pivot.index)
    intercept = left_pivot.price - (slope * left_pivot.index)
    opens_arr = arrays.opens
    highs_arr = arrays.highs
    lows_arr = arrays.lows
    closes_arr = arrays.closes
    atr_arr = arrays.atr
    # Check invalidation across the ENTIRE line span (left pivot to current),
    # not just from right pivot. Pierces between anchors must be caught too.
    # Skip the anchor pivot bars themselves (they define the line, not pierce it).
    anchor_indices = {left_pivot.index, right_pivot.index}
    invalidation_reason, invalidation_index = _detect_invalidation(
        arrays,
        slope,
        intercept,
        side=side,
        config=config,
        start_index=left_pivot.index,
        current_index=current_index,
        skip_indices=anchor_indices,
    )

    confirming_refs: list[tuple[int, float, Pivot]] = []
    for pivot in pivots:
        if pivot.index < left_pivot.index:
            continue
        line_value = project_price(slope, intercept, pivot.index)
        residual = abs(pivot.price - line_value)
        max_error = config.max_line_error(float(atr_arr[pivot.index]), float(closes_arr[pivot.index]))
        if residual > max_error:
            continue
        # Pivot must be on the correct side: support touches come from below, resistance from above
        if side == "support" and pivot.price < line_value - max_error:
            continue  # pivot far below support line = pierced, not touched
        if side == "resistance" and pivot.price > line_value + max_error:
            continue  # pivot far above resistance line = pierced, not touched
        confirming_refs.append((pivot.index, residual, pivot))
    if invalidation_index is not None:
        confirming_refs = [item for item in confirming_refs if item[0] <= invalidation_index]
    confirming_refs = _dedupe_touch_refs(confirming_refs, config.min_touch_spacing_bars)
    if len(confirming_refs) < 2:
        return None

    latest_touch_index = max(item[0] for item in confirming_refs)
    bars_since_last_touch = current_index - latest_touch_index
    if bars_since_last_touch > config.max_fresh_bars:
        return None

    bar_touch_refs: list[tuple[int, float]] = []
    non_touch_cross_count = 0
    bar_scan_end_index = invalidation_index if invalidation_index is not None else current_index
    for bar_index in range(left_pivot.index, bar_scan_end_index + 1):
        line_value = project_price(slope, intercept, bar_index)
        atr_value = float(atr_arr[bar_index])
        close_price = float(closes_arr[bar_index])
        tolerance = config.tolerance(atr_value, close_price)
        touch_residual = _touch_residual_arr(highs_arr, lows_arr, bar_index, line_value, side)
        if _is_bar_touch_arr(opens_arr, highs_arr, lows_arr, closes_arr, bar_index,
                             line_value, side, config, atr_value, close_price):
            bar_touch_refs.append((bar_index, touch_residual))
        elif _is_non_touch_cross_arr(opens_arr, closes_arr, bar_index, line_value, tolerance):
            non_touch_cross_count += 1

    bar_touch_refs = _dedupe_touch_refs(bar_touch_refs, config.min_touch_spacing_bars)
    confirming_touch_indices = {item[0] for item in confirming_refs}
    bar_touch_refs = [
        item
        for item in bar_touch_refs
        if item[0] not in confirming_touch_indices and item[0] > latest_touch_index
    ]
    if non_touch_cross_count > config.max_non_touch_crosses:
        return None

    recent_window_start = max(left_pivot.index, current_index - config.recent_test_window_bars + 1)
    recent_bar_touch_count = sum(1 for bar_index, _ in bar_touch_refs if bar_index >= recent_window_start)

    # ── BODY-CROSS QUALITY CHECK ─────────────────────────────────────
    # Only check bars between anchors up to the last confirming touch.
    # After the last touch, the line is a projection — violations there
    # are handled by invalidation, not by quality filtering.
    body_check_end = min(
        latest_touch_index,
        invalidation_index if invalidation_index is not None else current_index,
    )
    body_violation_count = 0
    for bar_index in range(left_pivot.index + 1, body_check_end + 1):
        lv = project_price(slope, intercept, bar_index)
        local_atr = float(atr_arr[bar_index])
        body_tol = local_atr * 0.05  # 5% ATR tolerance
        o = float(opens_arr[bar_index])
        c = float(closes_arr[bar_index])
        bh = o if o > c else c
        bl = c if o > c else o
        if side == "support" and bl < lv - body_tol:
            body_violation_count += 1
        elif side == "resistance" and bh > lv + body_tol:
            body_violation_count += 1
        if body_violation_count > 2:
            return None

    latest_touch_price = next(item[2].price for item in reversed(confirming_refs) if item[0] == latest_touch_index)
    # Promote to "confirmed" if 3+ confirming touches (the 3-touch rule)
    is_confirmed = len(confirming_refs) >= 3 and not invalidation_reason
    return Trendline(
        line_id=stable_id("line", side, left_pivot.pivot_id, right_pivot.pivot_id),
        side=side,
        symbol=symbol,
        timeframe=timeframe,
        state="invalidated" if invalidation_reason else ("confirmed" if is_confirmed else "candidate"),
        anchor_pivot_ids=(left_pivot.pivot_id, right_pivot.pivot_id),
        confirming_touch_pivot_ids=tuple(item[2].pivot_id for item in confirming_refs),
        anchor_indices=(left_pivot.index, right_pivot.index),
        anchor_prices=(left_pivot.price, right_pivot.price),
        slope=float(slope),
        intercept=float(intercept),
        confirming_touch_indices=tuple(item[0] for item in confirming_refs),
        bar_touch_indices=tuple(item[0] for item in bar_touch_refs),
        confirming_touch_count=len(confirming_refs),
        bar_touch_count=len(bar_touch_refs),
        recent_bar_touch_count=recent_bar_touch_count,
        residuals=tuple(item[1] for item in confirming_refs),
        score=0.0,
        projected_price_current=project_price(slope, intercept, current_index),
        projected_price_next=project_price(slope, intercept, current_index + 1),
        latest_confirming_touch_index=latest_touch_index,
        latest_confirming_touch_price=latest_touch_price,
        bars_since_last_confirming_touch=bars_since_last_touch,
        recent_test_count=recent_bar_touch_count,
        non_touch_cross_count=non_touch_cross_count,
        invalidation_reason=invalidation_reason,
        invalidation_index=invalidation_index,
    )


def _merge_similar_lines(df, lines: Sequence[Trendline], config: StrategyConfig, *, current_index: int) -> list[Trendline]:
    if not lines:
        return []

    atr = calculate_atr(df, config.atr_period)
    # Extract scalars ONCE (was re-extracted per candidate in a hot loop)
    atr_value = float(atr.to_numpy(copy=False)[current_index]) if hasattr(atr, "to_numpy") else float(atr[current_index])
    close_price = float(df["close"].to_numpy(copy=False)[current_index])
    price_eps = config.line_merge_price_eps(atr_value, close_price)
    slope_eps = config.line_merge_slope_eps
    ranked = sorted(lines, key=_line_sort_key)
    kept: list[Trendline] = []

    for candidate in ranked:
        duplicate = False
        for existing in kept:
            if abs(existing.slope - candidate.slope) > slope_eps:
                continue
            if abs(existing.projected_price_current - candidate.projected_price_current) > price_eps:
                continue
            duplicate = True
            break
        if not duplicate:
            kept.append(candidate)

    return kept


def _pre_score_prune_candidates(lines: Sequence[Trendline], config: StrategyConfig) -> list[Trendline]:
    ranked = sorted(
        lines,
        key=lambda line: (
            -line.confirming_touch_count,
            -line.recent_bar_touch_count,
            line.bars_since_last_confirming_touch,
            line.non_touch_cross_count,
            sum(line.residuals),
            line.line_id,
        ),
    )
    return ranked[: config.max_candidate_lines_per_side]


def _dedupe_touch_refs(refs: Sequence[tuple], min_spacing_bars: int) -> list[tuple]:
    if not refs:
        return []
    ordered = sorted(refs, key=lambda item: item[0])
    deduped: list[tuple] = [ordered[0]]
    for ref in ordered[1:]:
        last_ref = deduped[-1]
        if ref[0] - last_ref[0] >= min_spacing_bars:
            deduped.append(ref)
            continue
        if ref[1] < last_ref[1]:
            deduped[-1] = ref
    return deduped


def _line_sort_key(line: Trendline) -> tuple:
    state_rank = {"confirmed": 0, "candidate": 1, "expired": 2, "invalidated": 3}.get(line.state, 4)
    return (
        state_rank,
        -line.score,
        -line.confirming_touch_count,
        line.bars_since_last_confirming_touch,
        line.line_id,
    )


def _touch_residual_arr(highs: np.ndarray, lows: np.ndarray, bar_index: int,
                        line_value: float, side: str) -> float:
    if side == "resistance":
        return abs(float(highs[bar_index]) - line_value)
    return abs(float(lows[bar_index]) - line_value)


def _is_bar_touch_arr(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                      closes: np.ndarray, bar_index: int, line_value: float,
                      side: str, config: StrategyConfig,
                      atr_value: float, close_price: float) -> bool:
    """A valid touch requires:
    1. The wick reaches near the line (within tolerance)
    2. The close stays on the CORRECT side of the line (not pierced through)
    3. The body does NOT fully cross through the line
    """
    tolerance = config.tolerance(atr_value, close_price)
    open_price = float(opens[bar_index])
    close = float(closes[bar_index])
    body_high = open_price if open_price > close else close
    body_low = close if open_price > close else open_price

    slack = config.close_touch_slack(atr_value, close_price)

    if side == "resistance":
        high = float(highs[bar_index])
        # Wick must reach near the line
        if abs(high - line_value) > tolerance:
            return False
        # Close must be BELOW or near the line (small slack allowed)
        if close > line_value + slack:
            return False
        # Body should mostly be below the line
        if body_high > line_value + tolerance:
            return False
        return True

    # Support
    low = float(lows[bar_index])
    # Wick must reach near the line
    if abs(low - line_value) > tolerance:
        return False
    # Close must be ABOVE or near the line
    if close < line_value - slack:
        return False
    # Body should mostly be above the line
    if body_low < line_value - tolerance:
        return False
    return True


def _is_non_touch_cross_arr(opens: np.ndarray, closes: np.ndarray, bar_index: int,
                            line_value: float, tolerance: float) -> bool:
    """A cross is when the body spans across the line — either end beyond tolerance."""
    open_price = float(opens[bar_index])
    close_price = float(closes[bar_index])
    body_high = open_price if open_price > close_price else close_price
    body_low = close_price if open_price > close_price else open_price
    # Body crosses line if one side is above and the other below
    return body_low < line_value and body_high > line_value


def _detect_invalidation(
    arrays: _BarArrays,
    slope: float,
    intercept: float,
    *,
    side: str,
    config: StrategyConfig,
    start_index: int,
    current_index: int,
    skip_indices: set[int] | None = None,
) -> tuple[str | None, int | None]:
    consecutive_breaks = 0
    _skip = skip_indices or set()
    opens_arr = arrays.opens
    closes_arr = arrays.closes
    atr_arr = arrays.atr
    for bar_index in range(start_index, current_index + 1):
        if bar_index in _skip:
            continue
        line_value = project_price(slope, intercept, bar_index)
        atr_value = float(atr_arr[bar_index])
        close_price = float(closes_arr[bar_index])
        open_price = float(opens_arr[bar_index])
        break_distance = config.break_distance(atr_value, close_price)

        if side == "resistance":
            # Body fully above the line = strong break
            body_low = open_price if open_price < close_price else close_price
            if body_low > line_value + break_distance:
                return "body_break", bar_index
            # Close above line + break distance
            if close_price > line_value + break_distance:
                return "break_distance", bar_index
            broken = close_price > line_value
        else:
            # Body fully below the line = strong break
            body_high = open_price if open_price > close_price else close_price
            if body_high < line_value - break_distance:
                return "body_break", bar_index
            # Close below line - break distance
            if close_price < line_value - break_distance:
                return "break_distance", bar_index
            broken = close_price < line_value

        consecutive_breaks = consecutive_breaks + 1 if broken else 0
        if consecutive_breaks >= config.break_close_count:
            return "break_close_count", bar_index
    return None, None


__all__ = [
    "build_candidate_lines",
    "detect_trendlines",
    "select_active_lines",
]
