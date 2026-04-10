from __future__ import annotations

from typing import Sequence

from .config import StrategyConfig, calculate_atr
from .pivots import filter_confirmed_pivots
from .types import Pivot, Trendline, TrendlineDetectionResult, ensure_candles_df, project_price, stable_id


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

    from .scoring import score_lines

    selected: list[Trendline] = []
    for side in ("resistance", "support"):
        pivot_kind = "high" if side == "resistance" else "low"
        side_lines = _build_side_candidates(
            df,
            atr,
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
            if side == "resistance" and not right_pivot.price < left_pivot.price:
                continue
            if side == "support" and not right_pivot.price > left_pivot.price:
                continue
            candidate = _evaluate_candidate_line(
                df,
                atr,
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
    df,
    atr,
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

    confirming_refs: list[tuple[int, float, Pivot]] = []
    for pivot in pivots:
        if pivot.index < left_pivot.index:
            continue
        line_value = project_price(slope, intercept, pivot.index)
        residual = abs(pivot.price - line_value)
        max_error = config.max_line_error(float(atr.iloc[pivot.index]), float(df.iloc[pivot.index]["close"]))
        if residual <= max_error:
            confirming_refs.append((pivot.index, residual, pivot))
    confirming_refs = _dedupe_touch_refs(confirming_refs, config.min_touch_spacing_bars)
    if len(confirming_refs) < 2:
        return None

    latest_touch_index = max(item[0] for item in confirming_refs)
    bars_since_last_touch = current_index - latest_touch_index
    if bars_since_last_touch > config.max_fresh_bars:
        return None

    bar_touch_refs: list[tuple[int, float]] = []
    non_touch_cross_count = 0
    for bar_index in range(left_pivot.index, current_index + 1):
        line_value = project_price(slope, intercept, bar_index)
        atr_value = float(atr.iloc[bar_index])
        close_price = float(df.iloc[bar_index]["close"])
        tolerance = config.tolerance(atr_value, close_price)
        touch_residual = _touch_residual(df, bar_index, line_value, side)
        if _is_bar_touch(df, bar_index, line_value, side, config, atr_value, close_price):
            bar_touch_refs.append((bar_index, touch_residual))
        elif _is_non_touch_cross(df, bar_index, line_value, tolerance):
            non_touch_cross_count += 1

    bar_touch_refs = _dedupe_touch_refs(bar_touch_refs, config.min_touch_spacing_bars)
    if non_touch_cross_count > config.max_non_touch_crosses:
        return None

    recent_window_start = max(left_pivot.index, current_index - config.recent_test_window_bars + 1)
    recent_bar_touch_count = sum(1 for bar_index, _ in bar_touch_refs if bar_index >= recent_window_start)
    invalidation_reason, invalidation_index = _detect_invalidation(
        df,
        atr,
        slope,
        intercept,
        side=side,
        config=config,
        start_index=latest_touch_index,
        current_index=current_index,
    )

    latest_touch_price = next(item[2].price for item in reversed(confirming_refs) if item[0] == latest_touch_index)
    return Trendline(
        line_id=stable_id("line", side, left_pivot.pivot_id, right_pivot.pivot_id),
        side=side,
        symbol=symbol,
        timeframe=timeframe,
        state="invalidated" if invalidation_reason else "candidate",
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
    ranked = sorted(lines, key=_line_sort_key)
    kept: list[Trendline] = []

    for candidate in ranked:
        atr_value = float(atr.iloc[current_index])
        close_price = float(df.iloc[current_index]["close"])
        price_eps = config.line_merge_price_eps(atr_value, close_price)
        duplicate = False
        for existing in kept:
            if abs(existing.slope - candidate.slope) > config.line_merge_slope_eps:
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


def _touch_residual(df, bar_index: int, line_value: float, side: str) -> float:
    if side == "resistance":
        return abs(float(df.iloc[bar_index]["high"]) - line_value)
    return abs(float(df.iloc[bar_index]["low"]) - line_value)


def _is_bar_touch(df, bar_index: int, line_value: float, side: str, config: StrategyConfig, atr_value: float, close_price: float) -> bool:
    tolerance = config.tolerance(atr_value, close_price)
    slack = config.close_touch_slack(atr_value, close_price)
    if side == "resistance":
        high = float(df.iloc[bar_index]["high"])
        close = float(df.iloc[bar_index]["close"])
        return abs(high - line_value) <= tolerance and close <= line_value + slack
    low = float(df.iloc[bar_index]["low"])
    close = float(df.iloc[bar_index]["close"])
    return abs(low - line_value) <= tolerance and close >= line_value - slack


def _is_non_touch_cross(df, bar_index: int, line_value: float, tolerance: float) -> bool:
    open_price = float(df.iloc[bar_index]["open"])
    close_price = float(df.iloc[bar_index]["close"])
    return min(open_price, close_price) < (line_value - tolerance) and max(open_price, close_price) > (line_value + tolerance)


def _detect_invalidation(
    df,
    atr,
    slope: float,
    intercept: float,
    *,
    side: str,
    config: StrategyConfig,
    start_index: int,
    current_index: int,
) -> tuple[str | None, int | None]:
    consecutive_breaks = 0
    for bar_index in range(start_index, current_index + 1):
        line_value = project_price(slope, intercept, bar_index)
        atr_value = float(atr.iloc[bar_index])
        close_price = float(df.iloc[bar_index]["close"])
        break_distance = config.break_distance(atr_value, close_price)

        if side == "resistance":
            broken = close_price > line_value
            if close_price > line_value + break_distance:
                return "break_distance", bar_index
        else:
            broken = close_price < line_value
            if close_price < line_value - break_distance:
                return "break_distance", bar_index

        consecutive_breaks = consecutive_breaks + 1 if broken else 0
        if consecutive_breaks >= config.break_close_count:
            return "break_close_count", bar_index
    return None, None


__all__ = [
    "build_candidate_lines",
    "detect_trendlines",
    "select_active_lines",
]
