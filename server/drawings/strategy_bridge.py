from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..strategy import (
    StrategyConfig,
    build_signal_state_snapshots,
    generate_failed_breakout_signals,
    generate_pre_limit_signals,
    generate_rejection_signals,
    prioritize_signals,
    resolve_signal_conflicts,
    select_active_lines,
)
from ..strategy.replay import ReplaySnapshot
from ..strategy.types import Trendline, ensure_candles_df, project_price, stable_id
from .store import ManualTrendlineStore
from .types import ManualTrendline

manual_strategy_store = ManualTrendlineStore()
STRATEGY_OVERRIDE_MODES = {"promote_to_active", "strategy_input_enabled"}


def manual_strategy_signature(symbol: str, timeframe: str) -> tuple:
    drawings = manual_strategy_store.list(symbol=symbol, timeframe=timeframe)
    relevant = [
        drawing
        for drawing in drawings
        if drawing.override_mode in STRATEGY_OVERRIDE_MODES
    ]
    signature: list[object] = []
    for drawing in relevant:
        signature.extend(
            (
                drawing.manual_line_id,
                drawing.updated_at,
                drawing.override_mode,
                drawing.comparison_status,
                drawing.nearest_auto_line_id or "",
                drawing.t_start,
                drawing.t_end,
                drawing.price_start,
                drawing.price_end,
            )
        )
    return tuple(signature)


def augment_snapshot_with_manual_signals(
    snapshot: ReplaySnapshot,
    candles,
    config: StrategyConfig | None = None,
    *,
    symbol: str,
    timeframe: str,
    drawings: Sequence[ManualTrendline] | None = None,
    active_directions: Mapping[str, str] | None = None,
    enabled_trigger_modes: Sequence[str] | None = None,
) -> ReplaySnapshot:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    manual_lines = build_manual_signal_lines(
        drawings if drawings is not None else manual_strategy_store.list(symbol=symbol, timeframe=timeframe),
        df,
        cfg,
        symbol=symbol,
        timeframe=timeframe,
    )
    if not manual_lines:
        return snapshot

    auto_active_lines = select_active_lines(snapshot.candidate_lines, cfg)
    combined_lines = tuple(auto_active_lines) + tuple(manual_lines)
    detected = []
    enabled = set(("pre_limit", "rejection", "failed_breakout") if enabled_trigger_modes is None else enabled_trigger_modes)
    if "pre_limit" in enabled:
        detected.extend(generate_pre_limit_signals(df, combined_lines, cfg))
    if "rejection" in enabled:
        detected.extend(generate_rejection_signals(df, combined_lines, cfg))
    if "failed_breakout" in enabled:
        detected.extend(generate_failed_breakout_signals(df, combined_lines, cfg))
    prioritized = tuple(prioritize_signals(detected, cfg))
    selected = tuple(resolve_signal_conflicts(prioritized, active_directions=active_directions))
    signal_states = build_signal_state_snapshots(
        prioritized,
        selected,
        active_directions=active_directions,
    )
    return replace(snapshot, signals=selected, signal_states=signal_states)


def build_manual_signal_lines(
    drawings: Sequence[ManualTrendline],
    candles,
    config: StrategyConfig | None = None,
    *,
    symbol: str,
    timeframe: str,
) -> tuple[Trendline, ...]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if df.empty:
        return ()

    current_index = len(df) - 1
    lines: list[Trendline] = []
    for drawing in drawings:
        if drawing.symbol != symbol or drawing.timeframe != timeframe:
            continue
        if drawing.override_mode not in STRATEGY_OVERRIDE_MODES:
            continue
        start_index = _nearest_bar_index(df, drawing.t_start)
        end_index = _nearest_bar_index(df, drawing.t_end)
        latest_anchor = max(start_index, end_index)
        if latest_anchor > current_index:
            continue
        if start_index == end_index:
            if current_index > start_index:
                end_index = current_index
            else:
                continue
        start_price = drawing.price_start
        end_price = drawing.price_end if end_index >= start_index else drawing.price_start
        slope = (drawing.price_end - drawing.price_start) / max(end_index - start_index, 1)
        intercept = drawing.price_start - (slope * start_index)
        latest_touch_index = max(start_index, end_index)
        bars_since_last_touch = max(0, current_index - latest_touch_index)
        invalidation_reason, invalidation_index = _detect_manual_invalidation(
            df,
            slope,
            intercept,
            side=drawing.side,
            config=cfg,
            start_index=latest_touch_index,
            current_index=current_index,
        )
        if invalidation_reason:
            state = "invalidated"
        elif bars_since_last_touch > cfg.max_fresh_bars:
            state = "expired"
        else:
            state = "confirmed"

        confirming_indices = tuple(sorted({start_index, end_index}))
        source = _manual_source(drawing)
        score = 98.0 if drawing.override_mode == "promote_to_active" else 94.0
        lines.append(
            Trendline(
                line_id=stable_id("manual-line", drawing.manual_line_id),
                side=drawing.side,
                symbol=symbol,
                timeframe=timeframe,
                state=state,
                anchor_pivot_ids=(f"{drawing.manual_line_id}:start", f"{drawing.manual_line_id}:end"),
                confirming_touch_pivot_ids=(
                    f"{drawing.manual_line_id}:start",
                    f"{drawing.manual_line_id}:end",
                    f"{drawing.manual_line_id}:manual",
                ),
                anchor_indices=(start_index, end_index),
                anchor_prices=(start_price, end_price),
                slope=float(slope),
                intercept=float(intercept),
                confirming_touch_indices=confirming_indices,
                bar_touch_indices=(),
                confirming_touch_count=max(3, len(confirming_indices)),
                bar_touch_count=0,
                recent_bar_touch_count=0,
                residuals=tuple(0.0 for _ in confirming_indices),
                score=score,
                score_components={
                    "normalized_mean_residual": 0.0,
                    "manual_override": 1.0,
                },
                projected_price_current=project_price(slope, intercept, current_index),
                projected_price_next=project_price(slope, intercept, current_index + 1),
                latest_confirming_touch_index=latest_touch_index,
                latest_confirming_touch_price=float(drawing.price_end),
                bars_since_last_confirming_touch=bars_since_last_touch,
                recent_test_count=0,
                non_touch_cross_count=0,
                invalidation_reason=invalidation_reason,
                invalidation_index=invalidation_index,
                source=source,
            )
        )
    return tuple(lines)


def _manual_source(drawing: ManualTrendline) -> str:
    if drawing.nearest_auto_line_id and drawing.comparison_status in {"supports_auto", "near_auto"}:
        return "hybrid"
    return "manual"


def _nearest_bar_index(df, target_ts: int) -> int:
    timestamps = df["timestamp"].tolist()
    best_index = 0
    best_distance = 10**18
    for index, timestamp in enumerate(timestamps):
        distance = abs(int(timestamp) - int(target_ts))
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _detect_manual_invalidation(
    df,
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
        close_price = float(df.iloc[bar_index]["close"])
        atr_proxy = max(
            float(df.iloc[bar_index]["high"]) - float(df.iloc[bar_index]["low"]),
            config.tick_size,
        )
        break_distance = config.break_distance(atr_proxy, close_price)
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
    "augment_snapshot_with_manual_signals",
    "build_manual_signal_lines",
    "manual_strategy_signature",
]
