from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from .config import StrategyConfig, calculate_atr
from .factors import calculate_resistance_short_score, calculate_support_long_score
from .types import StrategySignal, Trendline, ensure_candles_df, project_price, stable_id


def generate_pre_limit_signals(candles, lines: Sequence[Trendline], config: StrategyConfig | None = None) -> list[StrategySignal]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    atr = calculate_atr(df, cfg.atr_period)
    current_index = len(df) - 1
    signals: list[StrategySignal] = []

    for line in lines:
        if line.state != "confirmed":
            continue

        atr_value = float(atr.iloc[current_index])
        close_price = float(df.iloc[current_index]["close"])
        arm_distance = cfg.arm_distance(atr_value, close_price)
        projected_next = project_price(line.slope, line.intercept, current_index + 1)
        if abs(close_price - projected_next) > arm_distance:
            continue

        if line.side == "resistance":
            factor_score, factor_components = calculate_resistance_short_score(df, line, cfg, bar_index=current_index)
            if factor_score < cfg.score_threshold:
                continue
            entry_price = projected_next - cfg.entry_buffer(atr_value, close_price)
            stop_source = max(projected_next, float(line.latest_confirming_touch_price or projected_next))
            stop_price = stop_source + cfg.stop_buffer(atr_value, close_price)
            direction = "short"
            signal_type = "PRE_LIMIT_SHORT"
            rr = cfg.dynamic_rr_target(score=factor_score)
            tp_price = entry_price - (rr * abs(stop_price - entry_price))
        else:
            factor_score, factor_components = calculate_support_long_score(df, line, cfg, bar_index=current_index)
            if factor_score < cfg.score_threshold:
                continue
            entry_price = projected_next + cfg.entry_buffer(atr_value, close_price)
            stop_source = min(projected_next, float(line.latest_confirming_touch_price or projected_next))
            stop_price = stop_source - cfg.stop_buffer(atr_value, close_price)
            direction = "long"
            signal_type = "PRE_LIMIT_LONG"
            rr = cfg.dynamic_rr_target(score=factor_score)
            tp_price = entry_price + (rr * abs(entry_price - stop_price))

        # RR gate
        risk = abs(entry_price - stop_price)
        reward = abs(tp_price - entry_price)
        if risk > 0 and (reward / risk) < cfg.min_rr_ratio:
            continue

        signals.append(
            _make_signal(
                df,
                line,
                trigger_mode="pre_limit",
                signal_type=signal_type,
                direction=direction,
                bar_index=current_index,
                score=factor_score,
                entry_price=entry_price,
                stop_price=stop_price,
                tp_price=tp_price,
                factor_components=factor_components,
            )
        )

    return signals


def generate_rejection_signals(candles, lines: Sequence[Trendline], config: StrategyConfig | None = None) -> list[StrategySignal]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    atr = calculate_atr(df, cfg.atr_period)
    current_index = len(df) - 1
    signals: list[StrategySignal] = []

    for line in lines:
        if line.state != "confirmed":
            continue

        atr_value = float(atr.iloc[current_index])
        close_price = float(df.iloc[current_index]["close"])
        line_value = project_price(line.slope, line.intercept, current_index)
        tolerance = cfg.tolerance(atr_value, close_price)
        rejection_buffer = cfg.rejection_close_buffer(atr_value, close_price)

        if line.side == "resistance":
            high = float(df.iloc[current_index]["high"])
            if high < line_value - tolerance or close_price >= line_value - rejection_buffer:
                continue
            wick_ratio = _upper_wick_ratio(df, current_index, cfg)
            if wick_ratio < cfg.wick_ratio_threshold:
                continue
            factor_score, factor_components = calculate_resistance_short_score(df, line, cfg, bar_index=current_index)
            if factor_score < cfg.score_threshold:
                continue
            entry_price = close_price
            stop_price = high + cfg.stop_buffer(atr_value, close_price)
            tp_price = entry_price - (cfg.dynamic_rr_target(score=factor_score) * abs(stop_price - entry_price))
            risk = abs(entry_price - stop_price)
            reward = abs(tp_price - entry_price)
            if risk > 0 and (reward / risk) < cfg.min_rr_ratio:
                continue
            signals.append(
                _make_signal(
                    df,
                    line,
                    trigger_mode="rejection",
                    signal_type="REJECTION_SHORT",
                    direction="short",
                    bar_index=current_index,
                    score=factor_score,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    factor_components=factor_components,
                )
            )
        else:
            low = float(df.iloc[current_index]["low"])
            if low > line_value + tolerance or close_price <= line_value + rejection_buffer:
                continue
            wick_ratio = _lower_wick_ratio(df, current_index, cfg)
            if wick_ratio < cfg.wick_ratio_threshold:
                continue
            factor_score, factor_components = calculate_support_long_score(df, line, cfg, bar_index=current_index)
            if factor_score < cfg.score_threshold:
                continue
            entry_price = close_price
            stop_price = low - cfg.stop_buffer(atr_value, close_price)
            tp_price = entry_price + (cfg.dynamic_rr_target(score=factor_score) * abs(entry_price - stop_price))
            risk = abs(entry_price - stop_price)
            reward = abs(tp_price - entry_price)
            if risk > 0 and (reward / risk) < cfg.min_rr_ratio:
                continue
            signals.append(
                _make_signal(
                    df,
                    line,
                    trigger_mode="rejection",
                    signal_type="REJECTION_LONG",
                    direction="long",
                    bar_index=current_index,
                    score=factor_score,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    factor_components=factor_components,
                )
            )

    return signals


def generate_failed_breakout_signals(candles, lines: Sequence[Trendline], config: StrategyConfig | None = None) -> list[StrategySignal]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if len(df) < 2:
        return []

    atr = calculate_atr(df, cfg.atr_period)
    current_index = len(df) - 1
    previous_index = current_index - 1
    signals: list[StrategySignal] = []

    for line in lines:
        if line.state != "confirmed":
            continue

        prev_atr = float(atr.iloc[previous_index])
        prev_close = float(df.iloc[previous_index]["close"])
        prev_line_value = project_price(line.slope, line.intercept, previous_index)
        break_tol = cfg.break_tolerance(prev_atr, prev_close)
        failed_buffer = cfg.failed_break_close_buffer(prev_atr, prev_close)
        trigger_buffer = cfg.trigger_buffer(float(atr.iloc[current_index]))

        if line.side == "resistance":
            prev_high = float(df.iloc[previous_index]["high"])
            prev_low = float(df.iloc[previous_index]["low"])
            current_low = float(df.iloc[current_index]["low"])
            if prev_high <= prev_line_value + break_tol or prev_close >= prev_line_value - failed_buffer or current_low >= prev_low - trigger_buffer:
                continue
            factor_score, factor_components = calculate_resistance_short_score(df, line, cfg, bar_index=current_index)
            if factor_score < cfg.score_threshold:
                continue
            entry_price = min(float(df.iloc[current_index]["close"]), prev_low - trigger_buffer)
            stop_price = prev_high + cfg.stop_buffer(float(atr.iloc[current_index]), float(df.iloc[current_index]["close"]))
            tp_price = entry_price - (cfg.dynamic_rr_target(score=factor_score) * abs(stop_price - entry_price))
            risk = abs(entry_price - stop_price)
            reward = abs(tp_price - entry_price)
            if risk > 0 and (reward / risk) < cfg.min_rr_ratio:
                continue
            signals.append(
                _make_signal(
                    df,
                    line,
                    trigger_mode="failed_breakout",
                    signal_type="FAILED_BREAKOUT_SHORT",
                    direction="short",
                    bar_index=current_index,
                    score=factor_score,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    factor_components=factor_components,
                )
            )
        else:
            prev_low = float(df.iloc[previous_index]["low"])
            prev_high = float(df.iloc[previous_index]["high"])
            current_high = float(df.iloc[current_index]["high"])
            if prev_low >= prev_line_value - break_tol or prev_close <= prev_line_value + failed_buffer or current_high <= prev_high + trigger_buffer:
                continue
            factor_score, factor_components = calculate_support_long_score(df, line, cfg, bar_index=current_index)
            if factor_score < cfg.score_threshold:
                continue
            entry_price = max(float(df.iloc[current_index]["close"]), prev_high + trigger_buffer)
            stop_price = prev_low - cfg.stop_buffer(float(atr.iloc[current_index]), float(df.iloc[current_index]["close"]))
            tp_price = entry_price + (cfg.dynamic_rr_target(score=factor_score) * abs(entry_price - stop_price))
            risk = abs(entry_price - stop_price)
            reward = abs(tp_price - entry_price)
            if risk > 0 and (reward / risk) < cfg.min_rr_ratio:
                continue
            signals.append(
                _make_signal(
                    df,
                    line,
                    trigger_mode="failed_breakout",
                    signal_type="FAILED_BREAKOUT_LONG",
                    direction="long",
                    bar_index=current_index,
                    score=factor_score,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    factor_components=factor_components,
                )
            )

    return signals


def generate_retest_signals(candles, lines: Sequence[Trendline], config: StrategyConfig | None = None) -> list[StrategySignal]:
    """Retest trigger: price broke through a line, pulled back to it, and got rejected.

    Old support becomes new resistance (short retest).
    Old resistance becomes new support (long retest).
    Requires: line was invalidated, price has returned to the broken level, wick rejection.
    """
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if len(df) < 5:
        return []

    atr = calculate_atr(df, cfg.atr_period)
    current_index = len(df) - 1
    signals: list[StrategySignal] = []

    for line in lines:
        # Only consider recently invalidated lines (broken within last 20 bars)
        if line.state != "invalidated":
            continue
        if line.invalidation_index is None:
            continue
        bars_since_break = current_index - line.invalidation_index
        if bars_since_break < 2 or bars_since_break > 20:
            continue

        atr_value = float(atr.iloc[current_index])
        close_price = float(df.iloc[current_index]["close"])
        line_value = project_price(line.slope, line.intercept, current_index)
        tolerance = cfg.tolerance(atr_value, close_price)

        if line.side == "support":
            # Broken support → now resistance → look for SHORT retest
            # Price should have come back UP toward the old support from below
            if close_price > line_value + tolerance:
                continue  # Price is above the line, not retesting from below
            if close_price < line_value - 2.0 * atr_value:
                continue  # Too far below — not a retest

            # Need rejection wick: upper wick touching the line area
            high = float(df.iloc[current_index]["high"])
            if high < line_value - tolerance:
                continue  # Didn't even reach the line

            wick_ratio = _upper_wick_ratio(df, current_index, cfg)
            if wick_ratio < cfg.wick_ratio_threshold:
                continue  # No rejection

            factor_score, factor_components = calculate_resistance_short_score(df, line, cfg, bar_index=current_index)
            entry_price = close_price
            stop_price = high + cfg.stop_buffer(atr_value, close_price)
            tp_price = entry_price - (cfg.dynamic_rr_target(score=factor_score) * abs(stop_price - entry_price))
            risk = abs(entry_price - stop_price)
            reward = abs(tp_price - entry_price)
            if risk <= 0 or (reward / risk) < cfg.min_rr_ratio:
                continue

            signals.append(_make_signal(
                df, line,
                trigger_mode="retest",
                signal_type="RETEST_SHORT",
                direction="short",
                bar_index=current_index,
                score=max(factor_score, 0.5),
                entry_price=entry_price,
                stop_price=stop_price,
                tp_price=tp_price,
                factor_components=factor_components,
            ))

        else:
            # Broken resistance → now support → look for LONG retest
            if close_price < line_value - tolerance:
                continue  # Price is below the line
            if close_price > line_value + 2.0 * atr_value:
                continue  # Too far above

            low = float(df.iloc[current_index]["low"])
            if low > line_value + tolerance:
                continue  # Didn't reach the line

            wick_ratio = _lower_wick_ratio(df, current_index, cfg)
            if wick_ratio < cfg.wick_ratio_threshold:
                continue

            factor_score, factor_components = calculate_support_long_score(df, line, cfg, bar_index=current_index)
            entry_price = close_price
            stop_price = low - cfg.stop_buffer(atr_value, close_price)
            tp_price = entry_price + (cfg.dynamic_rr_target(score=factor_score) * abs(entry_price - stop_price))
            risk = abs(entry_price - stop_price)
            reward = abs(tp_price - entry_price)
            if risk <= 0 or (reward / risk) < cfg.min_rr_ratio:
                continue

            signals.append(_make_signal(
                df, line,
                trigger_mode="retest",
                signal_type="RETEST_LONG",
                direction="long",
                bar_index=current_index,
                score=max(factor_score, 0.5),
                entry_price=entry_price,
                stop_price=stop_price,
                tp_price=tp_price,
                factor_components=factor_components,
            ))

    return signals


def generate_signals(candles, lines: Sequence[Trendline], config: StrategyConfig | None = None) -> list[StrategySignal]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    atr = calculate_atr(df, cfg.atr_period)
    current_index = len(df) - 1
    atr_value = float(atr.iloc[current_index]) if current_index >= 0 else 0.0

    signals: list[StrategySignal] = []
    signals.extend(generate_pre_limit_signals(candles, lines, cfg))
    signals.extend(generate_rejection_signals(candles, lines, cfg))
    signals.extend(generate_failed_breakout_signals(candles, lines, cfg))
    signals.extend(generate_retest_signals(candles, lines, cfg))

    # Profit-space gate: reject signals where opposing zone is too close
    if cfg.min_profit_space_atr_mult > 0 and atr_value > 0:
        signals = [
            sig for sig in signals
            if _check_profit_space(
                sig.entry_price, sig.direction, lines, atr_value, cfg.min_profit_space_atr_mult
            )
        ]

    prioritized = prioritize_signals(signals, cfg)
    return resolve_signal_conflicts(prioritized)


def prioritize_signals(signals: Sequence[StrategySignal], config: StrategyConfig | None = None) -> list[StrategySignal]:
    cfg = config or StrategyConfig()
    ordered = sorted(
        signals,
        key=lambda signal: (
            -signal.score,
            -signal.confirming_touch_count,
            signal.distance_to_line,
            -cfg.trigger_rank(signal.trigger_mode),
            signal.bars_since_last_confirming_touch,
            signal.line_id,
        ),
    )
    return [replace(signal, priority_rank=index + 1) for index, signal in enumerate(ordered)]


def resolve_signal_conflicts(
    signals: Sequence[StrategySignal],
    *,
    active_directions: Mapping[str, str] | None = None,
) -> list[StrategySignal]:
    """Minimal stage-3 conflict filter.

    This only prevents obvious same-symbol duplicates and reverse conflicts against
    already-active directions. Full position/order-aware arbitration belongs to the
    later execution and risk layers, not this pure strategy core.
    """
    active = active_directions or {}
    resolved: list[StrategySignal] = []
    seen_symbols: set[str] = set()

    for signal in signals:
        if signal.symbol in active and active[signal.symbol] != signal.direction:
            continue
        if signal.symbol in seen_symbols:
            continue
        resolved.append(signal)
        seen_symbols.add(signal.symbol)
    return resolved


def _make_signal(
    df,
    line: Trendline,
    *,
    trigger_mode: str,
    signal_type: str,
    direction: str,
    bar_index: int,
    score: float,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    factor_components,
) -> StrategySignal:
    timestamp = df.iloc[bar_index]["timestamp"]
    signal_id = stable_id(line.symbol, line.timeframe, line.line_id, timestamp, signal_type)
    risk = abs(entry_price - stop_price)
    reward = abs(tp_price - entry_price)
    risk_reward = (reward / risk) if risk else 0.0
    distance_to_line = abs(float(df.iloc[bar_index]["close"]) - project_price(line.slope, line.intercept, bar_index))

    return StrategySignal(
        signal_id=signal_id,
        line_id=line.line_id,
        symbol=line.symbol,
        timeframe=line.timeframe,
        source=line.source,
        signal_type=signal_type,
        direction=direction,
        trigger_mode=trigger_mode,
        timestamp=timestamp,
        trigger_bar_index=bar_index,
        score=score,
        priority_rank=None,
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        tp_price=float(tp_price),
        risk_reward=float(risk_reward),
        confirming_touch_count=line.confirming_touch_count,
        bars_since_last_confirming_touch=line.bars_since_last_confirming_touch,
        distance_to_line=float(distance_to_line),
        line_side=line.side,
        reason_code=signal_type.lower(),
        factor_components=factor_components,
    )


def _upper_wick_ratio(df, bar_index: int, config: StrategyConfig) -> float:
    high = float(df.iloc[bar_index]["high"])
    open_price = float(df.iloc[bar_index]["open"])
    close_price = float(df.iloc[bar_index]["close"])
    upper_wick = high - max(open_price, close_price)
    body = max(abs(close_price - open_price), config.min_body_unit)
    return upper_wick / body


def _lower_wick_ratio(df, bar_index: int, config: StrategyConfig) -> float:
    low = float(df.iloc[bar_index]["low"])
    open_price = float(df.iloc[bar_index]["open"])
    close_price = float(df.iloc[bar_index]["close"])
    lower_wick = min(open_price, close_price) - low
    body = max(abs(close_price - open_price), config.min_body_unit)
    return lower_wick / body


def _check_profit_space(
    entry_price: float,
    direction: str,
    opposing_lines: Sequence[Trendline],
    atr_value: float,
    min_space_atr: float,
) -> bool:
    """Check if there's enough room to the nearest opposing zone."""
    if not opposing_lines or atr_value <= 0:
        return True

    if direction == "long":
        above = [
            abs(line.projected_price_current - entry_price)
            for line in opposing_lines
            if line.side == "resistance" and line.projected_price_current > entry_price
               and line.state == "confirmed"
        ]
        if not above:
            return True
        nearest_distance = min(above)
    else:
        below = [
            abs(entry_price - line.projected_price_current)
            for line in opposing_lines
            if line.side == "support" and line.projected_price_current < entry_price
               and line.state == "confirmed"
        ]
        if not below:
            return True
        nearest_distance = min(below)

    return nearest_distance >= (min_space_atr * atr_value)


__all__ = [
    "generate_failed_breakout_signals",
    "generate_pre_limit_signals",
    "generate_rejection_signals",
    "generate_retest_signals",
    "generate_signals",
    "prioritize_signals",
    "resolve_signal_conflicts",
]
