from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence

from .config import StrategyConfig, calculate_atr
from .factors import calculate_resistance_short_score, calculate_support_long_score
from .types import StrategySignal, Trendline, ensure_candles_df, project_price

LineLifecycleState = Literal["candidate", "confirmed", "armed", "triggered", "invalidated", "expired", "closed"]
SignalLifecycleState = Literal["detected", "active", "suppressed", "expired", "consumed"]

LINE_TRANSITIONS: dict[LineLifecycleState, set[LineLifecycleState]] = {
    "candidate": {"confirmed", "invalidated", "expired"},
    "confirmed": {"candidate", "armed", "triggered", "invalidated", "expired"},
    "armed": {"candidate", "confirmed", "triggered", "invalidated", "expired"},
    "triggered": {"invalidated", "expired", "closed"},
    "invalidated": set(),
    "expired": set(),
    "closed": set(),
}

SIGNAL_TRANSITIONS: dict[SignalLifecycleState, set[SignalLifecycleState]] = {
    "detected": {"active", "suppressed", "expired"},
    "active": {"consumed", "expired"},
    "suppressed": set(),
    "expired": set(),
    "consumed": set(),
}


@dataclass(frozen=True, slots=True)
class LineStateSnapshot:
    line_id: str
    state: LineLifecycleState
    previous_state: LineLifecycleState | None
    side: str
    symbol: str
    timeframe: str
    line_score: float
    confirming_touch_count: int
    bar_touch_count: int
    projected_price_current: float
    projected_price_next: float
    latest_confirming_touch_index: int | None
    bars_since_last_confirming_touch: int
    invalidation_reason: str | None
    transition_reason: str
    factor_score: float | None = None
    arm_distance: float | None = None
    distance_to_next_projection: float | None = None
    signal_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SignalStateSnapshot:
    signal_id: str
    line_id: str
    symbol: str
    timeframe: str
    state: SignalLifecycleState
    priority_rank: int | None
    trigger_mode: str
    direction: str
    score: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def can_transition(current: str, target: str, transitions: Mapping[str, set[str]]) -> bool:
    return current == target or target in transitions.get(current, set())


def validate_line_transition(current: LineLifecycleState, target: LineLifecycleState) -> None:
    if not can_transition(current, target, LINE_TRANSITIONS):
        raise ValueError(f"Invalid line transition: {current} -> {target}")


def validate_signal_transition(current: SignalLifecycleState, target: SignalLifecycleState) -> None:
    if not can_transition(current, target, SIGNAL_TRANSITIONS):
        raise ValueError(f"Invalid signal transition: {current} -> {target}")


def evaluate_line_arming(
    candles,
    line: Trendline,
    config: StrategyConfig | None = None,
    *,
    bar_index: int | None = None,
) -> tuple[bool, dict[str, float | None]]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if df.empty:
        return False, {
            "factor_score": 0.0,
            "arm_distance": 0.0,
            "distance_to_next_projection": None,
        }

    current_index = len(df) - 1 if bar_index is None else bar_index
    if line.state != "confirmed" or line.invalidation_reason:
        return False, {
            "factor_score": 0.0,
            "arm_distance": 0.0,
            "distance_to_next_projection": None,
        }

    atr = calculate_atr(df, cfg.atr_period)
    atr_value = float(atr.iloc[current_index])
    close_price = float(df.iloc[current_index]["close"])
    projected_next = project_price(line.slope, line.intercept, current_index + 1)
    arm_distance = cfg.arm_distance(atr_value, close_price)
    distance_to_next_projection = abs(close_price - projected_next)

    if line.side == "resistance":
        factor_score, _ = calculate_resistance_short_score(df, line, cfg, bar_index=current_index)
    else:
        factor_score, _ = calculate_support_long_score(df, line, cfg, bar_index=current_index)

    is_armed = distance_to_next_projection <= arm_distance and factor_score >= cfg.score_threshold
    return is_armed, {
        "factor_score": float(factor_score),
        "arm_distance": float(arm_distance),
        "distance_to_next_projection": float(distance_to_next_projection),
    }


def advance_line_states(
    candles,
    lines: Sequence[Trendline],
    signals: Sequence[StrategySignal],
    config: StrategyConfig | None = None,
    *,
    previous_states: Mapping[str, LineLifecycleState] | None = None,
    bar_index: int | None = None,
) -> tuple[LineStateSnapshot, ...]:
    cfg = config or StrategyConfig()
    current_index = (len(ensure_candles_df(candles)) - 1) if bar_index is None else bar_index
    previous = dict(previous_states or {})
    signals_by_line: dict[str, list[StrategySignal]] = {}
    for signal in signals:
        signals_by_line.setdefault(signal.line_id, []).append(signal)

    snapshots: list[LineStateSnapshot] = []
    ordered_lines = sorted(lines, key=lambda line: (line.side, line.line_id))
    for line in ordered_lines:
        previous_state = previous.get(line.line_id)
        line_signals = tuple(signals_by_line.get(line.line_id, ()))
        target_state, reason, arm_context = _derive_line_state(
            candles,
            line,
            line_signals,
            cfg,
            previous_state=previous_state,
            bar_index=current_index,
        )

        if previous_state is not None:
            validate_line_transition(previous_state, target_state)

        snapshots.append(
            LineStateSnapshot(
                line_id=line.line_id,
                state=target_state,
                previous_state=previous_state,
                side=line.side,
                symbol=line.symbol,
                timeframe=line.timeframe,
                line_score=float(line.score),
                confirming_touch_count=line.confirming_touch_count,
                bar_touch_count=line.bar_touch_count,
                projected_price_current=float(line.projected_price_current),
                projected_price_next=float(line.projected_price_next),
                latest_confirming_touch_index=line.latest_confirming_touch_index,
                bars_since_last_confirming_touch=line.bars_since_last_confirming_touch,
                invalidation_reason=line.invalidation_reason,
                transition_reason=reason,
                factor_score=arm_context["factor_score"],
                arm_distance=arm_context["arm_distance"],
                distance_to_next_projection=arm_context["distance_to_next_projection"],
                signal_ids=tuple(signal.signal_id for signal in line_signals),
            )
        )

    return tuple(snapshots)


def close_line_state(line_state: LineStateSnapshot, *, reason: str = "completed") -> LineStateSnapshot:
    validate_line_transition(line_state.state, "closed")
    return LineStateSnapshot(
        line_id=line_state.line_id,
        state="closed",
        previous_state=line_state.state,
        side=line_state.side,
        symbol=line_state.symbol,
        timeframe=line_state.timeframe,
        line_score=line_state.line_score,
        confirming_touch_count=line_state.confirming_touch_count,
        bar_touch_count=line_state.bar_touch_count,
        projected_price_current=line_state.projected_price_current,
        projected_price_next=line_state.projected_price_next,
        latest_confirming_touch_index=line_state.latest_confirming_touch_index,
        bars_since_last_confirming_touch=line_state.bars_since_last_confirming_touch,
        invalidation_reason=line_state.invalidation_reason,
        transition_reason=reason,
        factor_score=line_state.factor_score,
        arm_distance=line_state.arm_distance,
        distance_to_next_projection=line_state.distance_to_next_projection,
        signal_ids=line_state.signal_ids,
    )


def build_signal_state_snapshots(
    prioritized_signals: Sequence[StrategySignal],
    selected_signals: Sequence[StrategySignal],
    *,
    active_directions: Mapping[str, str] | None = None,
) -> tuple[SignalStateSnapshot, ...]:
    active = active_directions or {}
    selected_ids = {signal.signal_id for signal in selected_signals}

    snapshots: list[SignalStateSnapshot] = []
    for signal in prioritized_signals:
        if signal.signal_id in selected_ids:
            target_state: SignalLifecycleState = "active"
            reason = "selected"
        elif signal.symbol in active and active[signal.symbol] != signal.direction:
            target_state = "suppressed"
            reason = "active_direction_conflict"
        else:
            target_state = "suppressed"
            reason = "same_symbol_conflict"

        validate_signal_transition("detected", target_state)
        snapshots.append(
            SignalStateSnapshot(
                signal_id=signal.signal_id,
                line_id=signal.line_id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                state=target_state,
                priority_rank=signal.priority_rank,
                trigger_mode=signal.trigger_mode,
                direction=signal.direction,
                score=float(signal.score),
                reason=reason,
            )
        )
    return tuple(snapshots)


def _derive_line_state(
    candles,
    line: Trendline,
    signals: Sequence[StrategySignal],
    config: StrategyConfig,
    *,
    previous_state: LineLifecycleState | None,
    bar_index: int,
) -> tuple[LineLifecycleState, str, dict[str, float | None]]:
    if previous_state in {"invalidated", "expired", "closed"}:
        return previous_state, "terminal_state", {
            "factor_score": None,
            "arm_distance": None,
            "distance_to_next_projection": None,
        }

    if line.invalidation_reason or line.state == "invalidated":
        return "invalidated", line.invalidation_reason or "invalidated", {
            "factor_score": None,
            "arm_distance": None,
            "distance_to_next_projection": None,
        }

    if line.state == "expired":
        return "expired", "max_fresh_bars", {
            "factor_score": None,
            "arm_distance": None,
            "distance_to_next_projection": None,
        }

    if previous_state == "triggered":
        return "triggered", "sticky_triggered", {
            "factor_score": None,
            "arm_distance": None,
            "distance_to_next_projection": None,
        }

    if signals:
        return "triggered", "signal_triggered", {
            "factor_score": None,
            "arm_distance": None,
            "distance_to_next_projection": None,
        }

    if line.state == "confirmed":
        armed, arm_context = evaluate_line_arming(candles, line, config, bar_index=bar_index)
        if armed:
            return "armed", "arm_zone", arm_context
        return "confirmed", "confirmed", arm_context

    return "candidate", "candidate", {
        "factor_score": None,
        "arm_distance": None,
        "distance_to_next_projection": None,
    }


__all__ = [
    "LINE_TRANSITIONS",
    "SIGNAL_TRANSITIONS",
    "LineLifecycleState",
    "LineStateSnapshot",
    "SignalLifecycleState",
    "SignalStateSnapshot",
    "advance_line_states",
    "build_signal_state_snapshots",
    "can_transition",
    "close_line_state",
    "evaluate_line_arming",
    "validate_line_transition",
    "validate_signal_transition",
]
