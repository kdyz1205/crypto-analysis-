from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Mapping, Sequence

from .config import StrategyConfig
from .pivots import detect_pivots
from .signals import (
    generate_failed_breakout_signals,
    generate_pre_limit_signals,
    generate_rejection_signals,
    prioritize_signals,
    resolve_signal_conflicts,
)
from .state_machine import (
    LineStateSnapshot,
    SignalStateSnapshot,
    advance_line_states,
    build_signal_state_snapshots,
)
from .regime import MarketRegime, detect_regime
from .trendlines import detect_trendlines
from .types import Pivot, StrategySignal, Trendline, ensure_candles_df
from .zones import detect_horizontal_zones
from .zone_signals import generate_zone_signals


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    bar_index: int
    timestamp: Any
    pivots: tuple[Pivot, ...]
    candidate_lines: tuple[Trendline, ...]
    active_lines: tuple[LineStateSnapshot, ...]
    line_states: tuple[LineStateSnapshot, ...]
    signals: tuple[StrategySignal, ...]
    signal_states: tuple[SignalStateSnapshot, ...]
    invalidations: tuple[LineStateSnapshot, ...]
    horizontal_zones: tuple = ()  # HorizontalZone objects
    market_regime: MarketRegime | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class ReplayResult:
    symbol: str
    timeframe: str
    snapshots: tuple[ReplaySnapshot, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def replay_strategy(
    candles,
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
    enabled_trigger_modes: Sequence[str] | None = None,
    active_directions: Mapping[str, str] | None = None,
) -> ReplayResult:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    previous_line_states: dict[str, str] = {}
    snapshots: list[ReplaySnapshot] = []

    for bar_index in range(len(df)):
        window = df.iloc[: bar_index + 1]
        pivots = tuple(detect_pivots(window, cfg))
        detection = detect_trendlines(window, pivots, cfg, symbol=symbol, timeframe=timeframe)
        prioritized_signals, selected_signals = _generate_signals_for_bar(
            window,
            detection.active_lines,
            cfg,
            enabled_trigger_modes=enabled_trigger_modes,
            active_directions=active_directions,
        )
        signal_states = build_signal_state_snapshots(
            prioritized_signals,
            selected_signals,
            active_directions=active_directions,
        )
        line_states = advance_line_states(
            window,
            detection.candidate_lines,
            selected_signals,
            cfg,
            previous_states=previous_line_states,
            bar_index=bar_index,
        )
        previous_line_states = {state.line_id: state.state for state in line_states}

        active_lines = tuple(
            state for state in line_states if state.state in {"confirmed", "armed", "triggered"}
        )
        invalidations = tuple(
            state for state in line_states if state.state in {"invalidated", "expired"}
        )
        snapshots.append(
            ReplaySnapshot(
                bar_index=bar_index,
                timestamp=window.iloc[bar_index]["timestamp"],
                pivots=pivots,
                candidate_lines=detection.candidate_lines,
                active_lines=active_lines,
                line_states=line_states,
                signals=selected_signals,
                signal_states=signal_states,
                invalidations=invalidations,
            )
        )

    return ReplayResult(
        symbol=symbol,
        timeframe=timeframe,
        snapshots=tuple(snapshots),
    )


def build_latest_snapshot(
    candles,
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
    enabled_trigger_modes: Sequence[str] | None = None,
    active_directions: Mapping[str, str] | None = None,
    skip_trendline_detection: bool = False,
) -> ReplaySnapshot:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if df.empty:
        raise ValueError("No candles available for latest snapshot")

    current_index = len(df) - 1
    pivots = tuple(detect_pivots(df, cfg))
    if skip_trendline_detection:
        # Fast path used when an evolved trendline detector will replace the
        # output anyway (see EVOLVED_TRENDLINES env). Avoids the expensive
        # O(pivots² × bars) production trendline pipeline.
        from .types import TrendlineDetectionResult
        detection = TrendlineDetectionResult(candidate_lines=(), active_lines=())
    else:
        detection = detect_trendlines(df, pivots, cfg, symbol=symbol, timeframe=timeframe)

    # Market regime detection (runs before signal generation so it can gate signals)
    regime = detect_regime(df, cfg)

    # Pass all candidate lines (including invalidated) so retest signals can find them
    all_lines_for_signals = list(detection.active_lines) + [
        line for line in detection.candidate_lines if line.state == "invalidated"
    ]
    prioritized_signals, selected_signals = _generate_signals_for_bar(
        df,
        all_lines_for_signals,
        cfg,
        enabled_trigger_modes=enabled_trigger_modes,
        active_directions=active_directions,
        regime=regime,
    )
    signal_states = build_signal_state_snapshots(
        prioritized_signals,
        selected_signals,
        active_directions=active_directions,
    )
    # Snapshot consumers only need the current line state, not a replay-accurate
    # transition chain. Keeping previous_states empty avoids replaying every prefix bar.
    line_states = advance_line_states(
        df,
        detection.candidate_lines,
        selected_signals,
        cfg,
        previous_states=None,
        bar_index=current_index,
    )
    active_lines = tuple(
        state for state in line_states if state.state in {"confirmed", "armed", "triggered"}
    )
    invalidations = tuple(
        state for state in line_states if state.state in {"invalidated", "expired"}
    )
    # Horizontal zone detection (always runs, even when trendlines are empty)
    zones = tuple(detect_horizontal_zones(
        df, pivots, cfg, symbol=symbol, timeframe=timeframe, max_zones_per_side=3,
    ))

    # Zone-based signals (supplement trendline signals)
    zone_sigs = generate_zone_signals(df, zones, cfg, symbol=symbol, timeframe=timeframe)
    selected_signals = tuple(list(selected_signals) + zone_sigs)

    return ReplaySnapshot(
        bar_index=current_index,
        timestamp=df.iloc[current_index]["timestamp"],
        pivots=pivots,
        candidate_lines=detection.candidate_lines,
        active_lines=active_lines,
        line_states=line_states,
        signals=selected_signals,
        signal_states=signal_states,
        invalidations=invalidations,
        horizontal_zones=zones,
        market_regime=regime,
    )


def build_tail_snapshots(
    candles,
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
    tail: int,
    enabled_trigger_modes: Sequence[str] | None = None,
    active_directions: Mapping[str, str] | None = None,
) -> tuple[ReplaySnapshot, ...]:
    if tail <= 0:
        return ()

    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if df.empty:
        return ()

    start_index = max(0, len(df) - tail)
    previous_state_map: dict[str, str] = {}

    snapshots: list[ReplaySnapshot] = []
    for bar_index in range(start_index, len(df)):
        snapshot = build_latest_snapshot(
            df.iloc[: bar_index + 1],
            cfg,
            symbol=symbol,
            timeframe=timeframe,
            enabled_trigger_modes=enabled_trigger_modes,
            active_directions=active_directions,
        )
        if previous_state_map:
            line_states = tuple(
                replace(state, previous_state=previous_state_map.get(state.line_id))
                for state in snapshot.line_states
            )
        else:
            line_states = snapshot.line_states

        active_lines = tuple(
            state for state in line_states if state.state in {"confirmed", "armed", "triggered"}
        )
        invalidations = tuple(
            state for state in line_states if state.state in {"invalidated", "expired"}
        )
        snapshots.append(
            ReplaySnapshot(
                bar_index=snapshot.bar_index,
                timestamp=snapshot.timestamp,
                pivots=snapshot.pivots,
                candidate_lines=snapshot.candidate_lines,
                active_lines=active_lines,
                line_states=line_states,
                signals=snapshot.signals,
                signal_states=snapshot.signal_states,
                invalidations=invalidations,
            )
        )
        previous_state_map = {state.line_id: state.state for state in line_states}

    return tuple(snapshots)


def iter_replay_snapshots(
    candles,
    config: StrategyConfig | None = None,
    *,
    symbol: str = "",
    timeframe: str = "",
    enabled_trigger_modes: Sequence[str] | None = None,
    active_directions: Mapping[str, str] | None = None,
):
    result = replay_strategy(
        candles,
        config,
        symbol=symbol,
        timeframe=timeframe,
        enabled_trigger_modes=enabled_trigger_modes,
        active_directions=active_directions,
    )
    yield from result.snapshots


def _generate_signals_for_bar(
    candles,
    lines: Sequence[Trendline],
    config: StrategyConfig,
    *,
    enabled_trigger_modes: Sequence[str] | None,
    active_directions: Mapping[str, str] | None,
    regime: MarketRegime | None = None,
) -> tuple[tuple[StrategySignal, ...], tuple[StrategySignal, ...]]:
    enabled = set(("pre_limit", "rejection", "failed_breakout", "retest") if enabled_trigger_modes is None else enabled_trigger_modes)
    detected: list[StrategySignal] = []

    if "pre_limit" in enabled:
        detected.extend(generate_pre_limit_signals(candles, lines, config))
    if "rejection" in enabled:
        detected.extend(generate_rejection_signals(candles, lines, config))
    if "failed_breakout" in enabled:
        detected.extend(generate_failed_breakout_signals(candles, lines, config))
    if "retest" in enabled:
        from .signals import generate_retest_signals
        detected.extend(generate_retest_signals(candles, lines, config))

    # Regime-based filtering: suppress signals that conflict with market state
    if regime is not None:
        detected = _filter_signals_by_regime(detected, regime)

    prioritized = tuple(prioritize_signals(detected, config))
    selected = tuple(resolve_signal_conflicts(prioritized, active_directions=active_directions))
    return prioritized, selected


def _filter_signals_by_regime(
    signals: list[StrategySignal],
    regime: MarketRegime,
) -> list[StrategySignal]:
    """Suppress signals that conflict with the current market regime.

    - Strong uptrend: suppress short signals (unless very high score)
    - Strong downtrend: suppress long signals (unless very high score)
    - Compressed volatility: suppress all pre-limit signals (wait for breakout)
    """
    if regime.trend_strength < 0.4:
        return signals  # Weak trend — no filtering

    filtered = []
    for sig in signals:
        # In strong trends, suppress counter-trend signals unless score is exceptional
        if regime.trend_direction == "up" and regime.trend_strength > 0.6:
            if sig.direction == "short" and sig.score < 0.80:
                continue
        if regime.trend_direction == "down" and regime.trend_strength > 0.6:
            if sig.direction == "long" and sig.score < 0.80:
                continue

        # In compressed volatility, suppress pre-limit (wait for breakout)
        if regime.is_compressed() and sig.trigger_mode == "pre_limit":
            continue

        filtered.append(sig)
    return filtered


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


__all__ = [
    "build_tail_snapshots",
    "build_latest_snapshot",
    "ReplayResult",
    "ReplaySnapshot",
    "iter_replay_snapshots",
    "replay_strategy",
]
