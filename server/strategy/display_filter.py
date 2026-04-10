from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import StrategyConfig, calculate_atr, clamp
from .types import Trendline, ensure_candles_df


@dataclass(frozen=True, slots=True)
class DisplayLineMeta:
    line_id: str
    display_rank: int
    display_class: str
    line_usability_score: float
    last_quality_touch_index: int | None
    collapsed_invalidation_count: int = 1


def build_display_line_meta(
    candles,
    lines: Sequence[Trendline],
    *,
    config: StrategyConfig | None = None,
) -> dict[str, DisplayLineMeta]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    if df.empty or not lines:
        return {}

    current_index = len(df) - 1
    close_price = float(df.iloc[current_index]["close"])
    atr = calculate_atr(df, cfg.atr_period)
    atr_value = float(atr.iloc[current_index]) if current_index >= 0 else 0.0

    metas: dict[str, DisplayLineMeta] = {}
    ranked_by_side: dict[str, list[tuple[Trendline, float]]] = {"resistance": [], "support": []}

    for line in lines:
        usability = _line_usability_score(line, close_price, atr_value, cfg)
        ranked_by_side.setdefault(line.side, []).append((line, usability))

    for side, ranked_lines in ranked_by_side.items():
        ranked_lines.sort(key=lambda item: _display_sort_key(item[0], item[1]))
        for index, (line, usability) in enumerate(ranked_lines):
            if index < cfg.display_active_lines_per_side and line.state not in {"invalidated", "expired"}:
                display_class = "primary" if index == 0 else "secondary"
                display_rank = index + 1
            else:
                display_class = "debug"
                display_rank = index + 1
            metas[line.line_id] = DisplayLineMeta(
                line_id=line.line_id,
                display_rank=display_rank,
                display_class=display_class,
                line_usability_score=round(usability, 4),
                last_quality_touch_index=line.latest_confirming_touch_index,
                collapsed_invalidation_count=1,
            )
    return metas


def filter_display_touch_indices(
    line: Trendline,
    *,
    config: StrategyConfig | None = None,
) -> tuple[list[int], list[int]]:
    cfg = config or StrategyConfig()
    confirming = list(line.confirming_touch_indices)
    bar_touches = list(line.bar_touch_indices)

    if line.state not in {"confirmed", "armed", "triggered"}:
        bar_touches = []

    bar_touches = bar_touches[-cfg.display_bar_touch_limit_per_line :]
    max_total = max(cfg.display_touch_limit_per_line, len(confirming))
    overflow = max(0, len(confirming) + len(bar_touches) - max_total)
    if overflow > 0:
        bar_touches = bar_touches[overflow:]

    if len(confirming) + len(bar_touches) > cfg.display_touch_limit_per_line:
        confirming = confirming[-cfg.display_touch_limit_per_line :]
        bar_touches = []

    return confirming, bar_touches


def collapse_display_invalidations(
    lines: Sequence[Trendline],
    display_meta: Mapping[str, DisplayLineMeta],
    *,
    config: StrategyConfig | None = None,
) -> list[tuple[Trendline, int]]:
    cfg = config or StrategyConfig()
    grouped: dict[str, list[Trendline]] = {"resistance": [], "support": []}
    for line in lines:
        if line.state not in {"invalidated", "expired"}:
            continue
        grouped.setdefault(line.side, []).append(line)

    collapsed: list[tuple[Trendline, int]] = []
    for side, side_lines in grouped.items():
        sorted_lines = sorted(
            side_lines,
            key=lambda line: (
                line.invalidation_index if line.invalidation_index is not None else 10**9,
                display_meta.get(line.line_id, DisplayLineMeta(line.line_id, 999, "debug", 0.0, None)).display_rank,
                -display_meta.get(line.line_id, DisplayLineMeta(line.line_id, 999, "debug", 0.0, None)).line_usability_score,
            ),
        )
        for line in sorted_lines:
            marker_index = line.invalidation_index if line.invalidation_index is not None else line.latest_confirming_touch_index
            if marker_index is None:
                marker_index = line.anchor_indices[-1]
            if collapsed:
                last_line, last_count = collapsed[-1]
                last_marker_index = (
                    last_line.invalidation_index
                    if last_line.invalidation_index is not None
                    else last_line.latest_confirming_touch_index
                )
                if last_marker_index is None:
                    last_marker_index = last_line.anchor_indices[-1]
                if last_line.side == side and abs(marker_index - last_marker_index) <= cfg.display_invalidation_merge_bars:
                    chosen = _choose_higher_priority_line(last_line, line, display_meta)
                    collapsed[-1] = (chosen, last_count + 1)
                    continue
            collapsed.append((line, 1))
    return collapsed


def _line_usability_score(
    line: Trendline,
    close_price: float,
    atr_value: float,
    config: StrategyConfig,
) -> float:
    score_component = clamp(line.score / 100.0)
    state_component = {
        "triggered": 1.0,
        "armed": 0.95,
        "confirmed": 0.9,
        "candidate": 0.65,
        "invalidated": 0.2,
        "expired": 0.1,
    }.get(line.state, 0.0)
    recency_component = clamp(1.0 - (line.bars_since_last_confirming_touch / max(config.max_fresh_bars, 1)))
    price_distance = abs(float(line.projected_price_current) - close_price)
    distance_component = clamp(1.0 - (price_distance / max(atr_value * 3.0, config.tick_size)))
    penalty = 0.15 if line.state in {"invalidated", "expired"} else 0.0
    return 100.0 * clamp(
        (0.45 * score_component)
        + (0.25 * state_component)
        + (0.15 * recency_component)
        + (0.15 * distance_component)
        - penalty
    )


def _display_sort_key(line: Trendline, usability: float) -> tuple:
    state_rank = {"triggered": 0, "armed": 1, "confirmed": 2, "candidate": 3, "invalidated": 4, "expired": 5}.get(
        line.state,
        6,
    )
    return (
        state_rank,
        -usability,
        line.bars_since_last_confirming_touch,
        line.line_id,
    )


def _choose_higher_priority_line(
    left: Trendline,
    right: Trendline,
    display_meta: Mapping[str, DisplayLineMeta],
) -> Trendline:
    left_meta = display_meta.get(left.line_id)
    right_meta = display_meta.get(right.line_id)
    left_rank = left_meta.display_rank if left_meta is not None else 10**6
    right_rank = right_meta.display_rank if right_meta is not None else 10**6
    if left_rank != right_rank:
        return left if left_rank < right_rank else right
    left_score = left_meta.line_usability_score if left_meta is not None else 0.0
    right_score = right_meta.line_usability_score if right_meta is not None else 0.0
    if left_score != right_score:
        return left if left_score >= right_score else right
    return left if left.line_id <= right.line_id else right


__all__ = [
    "DisplayLineMeta",
    "build_display_line_meta",
    "collapse_display_invalidations",
    "filter_display_touch_indices",
]
