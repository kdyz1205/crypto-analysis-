from __future__ import annotations

from typing import Sequence

from .config import StrategyConfig
from .types import Pivot, ensure_candles_df, stable_id


def detect_pivots(candles, config: StrategyConfig | None = None) -> list[Pivot]:
    cfg = config or StrategyConfig()
    df = ensure_candles_df(candles)
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    timestamps = df["timestamp"].tolist()

    pivots: list[Pivot] = []
    start = cfg.pivot_left
    end = len(df) - cfg.pivot_right

    for index in range(start, end):
        left_highs = highs[index - cfg.pivot_left : index]
        right_highs = highs[index + 1 : index + 1 + cfg.pivot_right]
        left_lows = lows[index - cfg.pivot_left : index]
        right_lows = lows[index + 1 : index + 1 + cfg.pivot_right]

        if highs[index] > max(left_highs) and highs[index] >= max(right_highs):
            pivots.append(
                Pivot(
                    pivot_id=stable_id("pivot", "high", index, timestamps[index], highs[index]),
                    kind="high",
                    index=index,
                    timestamp=timestamps[index],
                    price=float(highs[index]),
                    left_bars=cfg.pivot_left,
                    right_bars=cfg.pivot_right,
                    confirmed_at_index=index + cfg.pivot_right,
                )
            )

        if lows[index] < min(left_lows) and lows[index] <= min(right_lows):
            pivots.append(
                Pivot(
                    pivot_id=stable_id("pivot", "low", index, timestamps[index], lows[index]),
                    kind="low",
                    index=index,
                    timestamp=timestamps[index],
                    price=float(lows[index]),
                    left_bars=cfg.pivot_left,
                    right_bars=cfg.pivot_right,
                    confirmed_at_index=index + cfg.pivot_right,
                )
            )

    pivots.sort(key=lambda pivot: (pivot.index, pivot.kind))
    return pivots


def filter_confirmed_pivots(
    pivots: Sequence[Pivot],
    *,
    kind: str | None = None,
    up_to_index: int | None = None,
) -> list[Pivot]:
    filtered: list[Pivot] = []
    for pivot in pivots:
        if kind is not None and pivot.kind != kind:
            continue
        if up_to_index is not None and pivot.confirmed_at_index > up_to_index:
            continue
        filtered.append(pivot)
    return filtered


__all__ = [
    "detect_pivots",
    "filter_confirmed_pivots",
]
