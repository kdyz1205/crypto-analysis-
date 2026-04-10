import pandas as pd

from server.strategy.config import StrategyConfig
from server.strategy.display_filter import (
    build_display_line_meta,
    collapse_display_invalidations,
    filter_display_touch_indices,
)
from server.strategy.types import Trendline


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": list(range(30)),
            "open": [100.0 + (index * 0.1) for index in range(30)],
            "high": [101.0 + (index * 0.1) for index in range(30)],
            "low": [99.0 + (index * 0.1) for index in range(30)],
            "close": [100.5 + (index * 0.1) for index in range(30)],
            "volume": [1000.0] * 30,
        }
    )


def _line(
    line_id: str,
    *,
    side: str = "resistance",
    state: str = "confirmed",
    score: float = 80.0,
    bars_since_last_touch: int = 2,
    invalidation_index: int | None = None,
) -> Trendline:
    return Trendline(
        line_id=line_id,
        side=side,
        symbol="TEST",
        timeframe="4h",
        state=state,
        anchor_pivot_ids=("a", "b"),
        confirming_touch_pivot_ids=("a", "b", "c"),
        anchor_indices=(2, 12),
        anchor_prices=(110.0, 105.0),
        slope=-0.5 if side == "resistance" else 0.5,
        intercept=111.0 if side == "resistance" else 90.0,
        confirming_touch_indices=(2, 8, 12),
        bar_touch_indices=(2, 8, 12, 18, 20),
        confirming_touch_count=3,
        bar_touch_count=5,
        recent_bar_touch_count=2,
        residuals=(0.1, 0.12, 0.08),
        score=score,
        score_components={},
        projected_price_current=104.0 if side == "resistance" else 96.0,
        projected_price_next=103.5 if side == "resistance" else 96.5,
        latest_confirming_touch_index=12,
        latest_confirming_touch_price=105.0 if side == "resistance" else 95.0,
        bars_since_last_confirming_touch=bars_since_last_touch,
        recent_test_count=1,
        non_touch_cross_count=0,
        invalidation_reason="break_close_count" if invalidation_index is not None else None,
        invalidation_index=invalidation_index,
    )


def test_build_display_line_meta_limits_default_lines_per_side() -> None:
    lines = [
        _line("line-a", score=90.0),
        _line("line-b", score=85.0),
        _line("line-c", score=80.0),
        _line("line-d", score=75.0),
    ]

    meta = build_display_line_meta(
        _candles(),
        lines,
        config=StrategyConfig(display_active_lines_per_side=3),
    )

    assert meta["line-a"].display_class == "primary"
    assert meta["line-b"].display_class == "secondary"
    assert meta["line-c"].display_class == "secondary"
    assert meta["line-d"].display_class == "debug"


def test_filter_display_touch_indices_caps_bar_touches() -> None:
    confirming, bar_touches = filter_display_touch_indices(
        _line("line-a"),
        config=StrategyConfig(display_touch_limit_per_line=4, display_bar_touch_limit_per_line=2),
    )

    assert confirming == [2, 8, 12]
    assert bar_touches == [20]


def test_collapse_display_invalidations_merges_nearby_events() -> None:
    lines = [
        _line("line-a", state="invalidated", score=90.0, invalidation_index=20),
        _line("line-b", state="invalidated", score=70.0, invalidation_index=22),
        _line("line-c", side="support", state="invalidated", score=88.0, invalidation_index=24),
    ]
    meta = build_display_line_meta(_candles(), lines, config=StrategyConfig(display_active_lines_per_side=2))

    collapsed = collapse_display_invalidations(
        lines,
        meta,
        config=StrategyConfig(display_invalidation_merge_bars=3),
    )

    assert len(collapsed) == 2
    resistance_line, resistance_count = collapsed[0]
    support_line, support_count = collapsed[1]
    assert resistance_line.line_id == "line-a"
    assert resistance_count == 2
    assert support_line.line_id == "line-c"
    assert support_count == 1
