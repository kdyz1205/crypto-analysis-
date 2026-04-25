"""Runtime trendline detector.

Wraps `sr_patterns.detect_patterns` to operate on an in-memory DataFrame
(supplied by FeatureCache) and emit canonical TrendlineRecord objects.
Fails gracefully when sr_patterns is unavailable - returns [].
"""
from __future__ import annotations
from typing import Any

import pandas as pd

from ..schemas.trendline import TrendlineRecord, LineRole


def _role_from_pattern_result(side: str) -> LineRole:
    s = (side or "").lower()
    if "support" in s:
        return "support"
    if "resistance" in s:
        return "resistance"
    if "channel_upper" in s:
        return "channel_upper"
    if "channel_lower" in s:
        return "channel_lower"
    if "wedge" in s:
        return "wedge_side"
    if "triangle" in s:
        return "triangle_side"
    return "unknown"


def detect_lines(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    sr_params: dict[str, Any] | None = None,
    max_lines: int | None = None,
) -> list[TrendlineRecord]:
    """Detect trendlines on the given DataFrame, return canonical records."""
    if df is None or len(df) < 50:
        return []
    try:
        from sr_patterns import detect_patterns, SRParams
    except Exception as exc:
        print(f"[runtime_detector] sr_patterns unavailable: {exc}")
        return []

    sr_params = sr_params or {}
    params = SRParams(**{k: v for k, v in sr_params.items()
                         if k in SRParams.__dataclass_fields__})
    # sr_patterns expects polars; convert if we got pandas.
    df_for_detect = df
    try:
        import polars as pl  # noqa: F401
        if isinstance(df, pd.DataFrame):
            df_for_detect = pl.from_pandas(df)
    except ImportError:
        pass
    try:
        result = detect_patterns(df_for_detect, params)
    except Exception as exc:
        print(f"[runtime_detector] detect_patterns failed {symbol} {timeframe}: {exc}")
        return []

    buckets: list[tuple[str, list]] = [
        ("support", list(getattr(result, "support_lines", []) or [])),
        ("resistance", list(getattr(result, "resistance_lines", []) or [])),
    ]
    for tri in list(getattr(result, "triangles", []) or []):
        sup = getattr(tri, "support_line", None)
        res = getattr(tri, "resistance_line", None)
        if sup is not None:
            buckets.append(("triangle_side", [sup]))
        if res is not None:
            buckets.append(("triangle_side", [res]))

    lines: list[TrendlineRecord] = []
    ts_col = "open_time" if "open_time" in df.columns else (
        "timestamp" if "timestamp" in df.columns else None
    )
    idx_global = 0
    for role_tag, bucket in buckets:
        for p in bucket:
            a1_idx = int(getattr(p, "x1", 0) or 0)
            a2_idx = int(getattr(p, "x2", 0) or 0)
            a1_price = float(getattr(p, "y1", 0.0) or 0.0)
            a2_price = float(getattr(p, "y2", 0.0) or 0.0)
            if a2_idx <= a1_idx or a1_price <= 0 or a2_price <= 0:
                continue
            direction = ("up" if a2_price > a1_price * 1.001
                         else ("down" if a2_price < a1_price * 0.999 else "flat"))
            if ts_col is not None and 0 <= a1_idx < len(df) and 0 <= a2_idx < len(df):
                try:
                    t_start = int(df.iloc[a1_idx][ts_col])
                    t_end = int(df.iloc[a2_idx][ts_col])
                except (TypeError, ValueError):
                    t_start = int(pd.Timestamp(df.iloc[a1_idx][ts_col]).timestamp())
                    t_end = int(pd.Timestamp(df.iloc[a2_idx][ts_col]).timestamp())
                if t_start > 10_000_000_000:
                    t_start //= 1000
                    t_end //= 1000
            else:
                t_start = a1_idx
                t_end = a2_idx
            rid = f"runtime-{symbol}-{timeframe}-{a1_idx}-{a2_idx}-{role_tag}-{idx_global}"
            lines.append(TrendlineRecord(
                id=rid, symbol=symbol.upper(), exchange="bitget",
                timeframe=timeframe,
                start_time=t_start, end_time=t_end,
                start_bar_index=a1_idx, end_bar_index=a2_idx,
                start_price=a1_price, end_price=a2_price,
                line_role=_role_from_pattern_result(role_tag),
                direction=direction,
                touch_count=int(getattr(p, "touches", 2) or 2),
                label_source="auto",
                auto_method=f"sr_patterns.runtime",
                score=float(getattr(p, "strength", 0.0) or 0.0) or None,
                created_at=t_end,
            ))
            idx_global += 1
            if max_lines and len(lines) >= max_lines:
                return lines
    return lines
