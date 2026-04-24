"""Draw candidate trendlines for one symbol/timeframe at one SRParams point.

Reuses the existing `sr_patterns.detect_patterns` so we don't rebuild a
detector. Output = list[TrendlineRecord] under our canonical schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..schemas.trendline import TrendlineRecord, LineRole


def _ohlcv_dataframe(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load an OHLCV DataFrame for (symbol, tf) as pandas. Reuses CSVs
    via server.data_service (which returns polars) and converts. No
    look-ahead — just the raw history."""
    try:
        from server.data_service import _find_csv, _load_csv
        p = _find_csv(symbol, timeframe)
        if p is None:
            raise FileNotFoundError
        df = _load_csv(p)
        # Convert polars → pandas if needed
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        return df
    except Exception:
        cand = Path("data") / f"{symbol.upper()}_{timeframe}.csv"
        if cand.exists():
            return pd.read_csv(cand)
        return None


def _role_from_pattern_result(row: dict) -> LineRole:
    side = str(row.get("side") or row.get("type") or "").lower()
    if "support" in side:
        return "support"
    if "resistance" in side:
        return "resistance"
    if "channel_upper" in side:
        return "channel_upper"
    if "channel_lower" in side:
        return "channel_lower"
    if "wedge" in side:
        return "wedge_side"
    if "triangle" in side:
        return "triangle_side"
    return "unknown"


def draw_lines_for_symbol(
    symbol: str,
    timeframe: str,
    sr_params_kwargs: dict[str, Any],
    *,
    max_lines: int | None = None,
) -> list[TrendlineRecord]:
    """Run sr_patterns on a (symbol, tf) with given SRParams, return
    candidate lines as TrendlineRecords."""
    try:
        from sr_patterns import detect_patterns, SRParams
    except Exception as exc:
        raise RuntimeError(f"sr_patterns unavailable: {exc}") from exc

    df = _ohlcv_dataframe(symbol, timeframe)
    if df is None or len(df) < 50:
        return []

    params = SRParams(**{k: v for k, v in sr_params_kwargs.items()
                         if k in SRParams.__dataclass_fields__})
    try:
        result = detect_patterns(df, params)
    except Exception as exc:
        print(f"[evolve.draw] detect_patterns failed {symbol} {timeframe}: {exc}")
        return []

    lines: list[TrendlineRecord] = []

    # sr_patterns.PatternResult has:
    #   support_lines: list[TrendLine]  (x1, x2, y1, y2, slope, strength, tolerance, touches, line_type)
    #   resistance_lines: list[TrendLine]
    #   triangles: list[TrianglePattern]  each has .support_line + .resistance_line
    buckets: list[tuple[str, list]] = [
        ("support", list(getattr(result, "support_lines", []) or [])),
        ("resistance", list(getattr(result, "resistance_lines", []) or [])),
    ]
    # Triangles contribute two triangle_side lines each
    for tri in list(getattr(result, "triangles", []) or []):
        sup = getattr(tri, "support_line", None)
        res = getattr(tri, "resistance_line", None)
        if sup is not None:
            buckets.append(("triangle_side", [sup]))
        if res is not None:
            buckets.append(("triangle_side", [res]))

    idx_global = 0
    for role_tag, bucket in buckets:
        for p in bucket:
            # Pull TrendLine fields robustly
            a1_idx = int(getattr(p, "x1", 0) or 0)
            a2_idx = int(getattr(p, "x2", 0) or 0)
            a1_price = float(getattr(p, "y1", 0.0) or 0.0)
            a2_price = float(getattr(p, "y2", 0.0) or 0.0)
            if a2_idx <= a1_idx or a1_price <= 0 or a2_price <= 0:
                continue
            role: LineRole = role_tag  # already one of our enum values
            direction = ("up" if a2_price > a1_price * 1.001
                         else ("down" if a2_price < a1_price * 0.999 else "flat"))

            # Map bar index → wall-clock timestamp via the DataFrame
            ts_col = "open_time" if "open_time" in df.columns else (
                "timestamp" if "timestamp" in df.columns else None
            )
            if ts_col is not None and 0 <= a1_idx < len(df) and 0 <= a2_idx < len(df):
                t_start_val = df[ts_col][a1_idx] if hasattr(df, "__getitem__") else df.iloc[a1_idx][ts_col]
                t_end_val = df[ts_col][a2_idx] if hasattr(df, "__getitem__") else df.iloc[a2_idx][ts_col]
                # polars returns np.int64 / datetime — coerce
                try:
                    t_start = int(t_start_val)
                    t_end = int(t_end_val)
                except (TypeError, ValueError):
                    import pandas as _pd
                    t_start = int(_pd.Timestamp(t_start_val).timestamp())
                    t_end = int(_pd.Timestamp(t_end_val).timestamp())
                # Bitget timestamps may be ms → downshift
                if t_start > 10_000_000_000:
                    t_start //= 1000
                    t_end //= 1000
            else:
                t_start = a1_idx
                t_end = a2_idx

            rid = f"evolve-{symbol}-{timeframe}-{a1_idx}-{a2_idx}-{role}-{idx_global}"
            lines.append(TrendlineRecord(
                id=rid,
                symbol=symbol.upper(),
                exchange="bitget",
                timeframe=timeframe,
                start_time=t_start,
                end_time=t_end,
                start_bar_index=a1_idx,
                end_bar_index=a2_idx,
                start_price=a1_price,
                end_price=a2_price,
                line_role=role,
                direction=direction,
                touch_count=int(getattr(p, "touches", 2) or 2),
                rejection_strength_atr=None,
                label_source="auto",
                auto_method=f"sr_patterns.live({','.join(f'{k}={v}' for k,v in sr_params_kwargs.items())})",
                score=float(getattr(p, "strength", 0.0) or 0.0) or None,
                created_at=t_end,
            ))
            idx_global += 1
            if max_lines and len(lines) >= max_lines:
                return lines
    return lines
