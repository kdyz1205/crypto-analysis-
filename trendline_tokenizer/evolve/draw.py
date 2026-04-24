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
    patterns = getattr(result, "patterns", None) or getattr(result, "lines", None) or []
    for idx, p in enumerate(patterns):
        # sr_patterns returns PatternResult-like objects; defensively pull fields
        pd_dict = p.__dict__ if hasattr(p, "__dict__") else (p if isinstance(p, dict) else {})
        a1_idx = int(pd_dict.get("anchor1_idx") or pd_dict.get("x1") or pd_dict.get("start_bar") or 0)
        a2_idx = int(pd_dict.get("anchor2_idx") or pd_dict.get("x2") or pd_dict.get("end_bar") or max(1, a1_idx + 1))
        a1_price = float(pd_dict.get("anchor1_price") or pd_dict.get("y1") or 0.0)
        a2_price = float(pd_dict.get("anchor2_price") or pd_dict.get("y2") or a1_price)
        if a2_idx <= a1_idx or a1_price <= 0 or a2_price <= 0:
            continue
        role = _role_from_pattern_result(pd_dict)
        direction = "up" if a2_price > a1_price * 1.001 else ("down" if a2_price < a1_price * 0.999 else "flat")

        t_start = int(df.iloc[a1_idx]["timestamp"]) if "timestamp" in df.columns else int(a1_idx)
        t_end = int(df.iloc[a2_idx]["timestamp"]) if "timestamp" in df.columns else int(a2_idx)

        rid = f"evolve-{symbol}-{timeframe}-{a1_idx}-{a2_idx}-{role}-{idx}"
        line = TrendlineRecord(
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
            touch_count=int(pd_dict.get("touches") or pd_dict.get("touch_count") or 2),
            label_source="auto",
            auto_method=f"sr_patterns.evolve[{','.join(f'{k}={v}' for k,v in sr_params_kwargs.items())}]",
            score=float(pd_dict.get("score") or pd_dict.get("touch_quality") or 0.0) or None,
            created_at=t_end,
        )
        lines.append(line)
        if max_lines and len(lines) >= max_lines:
            break
    return lines
