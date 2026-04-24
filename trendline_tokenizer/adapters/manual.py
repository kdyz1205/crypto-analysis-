"""Adapter: data/manual_trendlines.json → TrendlineRecord list.

The legacy manual JSON lacks several fields (start_bar_index,
touch_count, outcomes). Those are left as None — fine for the rule
tokeniser, which substitutes neutral values, and for using manual data
purely as a gold set on geometry + role.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..schemas.trendline import TrendlineRecord, LineRole


_SIDE_TO_ROLE: dict[str, LineRole] = {
    "support": "support",
    "resistance": "resistance",
    "channel_upper": "channel_upper",
    "channel_lower": "channel_lower",
    "wedge_side": "wedge_side",
    "triangle_side": "triangle_side",
}


def _direction_from_prices(p_start: float, p_end: float) -> str:
    if p_end > p_start * 1.001:
        return "up"
    if p_end < p_start * 0.999:
        return "down"
    return "flat"


def load_manual_records(path: str | Path) -> list[TrendlineRecord]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Allow {"drawings": [...]} or {"lines": [...]} wrappers.
        for k in ("drawings", "lines", "items"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
    if not isinstance(data, list):
        return []
    out: list[TrendlineRecord] = []
    for row in data:
        try:
            out.append(_row_to_record(row))
        except Exception as exc:
            print(f"[adapter.manual] skip {row.get('manual_line_id')}: {exc}")
    return out


def _row_to_record(row: dict) -> TrendlineRecord:
    side = str(row.get("side") or "unknown")
    role: LineRole = _SIDE_TO_ROLE.get(side, "unknown")   # type: ignore[assignment]
    p_start = float(row["price_start"])
    p_end = float(row["price_end"])
    t_start = int(row["t_start"])
    t_end = int(row["t_end"])
    return TrendlineRecord(
        id=str(row["manual_line_id"]),
        symbol=str(row["symbol"]).upper(),
        exchange="bitget",
        timeframe=str(row["timeframe"]),
        start_time=t_start,
        end_time=t_end,
        start_bar_index=0,
        end_bar_index=max(1, _bars_between(t_start, t_end, row["timeframe"])),
        start_price=p_start,
        end_price=p_end,
        extend_left=bool(row.get("extend_left", False)),
        extend_right=bool(row.get("extend_right", False)),
        line_role=role,
        direction=_direction_from_prices(p_start, p_end),
        touch_count=0,                            # not tracked in legacy manual schema
        label_source="manual",
        confidence=None,
        created_at=int(row.get("created_at") or t_start),
        notes=str(row.get("notes") or ""),
    )


_TF_SEC = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
    "1d": 86400, "1w": 7 * 86400,
}


def _bars_between(t_start: int, t_end: int, tf: str) -> int:
    sec = _TF_SEC.get(tf, 3600)
    return max(1, int((t_end - t_start) // sec))


def iter_manual_records(path: str | Path) -> Iterable[TrendlineRecord]:
    for r in load_manual_records(path):
        yield r
