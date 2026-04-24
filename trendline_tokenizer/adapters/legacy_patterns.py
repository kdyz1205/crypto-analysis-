"""Adapter: data/patterns/*.jsonl → TrendlineRecord.

Legacy row shape (from sr_patterns.detect_patterns output, 2026 rev.):

    {
      "pattern_id": "41722395e8fd",
      "features": {
        "slope_atr": 0.0752, "length_bars": 22, "volatility": 0.0195,
        "trend_context": "downtrend", "side": "support", "rsi": 89.4,
        "ma_distance_atr": -0.54, "touch_quality": 1.0,
        "symbol": "BTCUSDT", "timeframe": "4h"
      },
      "outcome": {
        "third_touch": true, "bounced": true,
        "bounce_magnitude_atr": 5.3, "broke": true, "break_bars_later": 9,
        "fake_break": true, "max_return_atr": 5.3, "max_drawdown_atr": 8.14
      },
      "anchor1_idx": 35, "anchor2_idx": 57,
      "anchor1_price": 82228.0, "anchor2_price": 85042.8,
      "detected_at_bar": 57, "split_bucket": "train", "time_position": 0.0228
    }

Mapping rules:
- features.side → line_role   (support / resistance / …; unknown otherwise)
- anchor1/2_idx → start_bar_index / end_bar_index
- anchor1/2_price → start_price / end_price
- outcome.third_touch / bounced → touch_count, bounce_after
- outcome.broke → break_after
- features.volatility → volatility_atr_pct (already in ATR/price units)
- outcome.bounce_magnitude_atr → bounce_strength_atr
- outcome.max_return_atr / outcome.max_drawdown_atr → break_distance_atr (signed)
- start_time / end_time / created_at: unknown here (legacy format drops
  wall-clock anchors). Use a deterministic placeholder `detected_at_bar`
  so ids stay stable; downstream code that needs wall-clock joins with
  the OHLCV CSV on bar_index.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ..schemas.trendline import TrendlineRecord, LineRole


_SIDE_TO_ROLE: dict[str, LineRole] = {
    "support": "support",
    "resistance": "resistance",
}


def _role_from_features(feats: dict) -> LineRole:
    s = str(feats.get("side") or "").lower()
    if s in _SIDE_TO_ROLE:
        return _SIDE_TO_ROLE[s]   # type: ignore[return-value]
    # fallback heuristics for channel/wedge/triangle variants if present
    if "channel_upper" in s:
        return "channel_upper"
    if "channel_lower" in s:
        return "channel_lower"
    if "wedge" in s:
        return "wedge_side"
    if "triangle" in s:
        return "triangle_side"
    return "unknown"


def _direction_from_prices(p1: float, p2: float) -> str:
    if p2 > p1 * 1.001:
        return "up"
    if p2 < p1 * 0.999:
        return "down"
    return "flat"


def _touch_count_from_outcome(outcome: dict) -> int:
    # Legacy only records "third_touch" (bool). Infer minimal count:
    # anchors themselves count as touches 1 and 2; third_touch adds one.
    base = 2
    if outcome.get("third_touch"):
        base += 1
    return base


def _row_to_record(row: dict) -> TrendlineRecord:
    feats = row.get("features") or {}
    outcome = row.get("outcome") or {}
    symbol = str(feats.get("symbol") or "UNKNOWN").upper()
    tf = str(feats.get("timeframe") or "1h")
    a1_idx = int(row.get("anchor1_idx") or 0)
    a2_idx = int(row.get("anchor2_idx") or max(1, a1_idx + 1))
    a1_price = float(row.get("anchor1_price") or 0.0)
    a2_price = float(row.get("anchor2_price") or a1_price)
    role = _role_from_features(feats)
    direction = _direction_from_prices(a1_price, a2_price)

    # break_distance: use signed max_return_atr if broke, else 0
    broke = bool(outcome.get("broke"))
    if broke:
        break_dist = float(outcome.get("max_return_atr") or 0.0)
    else:
        break_dist = 0.0

    pid = str(row.get("pattern_id") or hashlib.md5(json.dumps(row, sort_keys=True).encode()).hexdigest()[:12])

    return TrendlineRecord(
        id=f"legacy-{pid}",
        symbol=symbol,
        exchange="bitget",
        timeframe=tf,
        # Legacy rows lack wall-clock times. Synthesise from bar indices
        # + a symbolic epoch origin so ids are deterministic. Downstream
        # code that needs real t must join with OHLCV on bar index.
        start_time=int(a1_idx),
        end_time=int(a2_idx),
        start_bar_index=a1_idx,
        end_bar_index=a2_idx,
        start_price=a1_price,
        end_price=a2_price,
        line_role=role,
        direction=direction,
        touch_count=_touch_count_from_outcome(outcome),
        rejection_strength_atr=float(outcome.get("bounce_magnitude_atr") or 0.0) if outcome.get("bounced") else 0.0,
        bounce_after=bool(outcome.get("bounced")),
        bounce_strength_atr=float(outcome.get("bounce_magnitude_atr") or 0.0) if outcome.get("bounced") else None,
        break_after=broke,
        break_distance_atr=break_dist if broke else None,
        retested_after_break=bool(outcome.get("fake_break")),
        volatility_atr_pct=float(feats.get("volatility") or 0.0) or None,
        volume_z_score=None,  # not in legacy schema
        distance_to_ma20_atr=float(feats.get("ma_distance_atr")) if feats.get("ma_distance_atr") is not None else None,
        distance_to_recent_high_atr=None,
        distance_to_recent_low_atr=None,
        label_source="auto",
        auto_method="sr_patterns.legacy_jsonl",
        score=float(feats.get("touch_quality")) if feats.get("touch_quality") is not None else None,
        quality_flags=[],
        created_at=int(a2_idx),
        notes=f"split={row.get('split_bucket', '?')}",
    )


def iter_legacy_pattern_records(path: str | Path) -> Iterable[TrendlineRecord]:
    """Yield TrendlineRecord per line in a single legacy JSONL file."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for ln_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                yield _row_to_record(row)
            except Exception as exc:
                print(f"[adapter.legacy_patterns] {p}:{ln_num} skip: {exc}")


def load_legacy_pattern_records(root: str | Path, *, limit: int | None = None) -> list[TrendlineRecord]:
    """Load TrendlineRecord list from a directory of legacy JSONL files."""
    root_p = Path(root)
    out: list[TrendlineRecord] = []
    files = sorted(root_p.glob("*.jsonl"))
    for f in files:
        for rec in iter_legacy_pattern_records(f):
            out.append(rec)
            if limit is not None and len(out) >= limit:
                return out
    return out
