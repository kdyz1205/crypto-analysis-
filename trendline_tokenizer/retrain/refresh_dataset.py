"""Rebuild the training pool from the legacy auto rows + accepted
corrected lines + rejected-signal-aware filtering.

Output: a JSONL of TrendlineRecord ready to feed into train_fusion's
record loader.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

from ..adapters import iter_legacy_pattern_records
from ..schemas.trendline import TrendlineRecord
from ..feedback.schemas import CorrectedTrendline
from ..feedback.store import FeedbackStore


def collect_records(
    *,
    patterns_dir: Path,
    feedback_path: Path | None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    max_legacy_per_pair: int | None = None,
) -> list[TrendlineRecord]:
    """Build the training pool: auto rows first, then corrected lines
    (override / augment). Corrected lines mark `label_source='manual'`
    so the trainer can up-weight them later if desired."""
    out: list[TrendlineRecord] = []
    legacy_seen_ids: set[str] = set()

    if symbols and timeframes:
        for s in symbols:
            for tf in timeframes:
                f = patterns_dir / f"{s.upper()}_{tf}.jsonl"
                if not f.exists():
                    continue
                count = 0
                for rec in iter_legacy_pattern_records(f):
                    out.append(rec)
                    legacy_seen_ids.add(rec.id)
                    count += 1
                    if max_legacy_per_pair and count >= max_legacy_per_pair:
                        break
    else:
        for f in patterns_dir.glob("*.jsonl"):
            count = 0
            for rec in iter_legacy_pattern_records(f):
                out.append(rec)
                legacy_seen_ids.add(rec.id)
                count += 1
                if max_legacy_per_pair and count >= max_legacy_per_pair:
                    break

    if feedback_path is not None and feedback_path.exists():
        store = FeedbackStore(feedback_path)
        for ev in store:
            if isinstance(ev, CorrectedTrendline):
                rec = ev.corrected
                # If user corrected an existing auto line, replace it.
                if ev.original_id is not None:
                    out = [r for r in out if r.id != ev.original_id]
                out.append(rec)

    return out


def write_pool_jsonl(records: Iterable[TrendlineRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
            n += 1
    return n
