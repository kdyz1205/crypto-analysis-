"""Build the training pool from EVERY available source:
  - data/manual_trendlines.json (78 user-drawn gold lines)
  - data/user_drawing_outcomes.jsonl (1393 simulated-PnL rows -> bounce/break labels)
  - data/user_drawing_labels.jsonl (47 best-config win/lose labels)
  - data/patterns/*.jsonl (~271k auto sr_patterns rows)
  - data/feedback/*.jsonl (CorrectedTrendline events from the UI)

Manual records are HEAVY-oversampled (default 50x) so the model is
pulled hard toward the user's drawing style.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Sequence

from ..adapters import iter_legacy_pattern_records
from ..adapters.manual import load_manual_records
from ..adapters.user_outcomes import (
    enrich_records_with_outcomes, outcomes_coverage_report,
)
from ..feedback.schemas import CorrectedTrendline
from ..feedback.store import FeedbackStore
from ..schemas.trendline import TrendlineRecord


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATTERNS = ROOT / "data" / "patterns"
DEFAULT_MANUAL = ROOT / "data" / "manual_trendlines.json"
DEFAULT_OUTCOMES = ROOT / "data" / "user_drawing_outcomes.jsonl"
DEFAULT_LABELS = ROOT / "data" / "user_drawing_labels.jsonl"
DEFAULT_ML = ROOT / "data" / "user_drawings_ml.jsonl"


def collect_records(
    *,
    patterns_dir: Path | None = None,
    manual_path: Path | None = None,
    outcomes_path: Path | None = None,
    labels_path: Path | None = None,
    ml_path: Path | None = None,
    feedback_path: Path | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    max_legacy_per_pair: int | None = None,
    manual_oversample: int = 50,
    include_legacy: bool = True,
) -> tuple[list[TrendlineRecord], dict]:
    """Return (records, stats).

    Order is MANUAL_OVERSAMPLED + auto + corrected, so a `random_split`
    later still distributes the manual gold across train/val.
    """
    patterns_dir = patterns_dir or DEFAULT_PATTERNS
    manual_path = manual_path or DEFAULT_MANUAL
    outcomes_path = outcomes_path or DEFAULT_OUTCOMES
    labels_path = labels_path or DEFAULT_LABELS
    ml_path = ml_path or DEFAULT_ML

    out: list[TrendlineRecord] = []
    stats = {"manual_loaded": 0, "manual_after_enrich": 0,
             "manual_oversample_factor": manual_oversample,
             "manual_total_in_pool": 0,
             "auto_loaded": 0, "feedback_corrected": 0,
             "outcomes_coverage": None}

    # 1. Manual (gold) -> enrich with outcomes -> oversample.
    manual = load_manual_records(manual_path)
    stats["manual_loaded"] = len(manual)
    if manual:
        manual = enrich_records_with_outcomes(
            manual,
            outcomes_path=outcomes_path,
            labels_path=labels_path,
            ml_path=ml_path,
        )
        stats["manual_after_enrich"] = len(manual)
        stats["outcomes_coverage"] = outcomes_coverage_report(
            manual, outcomes_path, labels_path,
        )
    for _ in range(max(1, manual_oversample)):
        out.extend(manual)
    stats["manual_total_in_pool"] = len(manual) * max(1, manual_oversample)

    # 2. Auto patterns.
    if include_legacy and patterns_dir.exists():
        if symbols and timeframes:
            for s in symbols:
                for tf in timeframes:
                    f = patterns_dir / f"{s.upper()}_{tf}.jsonl"
                    if not f.exists():
                        continue
                    count = 0
                    for rec in iter_legacy_pattern_records(f):
                        out.append(rec)
                        stats["auto_loaded"] += 1
                        count += 1
                        if max_legacy_per_pair and count >= max_legacy_per_pair:
                            break
        else:
            for f in patterns_dir.glob("*.jsonl"):
                count = 0
                for rec in iter_legacy_pattern_records(f):
                    out.append(rec)
                    stats["auto_loaded"] += 1
                    count += 1
                    if max_legacy_per_pair and count >= max_legacy_per_pair:
                        break

    # 3. Corrected lines from the feedback store (overrides matching auto by id).
    if feedback_path is not None and feedback_path.exists():
        store = FeedbackStore(feedback_path)
        corrected_ids: set[str] = set()
        for ev in store:
            if isinstance(ev, CorrectedTrendline):
                stats["feedback_corrected"] += 1
                if ev.original_id is not None:
                    corrected_ids.add(ev.original_id)
                out.append(ev.corrected)
        if corrected_ids:
            out = [r for r in out if r.id not in corrected_ids]

    return out, stats


def write_pool_jsonl(records: Iterable[TrendlineRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
            n += 1
    return n
