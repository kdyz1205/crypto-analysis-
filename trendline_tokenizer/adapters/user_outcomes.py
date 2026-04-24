"""Adapter: user's real trade outcomes → supervised labels joined onto
manual TrendlineRecords.

Files consumed:
  data/user_drawing_outcomes.jsonl  — 1,393 rows, per-config realised PnL
                                       across multiple buffer/rr configs per line
  data/user_drawing_labels.jsonl    — 47 rows, per-line BEST config + label_trade_win
  data/user_drawings_ml.jsonl       — 430 rows, per-line event stream with
                                       derived geometry features

Join key: manual_line_id. One line can have many outcome rows (one per
config tried); we aggregate them to a single label record:
  - label_trade_win           bool  — ever profitable under the best config
  - best_realized_r           float — highest realized_r across configs
  - best_config               dict  — the config that produced it
  - mfe_r_max, mae_r_min      float — envelope of excursion
  - n_configs_tried           int   — diversity of tested setups

These labels are richer and more reliable than the legacy pattern
rows' weak bounce/break flags.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from ..schemas.trendline import TrendlineRecord


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_line_outcomes(outcomes_path: Path) -> dict[str, dict]:
    """line_id → aggregated outcome record."""
    by_line: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(outcomes_path):
        mid = row.get("manual_line_id")
        if mid:
            by_line[mid].append(row)

    agg: dict[str, dict] = {}
    for line_id, rows in by_line.items():
        filled = [r for r in rows if r.get("filled")]
        if not filled:
            agg[line_id] = {
                "n_configs_tried": len(rows),
                "ever_filled": False,
            }
            continue
        # best config = highest realized_r
        best = max(filled, key=lambda r: float(r.get("realized_r") or 0.0))
        all_r = [float(r.get("realized_r") or 0.0) for r in filled]
        mfe = [float(r.get("mfe_r") or 0.0) for r in filled]
        mae = [float(r.get("mae_r") or 0.0) for r in filled]
        exit_reasons = [r.get("exit_reason") for r in filled]
        agg[line_id] = {
            "n_configs_tried": len(rows),
            "ever_filled": True,
            "ever_won": any(float(r.get("realized_r") or 0.0) > 0 for r in filled),
            "best_realized_r": max(all_r),
            "worst_realized_r": min(all_r),
            "mean_realized_r": sum(all_r) / max(1, len(all_r)),
            "mfe_r_max": max(mfe) if mfe else 0.0,
            "mae_r_min": min(mae) if mae else 0.0,
            "best_config": best.get("config") or {},
            "best_exit_reason": best.get("exit_reason"),
            "best_minutes_held": float(best.get("minutes_held") or 0.0),
            "best_direction": best.get("direction"),
            "exit_reason_mix": {er: exit_reasons.count(er) for er in set(exit_reasons)},
        }
    return agg


def load_line_labels(labels_path: Path) -> dict[str, dict]:
    """line_id → human-judged best-config label record."""
    out: dict[str, dict] = {}
    for row in _read_jsonl(labels_path):
        mid = row.get("manual_line_id")
        if mid:
            out[mid] = row
    return out


def load_line_ml_events(ml_path: Path) -> dict[str, dict]:
    """line_id → last ML event (has derived features). When a line has
    multiple events keep the latest."""
    latest: dict[str, dict] = {}
    latest_ts: dict[str, int] = {}
    for row in _read_jsonl(ml_path):
        mid = row.get("manual_line_id")
        if not mid:
            # user_drawings_ml uses symbol+t_start+t_end as implicit id
            mid_derived = f"manual-{row.get('symbol','?')}-{row.get('timeframe','?')}-{row.get('side','?')}-{row.get('t_start','?')}-{row.get('t_end','?')}"
            mid = mid_derived
        ts = int(row.get("ts") or 0)
        if mid not in latest or ts > latest_ts.get(mid, 0):
            latest[mid] = row
            latest_ts[mid] = ts
    return latest


def enrich_record_with_outcomes(
    record: TrendlineRecord,
    outcomes_agg: dict[str, dict],
    labels: dict[str, dict],
    ml_events: dict[str, dict],
) -> TrendlineRecord:
    """Return a NEW TrendlineRecord with outcome-derived fields set."""
    line_id = record.id
    outc = outcomes_agg.get(line_id)
    label = labels.get(line_id)
    ml = ml_events.get(line_id)

    # Start from the record's fields
    fields = record.model_dump()

    # touch / bounce / break from outcomes
    if outc and outc.get("ever_filled"):
        # ever_won ≈ "bounced" in the sense: the line's setup produced a positive R
        # If best_realized_r > 0 under ANY config, we tag bounce_after = True.
        ever_won = bool(outc.get("ever_won"))
        best_r = float(outc.get("best_realized_r") or 0.0)
        mfe = float(outc.get("mfe_r_max") or 0.0)
        mae = float(outc.get("mae_r_min") or 0.0)
        best_reason = outc.get("best_exit_reason") or ""

        fields["bounce_after"] = ever_won
        fields["bounce_strength_atr"] = max(0.0, mfe) if ever_won else None
        # exit_reason == "stop" means the line broke before the target.
        fields["break_after"] = best_reason == "stop"
        fields["break_distance_atr"] = abs(mae) if best_reason == "stop" else None
        fields["retested_after_break"] = False  # not tracked in legacy labels
        fields["rejection_strength_atr"] = mfe  # max excursion in favor as rejection proxy

        # promote to "auto_approved" if we have outcome evidence
        if fields["label_source"] == "manual":
            fields["label_source"] = "manual"
        fields["confidence"] = (1.0 if (label and label.get("label_trade_win")) else
                                0.6 if ever_won else 0.3)
        # store the winning config as notes
        cfg = outc.get("best_config") or {}
        if cfg:
            note_parts = [f"best_config: buffer={cfg.get('buffer_pct')}, rr={cfg.get('rr')}"]
            note_parts.append(f"best_r={best_r:.2f}")
            note_parts.append(f"n_configs={outc.get('n_configs_tried')}")
            fields["notes"] = " | ".join(note_parts)

    # derived slope / geometry from ml events
    if ml:
        slope = ml.get("slope_per_bar") or ml.get("slope")
        # already captured in the record geometry; only used for sanity

    return TrendlineRecord(**fields)


def enrich_records_with_outcomes(
    records: list[TrendlineRecord],
    *,
    outcomes_path: Path,
    labels_path: Path,
    ml_path: Optional[Path] = None,
) -> list[TrendlineRecord]:
    outc = load_line_outcomes(outcomes_path)
    lbls = load_line_labels(labels_path)
    ml = load_line_ml_events(ml_path) if ml_path else {}
    return [enrich_record_with_outcomes(r, outc, lbls, ml) for r in records]


def outcomes_coverage_report(
    records: list[TrendlineRecord],
    outcomes_path: Path,
    labels_path: Path,
) -> dict:
    outc = load_line_outcomes(outcomes_path)
    lbls = load_line_labels(labels_path)
    ids = {r.id for r in records}
    with_outcome = sum(1 for i in ids if i in outc)
    with_label = sum(1 for i in ids if i in lbls)
    filled = sum(1 for i in ids if outc.get(i, {}).get("ever_filled"))
    won = sum(1 for i in ids if outc.get(i, {}).get("ever_won"))
    return {
        "n_records": len(records),
        "n_unique_ids": len(ids),
        "n_outcomes_join": with_outcome,
        "n_labels_join": with_label,
        "n_ever_filled": filled,
        "n_ever_won": won,
        "total_outcome_rows": sum(v.get("n_configs_tried", 0) for v in outc.values()),
    }
