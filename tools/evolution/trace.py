"""Execution trace: per-SETUP records (not per-line).

A line can produce multiple setups (one per touch >= 2, each awaiting the
next touch). Each setup is backtested independently. The reflection agent
reads these records to find failure clusters — especially "does the edge
vary by touch_number" (the user's core demand: "3 touch 进场 vs 4 touch
进场 哪个更赚").

Bug fix history:
- Before: LineTrace (one per line). Broke when multi-setup was added
  because the evaluator used sequential iterator matching, which misaligned
  the moment a line produced more than 1 fade. ALL trace data was
  corrupted from the second line onwards.
- Now: SetupTrace (one per setup), correlated by (line_id, setup_touch_number)
  carried on the Trade object itself. Deterministic alignment, no guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
from collections import defaultdict
import json
from pathlib import Path


@dataclass(slots=True)
class SetupTrace:
    """One record = one backtested setup (fade + optional flip).

    Multiple SetupTraces share the same `line_id` if they come from the
    same detected line. They're distinguished by `setup_touch_number`
    (the touch count the setup was waiting for).
    """
    # Identity
    variant: str
    symbol: str
    timeframe: str
    split: str                    # "train" | "test"
    line_id: int                  # stable index of source line in variant output
    setup_touch_number: int       # the touch being awaited (2 = "enter after touch 1 wait for touch 2", etc.)

    # Line spec (features the reflection agent analyzes)
    side: str                     # "support" | "resistance"
    span_bars: int
    span_pct_of_available: float
    total_touch_count: int        # full touch_count of the source line
    slope_pct_per_bar: float
    anchor_prominence: float
    anchor_bar: int               # bar index we "entered from" (touch_number - 1)
    total_bars: int
    atr_at_anchor: float
    price_at_anchor: float
    vol_regime: str               # "low" | "normal" | "high"

    # Outcome: fade leg
    fade_triggered: bool
    fade_result: str              # "target" | "stop" | "timeout" | "not_triggered"
    fade_R: float
    fade_bars_to_trigger: int = -1
    fade_bars_held: int = -1

    # Outcome: flip leg
    flip_triggered: bool = False
    flip_result: str = "none"
    flip_R: float = 0.0
    flip_bars_held: int = -1

    total_R: float = 0.0          # fade_R + flip_R

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_traces_jsonl(path: Path, traces: list[SetupTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), default=str) + "\n")


def read_traces_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ──────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────
def _rows_of(traces: list[SetupTrace] | list[dict]) -> list[dict]:
    if not traces:
        return []
    if isinstance(traces[0], SetupTrace):
        return [t.to_dict() for t in traces]
    return list(traces)  # type: ignore[return-value]


def _aggregate(rows: list[dict]) -> dict[str, Any]:
    """Aggregate a row set into overall metrics.

    Treats fade-only trades (no flip) as "1 leg".
    """
    n_setups = len(rows)
    triggered = [r for r in rows if r["fade_triggered"]]
    n_trig = len(triggered)
    if n_setups == 0 or n_trig == 0:
        return {
            "n_setups": n_setups,
            "n_setups_triggered": n_trig,
            "trigger_rate": 0.0,
            "fade_win_rate": 0.0,
            "avg_total_R": 0.0,
            "total_R": 0.0,
            "n_flip": 0,
            "flip_win_rate": 0.0,
            "avg_fade_R": 0.0,
        }

    fade_wins = sum(1 for r in triggered if r["fade_R"] > 0)
    flip_rows = [r for r in triggered if r["flip_triggered"]]
    flip_wins = sum(1 for r in flip_rows if r["flip_R"] > 0)
    total_R = sum(r["total_R"] for r in triggered)
    total_fade_R = sum(r["fade_R"] for r in triggered)

    return {
        "n_setups": n_setups,
        "n_setups_triggered": n_trig,
        "trigger_rate": round(n_trig / n_setups, 4),
        "fade_win_rate": round(fade_wins / n_trig, 4),
        "avg_total_R": round(total_R / n_trig, 4),
        "avg_fade_R": round(total_fade_R / n_trig, 4),
        "total_R": round(total_R, 4),
        "n_flip": len(flip_rows),
        "flip_win_rate": round(flip_wins / len(flip_rows), 4) if flip_rows else 0.0,
    }


def summarize_traces(traces: list[SetupTrace] | list[dict]) -> dict[str, Any]:
    """Overall summary PLUS per-touch-number breakdown.

    The per_setup_touch slice is the user's core demand — it answers
    "does the edge change as we wait for more touches?"
    """
    rows = _rows_of(traces)
    if not rows:
        return {"n_setups": 0}

    overall = _aggregate(rows)

    # By touch number
    by_touch: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_touch[r["setup_touch_number"]].append(r)

    by_setup_touch: dict[int, dict] = {}
    for k in sorted(by_touch.keys()):
        by_setup_touch[k] = _aggregate(by_touch[k])

    # By volatility regime
    by_vol: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_vol[r["vol_regime"]].append(r)
    by_vol_regime = {k: _aggregate(v) for k, v in by_vol.items()}

    # By side
    by_side: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_side[r["side"]].append(r)
    by_side_agg = {k: _aggregate(v) for k, v in by_side.items()}

    return {
        **overall,
        "by_setup_touch": by_setup_touch,
        "by_vol_regime": by_vol_regime,
        "by_side": by_side_agg,
    }
