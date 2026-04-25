"""Iteration report — what the agent did this round.

Written to data/agent_reports.jsonl. Each line is a complete iteration
summary: how much new data we saw, whether we retrained, how many
auto-drawn lines we generated, and headline backtest metrics.

Designed to be tail-able from the UI or a TG bot.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from ..registry.paths import ROOT


DEFAULT_REPORT_PATH = ROOT / "data" / "trendline_agent_reports.jsonl"


@dataclass
class IterationReport:
    iteration: int
    started_at: int
    finished_at: int = 0
    duration_s: float = 0.0
    # data deltas
    n_new_outcomes: int = 0
    n_new_manual: int = 0
    n_new_feedback: int = 0
    # actions
    retrained: bool = False
    new_artifact: str = ""
    n_lines_auto_drawn: int = 0
    n_symbols_processed: int = 0
    # quality (per backtest run)
    backtest_summary: dict = field(default_factory=dict)
    # errors
    errors: list[str] = field(default_factory=list)
    notes: str = ""

    def append(self, path: Path | str = DEFAULT_REPORT_PATH):
        if not self.finished_at:
            self.finished_at = int(time.time())
        if not self.duration_s:
            self.duration_s = float(self.finished_at - self.started_at)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(self)) + "\n")

    def as_dict(self) -> dict:
        return asdict(self)


def tail_reports(path: Path | str = DEFAULT_REPORT_PATH, n: int = 20) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = p.read_text(encoding="utf-8").splitlines()
    out = []
    for line in rows[-n:]:
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out
