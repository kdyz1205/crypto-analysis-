"""Append-only JSONL feedback store. Single file per project.

Reading is bulk (load all rows, parse). Writing is one event per call,
flushed immediately so a crash mid-session never loses data.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Iterator, Union

from .schemas import (
    CorrectedTrendline, SignalAccepted, SignalRejected, parse_feedback_line,
)


FeedbackRecord = Union[CorrectedTrendline, SignalAccepted, SignalRejected]


class FeedbackStore:
    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: FeedbackRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")

    def append_many(self, events: Iterable[FeedbackRecord]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            for e in events:
                fh.write(e.model_dump_json() + "\n")

    def __iter__(self) -> Iterator[FeedbackRecord]:
        if not self.log_path.exists():
            return
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield parse_feedback_line(json.loads(line))

    def count(self) -> int:
        if not self.log_path.exists():
            return 0
        n = 0
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n

    def count_by_type(self) -> dict[str, int]:
        d = {"corrected_trendline": 0, "signal_accepted": 0, "signal_rejected": 0}
        for e in self:
            d[e.event_type] = d.get(e.event_type, 0) + 1
        return d

    def corrected_trendlines(self) -> list[CorrectedTrendline]:
        return [e for e in self if isinstance(e, CorrectedTrendline)]
