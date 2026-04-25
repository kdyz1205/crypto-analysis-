"""Paper dispatcher: append-only JSONL log of all signals.

This is the LAST mile of the inference pipeline. It does not place
orders, does not call exchanges, does not touch any live system. It
exists so we can replay later, compute hit rate, and verify the model
is producing decisions worth wiring up.

Live execution is OUT OF SCOPE for milestone 2 by design.
"""
from __future__ import annotations
import json
from pathlib import Path

from .signal_engine import SignalRecord


class PaperDispatcher:
    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def dispatch(self, signal: SignalRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(signal.to_dict()) + "\n")

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        out = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
