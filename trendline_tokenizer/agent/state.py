"""Persistent agent state — what we've seen, when we last ran, etc.

Survives restarts. Stored as JSON at data/agent_state.json so the next
run picks up where the previous one left off.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from ..registry.paths import ROOT


DEFAULT_STATE_PATH = ROOT / "data" / "agent_state.json"


@dataclass
class AgentState:
    iteration: int = 0
    last_run_ts: int = 0
    last_train_ts: int = 0
    last_train_artifact: str = ""
    last_seen_outcome_ts: int = 0
    last_seen_manual_count: int = 0
    n_lines_auto_drawn_total: int = 0
    n_retrain_triggered: int = 0
    log: list[dict] = field(default_factory=list)

    def save(self, path: Path | str = DEFAULT_STATE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = DEFAULT_STATE_PATH) -> "AgentState":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(**data)
        except Exception:
            return cls()

    def record_iteration(self, payload: dict):
        self.log.append({"ts": int(time.time()), "iteration": self.iteration, **payload})
        # Keep last 200 iterations only
        if len(self.log) > 200:
            self.log = self.log[-200:]
