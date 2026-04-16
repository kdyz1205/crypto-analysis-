"""Agent state — persisted to disk, survives restarts."""

from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import dataclass, field, asdict

STATE_PATH = Path(__file__).parent.parent / "data" / "agent_state.json"


@dataclass
class AgentState:
    current_generation: int = 0
    last_run_at: float = 0.0
    current_job_id: str = ""
    recent_strategy_ids: list[str] = field(default_factory=list)
    recent_result_ids: list[str] = field(default_factory=list)
    recent_top_entries: list[dict] = field(default_factory=list)
    last_error: str = ""
    worker_status: str = "idle"  # idle | running | error | stopped
    total_strategies_generated: int = 0
    total_results_produced: int = 0
    total_profitable: int = 0

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls) -> "AgentState":
        if STATE_PATH.exists():
            try:
                return cls(**json.loads(STATE_PATH.read_text(encoding="utf-8")))
            except Exception:
                pass
        return cls()

    def begin_generation(self, gen: int, job_id: str):
        self.current_generation = gen
        self.current_job_id = job_id
        self.worker_status = "running"
        self.last_run_at = time.time()
        self.last_error = ""
        self.save()

    def end_generation(self, strategy_ids: list[str], result_ids: list[str], top: list[dict],
                       profitable: int):
        self.recent_strategy_ids = strategy_ids[-20:]
        self.recent_result_ids = result_ids[-20:]
        self.recent_top_entries = top[:5]
        self.total_strategies_generated += len(strategy_ids)
        self.total_results_produced += len(result_ids)
        self.total_profitable += profitable  # same dimension as results_produced
        self.worker_status = "idle"
        self.save()

    def record_error(self, error: str):
        self.last_error = error
        self.worker_status = "error"
        self.save()
