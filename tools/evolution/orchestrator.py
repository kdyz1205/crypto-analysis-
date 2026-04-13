"""Evolution orchestrator: baseline → reflect → architect → evaluate → judge → repeat.

Inspired by NousResearch/hermes-agent-self-evolution (GEPA-style reflection).
Key insight: mutation is TRACE-INFORMED, not random. The reflection agent
reads execution traces and identifies failure clusters, then the architect
proposes targeted variants addressing those clusters.

This module only defines the control flow and state. The actual reflection
and architect steps are spawned as Claude subagents via spawn_reflection() /
spawn_architect() — those functions write requests to a queue file that an
external driver (or a Claude Code session) processes, then writes responses
back. Keeps this module free of LLM API coupling.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator
import json
import os
import tempfile
import time


EVOLUTION_DIR = Path(__file__).parent.parent.parent / "data" / "evolution"
ROUNDS_DIR = EVOLUTION_DIR / "rounds"
STATE_PATH = EVOLUTION_DIR / "state.json"
LOCK_PATH = EVOLUTION_DIR / "state.lock"


@contextmanager
def _state_lock(timeout_s: float = 30.0) -> Iterator[None]:
    """File-based mutex. Prevents concurrent writers from clobbering state.

    Portable across OS (no fcntl). Uses O_CREAT|O_EXCL on a lock file.
    """
    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fd = None
    while True:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"state.lock held > {timeout_s}s")
            time.sleep(0.1)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        try:
            os.close(fd)
        finally:
            try:
                os.unlink(LOCK_PATH)
            except FileNotFoundError:
                pass


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


@dataclass(slots=True)
class RoundRecord:
    round_id: int
    baseline_variant: str          # what this round starts from
    candidate_variants: list[str]  # produced by architect
    winner: str | None = None
    winner_fitness_train: float = 0.0
    winner_fitness_test: float = 0.0
    baseline_fitness_train: float = 0.0
    baseline_fitness_test: float = 0.0
    improvement_pct: float = 0.0
    reflection_summary: str = ""
    architect_notes: str = ""
    stopped: bool = False
    stop_reason: str = ""


@dataclass(slots=True)
class EvolutionState:
    current_baseline: str = "v0_baseline"
    current_fitness_train: float = 0.0
    current_fitness_test: float = 0.0
    completed_rounds: list[RoundRecord] = field(default_factory=list)
    no_improvement_streak: int = 0
    max_rounds: int = 15
    stop_streak_threshold: int = 3
    min_improvement_pct: float = 10.0  # Hermes-style 10% gate

    def to_dict(self) -> dict:
        d = asdict(self)
        d["completed_rounds"] = [asdict(r) for r in self.completed_rounds]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EvolutionState":
        rounds = [RoundRecord(**r) for r in d.get("completed_rounds", [])]
        s = cls(
            current_baseline=d.get("current_baseline", "v0_baseline"),
            current_fitness_train=d.get("current_fitness_train", 0.0),
            current_fitness_test=d.get("current_fitness_test", 0.0),
            no_improvement_streak=d.get("no_improvement_streak", 0),
            max_rounds=d.get("max_rounds", 15),
            stop_streak_threshold=d.get("stop_streak_threshold", 3),
            min_improvement_pct=d.get("min_improvement_pct", 10.0),
        )
        s.completed_rounds = rounds
        return s


def load_state() -> EvolutionState:
    """Read state. Does NOT acquire the lock — reading a partial file is
    tolerated because writes are atomic-rename; a reader either sees the
    previous fully-valid version or the new fully-valid version, never a half.
    """
    if not STATE_PATH.exists():
        return EvolutionState()
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            return EvolutionState.from_dict(json.load(f))
    except (json.JSONDecodeError, OSError) as e:
        # Deliberate: load failure must NOT be silently swallowed — that's
        # what caused the "generation jumps" bug. Re-raise so orchestrator
        # surfaces it instead of starting from a blank slate.
        raise RuntimeError(f"state.json corrupt: {e}") from e


def save_state(state: EvolutionState) -> None:
    """Write state atomically under the file lock."""
    with _state_lock():
        _atomic_write_json(STATE_PATH, state.to_dict())


def round_dir(round_id: int) -> Path:
    p = ROUNDS_DIR / f"round_{round_id:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def should_stop(state: EvolutionState) -> tuple[bool, str]:
    if len(state.completed_rounds) >= state.max_rounds:
        return True, f"reached max_rounds ({state.max_rounds})"
    if state.no_improvement_streak >= state.stop_streak_threshold:
        return True, f"no improvement for {state.no_improvement_streak} rounds"
    return False, ""


def record_round(state: EvolutionState, record: RoundRecord) -> None:
    """Mutate state + persist under lock. Atomically reads latest on-disk
    state first so a concurrent worker's update is not clobbered.
    """
    with _state_lock():
        # Re-read latest under lock
        if STATE_PATH.exists():
            try:
                with STATE_PATH.open("r", encoding="utf-8") as f:
                    fresh = EvolutionState.from_dict(json.load(f))
                # Merge: keep whichever has more rounds completed
                if len(fresh.completed_rounds) > len(state.completed_rounds):
                    state = fresh
            except (json.JSONDecodeError, OSError):
                pass  # use in-memory state
        state.completed_rounds.append(record)
        if record.winner and record.improvement_pct >= state.min_improvement_pct:
            state.current_baseline = record.winner
            state.current_fitness_train = record.winner_fitness_train
            state.current_fitness_test = record.winner_fitness_test
            state.no_improvement_streak = 0
        else:
            state.no_improvement_streak += 1
        _atomic_write_json(STATE_PATH, state.to_dict())
