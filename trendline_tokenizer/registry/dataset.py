"""DatasetManifest — provenance for every training pool.

No training run is allowed to use a pool that doesn't have a manifest.
The manifest captures:
  - dataset_version              (e.g. "trendline-structure-v0.1.0")
  - created_at + git_commit
  - raw sources + sha256 of each
  - symbols, timeframes, time bounds
  - record counts (manual / auto / sweep / outcomes)
  - split policy + bounds (time-forward / symbol-heldout / regime-heldout)
  - labeling policy version

Per the user's institutional-rigor spec: "No training run is valid unless
it writes git_commit, dataset_version, tokenizer_version, model_config,
train/val/test split policy, metrics.json, checkpoint sha256."

This module produces the dataset_version + sha256 artifacts that show up
in ExperimentManifest later.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field


SplitPolicy = Literal[
    "random",                # smoke tests only
    "time_forward",          # train < val_start_ts <= val < test_start_ts <= test
    "symbol_heldout",        # train symbols ∩ test symbols = ∅
    "regime_heldout",        # train regimes != test regime
    "stratified_random",     # random with class balance (only for unit tests)
]


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA256 of a file. Returns hex string.

    For DIRECTORIES, returns sha256 of a sorted listing of
    (relative_path, size_bytes) of every file in the tree. Lets us
    fingerprint a data dir cheaply without hashing every byte.
    """
    h = hashlib.sha256()
    p = Path(path)
    if not p.exists():
        return ""
    if p.is_dir():
        entries: list[str] = []
        for sub in sorted(p.rglob("*")):
            if sub.is_file():
                try:
                    rel = sub.relative_to(p).as_posix()
                    entries.append(f"{rel}\t{sub.stat().st_size}")
                except (PermissionError, OSError):
                    continue
        h.update("\n".join(entries).encode("utf-8"))
        return h.hexdigest()
    try:
        with p.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except (PermissionError, OSError):
        return ""
    return h.hexdigest()


def git_head_commit(repo_root: Path | None = None) -> str:
    """Best-effort current git HEAD sha. Empty string if not in git."""
    cwd = repo_root or Path.cwd()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


class RawSource(BaseModel):
    path: str
    sha256: str = ""
    size_bytes: int = 0
    role: Literal["ohlcv", "manual_lines", "auto_patterns",
                  "patterns_sweep", "user_outcomes", "user_labels",
                  "feedback", "other"] = "other"
    n_records: Optional[int] = None
    notes: str = ""


class SplitSpec(BaseModel):
    policy: SplitPolicy
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    # for time_forward
    val_start_ts: Optional[int] = None
    test_start_ts: Optional[int] = None
    # for symbol_heldout
    train_symbols: Optional[list[str]] = None
    val_symbols: Optional[list[str]] = None
    test_symbols: Optional[list[str]] = None
    # for regime_heldout
    train_regimes: Optional[list[str]] = None
    test_regimes: Optional[list[str]] = None
    notes: str = ""


class DatasetManifest(BaseModel):
    """Versioned dataset descriptor written alongside the dataset."""
    dataset_name: str
    dataset_version: str                                # e.g. "trendline-structure-v0.1.0"
    created_at: int                                     # unix seconds
    git_commit: str = ""
    raw_sources: list[RawSource] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    n_ohlcv_bars: Optional[int] = None
    n_trendline_records_total: int = 0
    n_manual_records: int = 0
    n_auto_records: int = 0
    n_sweep_records: int = 0
    n_outcome_records: int = 0
    labeling_policy: str = "outcome_labeler.v1"
    split_policy: SplitSpec = Field(default_factory=lambda: SplitSpec(policy="random"))
    extras: dict = Field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "DatasetManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def manifest_sha256(self) -> str:
        """SHA256 of the canonical-JSON serialization of THIS manifest.
        Lets ExperimentManifest pin the EXACT manifest used."""
        canon = self.model_dump_json(indent=None)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def build_manifest_from_collect_records(
    *,
    dataset_name: str,
    dataset_version: str,
    raw_paths: dict[str, Path],
    symbols: list[str],
    timeframes: list[str],
    stats: dict,
    split_spec: SplitSpec | None = None,
    repo_root: Path | None = None,
) -> DatasetManifest:
    """Convenience: build a manifest from `collect_records` stats + paths.

    Caller passes the dict it got from refresh_dataset.collect_records.
    """
    sources: list[RawSource] = []
    for role, p in raw_paths.items():
        if not p.exists():
            continue
        sources.append(RawSource(
            path=str(p), sha256=sha256_file(p),
            size_bytes=p.stat().st_size,
            role=role,                          # type: ignore[arg-type]
        ))

    n_manual = stats.get("manual_loaded", 0)
    n_auto = stats.get("auto_loaded", 0)
    n_sweep = stats.get("auto_sweep_loaded", 0)
    n_outcomes = (stats.get("outcomes_coverage") or {}).get("total_outcome_rows", 0)

    return DatasetManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        created_at=int(time.time()),
        git_commit=git_head_commit(repo_root),
        raw_sources=sources,
        symbols=symbols, timeframes=timeframes,
        n_trendline_records_total=n_manual + n_auto + n_sweep,
        n_manual_records=n_manual,
        n_auto_records=n_auto,
        n_sweep_records=n_sweep,
        n_outcome_records=n_outcomes,
        split_policy=split_spec or SplitSpec(policy="random"),
        extras={"collect_records_stats": stats},
    )
