"""Retrain trigger: simple threshold check + dispatch to train_fusion.

Policy: if the feedback store has >= MIN_NEW_FEEDBACK rows since the
last manifest's `created_at`, request a retrain. The actual training
happens in a subprocess so this module stays import-light.

This is a polling helper; a cron / file-watcher schedules the calls.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from ..feedback.store import FeedbackStore
from ..registry.manifest import ArtifactManifest
from ..registry.paths import latest_fusion


MIN_NEW_FEEDBACK = 20


def count_new_feedback_since(store: FeedbackStore, since_ts: int) -> int:
    n = 0
    for ev in store:
        if getattr(ev, "created_at", 0) >= since_ts:
            n += 1
    return n


def should_retrain(
    *,
    feedback_path: Path,
    last_manifest: Optional[ArtifactManifest],
    min_new: int = MIN_NEW_FEEDBACK,
) -> tuple[bool, int]:
    """Return (should_retrain, n_new_feedback)."""
    if not feedback_path.exists():
        return False, 0
    store = FeedbackStore(feedback_path)
    since = last_manifest.created_at if last_manifest else 0
    n_new = count_new_feedback_since(store, since)
    return n_new >= min_new, n_new


def latest_fusion_manifest(model_version: str) -> Optional[ArtifactManifest]:
    d = latest_fusion(model_version)
    if d is None:
        return None
    mp = d / "manifest.json"
    return ArtifactManifest.load(mp) if mp.exists() else None


def kick_retrain(
    *,
    symbols: list[str],
    timeframes: list[str],
    max_records: int = 5000,
    epochs: int = 1,
    batch_size: int = 32,
    blocking: bool = False,
) -> subprocess.Popen | int:
    """Spawn `train_fusion` in a subprocess. Returns the Popen handle if
    non-blocking, or the exit code if blocking."""
    cmd = [
        sys.executable, "-m", "trendline_tokenizer.training.train_fusion",
        "--symbols", *symbols,
        "--timeframes", *timeframes,
        "--max-records", str(max_records),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    if blocking:
        return subprocess.call(cmd)
    return subprocess.Popen(cmd)
