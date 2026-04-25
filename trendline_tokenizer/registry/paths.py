"""Standardised checkpoint paths."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"


def fusion_dir(artifact_name: str) -> Path:
    p = CHECKPOINTS / "fusion" / artifact_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_fusion(model_version: str) -> Path | None:
    base = CHECKPOINTS / "fusion"
    if not base.exists():
        return None
    candidates = [d for d in base.iterdir()
                  if d.is_dir() and d.name.startswith(model_version)]
    if not candidates:
        return None
    return sorted(candidates)[-1]
