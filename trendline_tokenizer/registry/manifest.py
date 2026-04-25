"""Artifact manifest: every saved model carries its versions."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator


class ArtifactManifest(BaseModel):
    artifact_name: str
    model_kind: str               # "fusion" | "vqvae" | "rule_tokenizer"
    model_version: str            # e.g. "fusion.v0.1"
    tokenizer_version: str        # e.g. "rule.v1+vqvae.v0+raw.v1"
    training_dataset_version: str
    feature_norm_version: str
    ckpt_path: str
    config_path: str
    metrics: dict[str, Any]
    created_at: int               # unix seconds

    @field_validator("model_version", "tokenizer_version", "training_dataset_version")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("version field must not be empty")
        return v

    def save(self, path: Path | str):
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ArtifactManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
