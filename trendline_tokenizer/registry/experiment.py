"""ExperimentManifest — full provenance for one training run.

Every training run writes:
    experiments/<run_id>/config.yaml      (or .json — model + training hyperparams)
    experiments/<run_id>/manifest.json    (this — pins commit, dataset, tokenizer, ckpt sha256)
    experiments/<run_id>/metrics.json     (per-epoch + final val/test metrics)
    checkpoints/<run_id>/model.pt         (the actual weights)

ExperimentManifest is the single object that ties them all together. It
references:
  - git_commit (the code that produced the run)
  - dataset_manifest_sha256 (which dataset version)
  - tokenizer_version (e.g. "rule.v1+vqvae.v0+raw.v1")
  - model config (a serialized FusionConfig or kindred)
  - split policy (from DatasetManifest.split_policy)
  - checkpoint_sha256 (proves what binary you actually have)
  - per-epoch metrics + final val/test
  - per-benchmark metrics (Hungarian-matching, bounce/break AUROC, ...)

Per the user's institutional spec: "Every experiment must be reproducible
from one command." Combined with DatasetManifest, this is enough.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .dataset import git_head_commit, sha256_file


class EpochMetric(BaseModel):
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    n_ok: Optional[int] = None
    n_skipped_nan: Optional[int] = None
    seconds: Optional[float] = None
    extra: dict = Field(default_factory=dict)


class ExperimentManifest(BaseModel):
    run_id: str                                        # e.g. "pretrain-mask-v0.1-1777154272"
    experiment_kind: str                               # "pretrain" | "supervised" | "ablation"
    created_at: int

    # Provenance — the spec's non-negotiables
    git_commit: str = ""
    dataset_manifest_path: str = ""                    # path to DatasetManifest .json
    dataset_manifest_sha256: str = ""                  # hash of THAT manifest
    tokenizer_version: str = ""                        # rule.v1 / vqvae.v0 / etc.
    split_policy: str = ""                             # echoed from DatasetManifest

    # The actual trained artifact
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    config_path: str = ""

    # Hyperparams + metrics
    cli_args: dict = Field(default_factory=dict)
    # `model_config` is a Pydantic-reserved name; we use model_cfg here
    model_cfg: dict = Field(default_factory=dict)
    epoch_metrics: list[EpochMetric] = Field(default_factory=list)
    final_metrics: dict = Field(default_factory=dict)

    # Optional benchmarks (filled later by separate scripts)
    benchmarks: dict = Field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def add_epoch(self, **kwargs) -> None:
        self.epoch_metrics.append(EpochMetric(**kwargs))


def begin_experiment(
    *,
    experiment_kind: str,
    cli_args: dict,
    model_cfg: dict,
    dataset_manifest_path: Path | str,
    tokenizer_version: str,
    repo_root: Path | None = None,
) -> ExperimentManifest:
    """Create a manifest at run start so even crashed runs are auditable."""
    from .dataset import DatasetManifest
    ts = int(time.time())
    run_id = f"{experiment_kind}-{ts}"
    dataset_manifest = DatasetManifest.load(dataset_manifest_path)
    return ExperimentManifest(
        run_id=run_id,
        experiment_kind=experiment_kind,
        created_at=ts,
        git_commit=git_head_commit(repo_root),
        dataset_manifest_path=str(dataset_manifest_path),
        dataset_manifest_sha256=dataset_manifest.manifest_sha256(),
        tokenizer_version=tokenizer_version,
        split_policy=dataset_manifest.split_policy.policy,
        cli_args=cli_args,
        model_cfg=model_cfg,
    )


def finalize_experiment(
    manifest: ExperimentManifest,
    *,
    checkpoint_path: Path | str,
    config_path: Path | str,
    final_metrics: dict,
    out_dir: Path | str,
) -> Path:
    """Stamp checkpoint sha256 + write the manifest to disk."""
    manifest.checkpoint_path = str(checkpoint_path)
    manifest.checkpoint_sha256 = sha256_file(checkpoint_path)
    manifest.config_path = str(config_path)
    manifest.final_metrics = final_metrics
    out_path = Path(out_dir) / "experiment_manifest.json"
    manifest.save(out_path)
    return out_path
