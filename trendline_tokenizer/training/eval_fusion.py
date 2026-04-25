"""Holdout evaluation: load a manifest, print its metrics, sanity-check
the checkpoint can be loaded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from ..models.config import FusionConfig
from ..models.full_model import TrendlineFusionModel
from ..registry.manifest import ArtifactManifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    manifest = ArtifactManifest.load(args.manifest)
    cfg = FusionConfig(**json.loads(Path(manifest.config_path).read_text(encoding="utf-8")))
    model = TrendlineFusionModel(cfg)
    state = torch.load(manifest.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    print(f"[eval] loaded {manifest.artifact_name}")
    print(f"[eval] model_version={manifest.model_version}")
    print(f"[eval] tokenizer_version={manifest.tokenizer_version}")
    print(f"[eval] metrics={json.dumps(manifest.metrics, indent=2)}")


if __name__ == "__main__":
    main()
