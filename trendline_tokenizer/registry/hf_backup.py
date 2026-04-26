"""HuggingFace Hub backup for trained checkpoints.

Per CLAUDE.md DO NOT LIE: data was lost mid-session. This module backs
up every saved checkpoint to a HF Hub repo so even if the local disk
gets wiped, the artifact survives.

Usage (auto-called from train_fusion / mask_pretrain after save):
    from trendline_tokenizer.registry.hf_backup import maybe_upload
    maybe_upload(out_dir, repo_id="kdyz1205/trendline-fusion-v0.1")

Auth: relies on HF_TOKEN env var (or `huggingface-cli login`).
Silently no-ops if token missing or upload fails — never blocks training.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional


def maybe_upload(
    artifact_dir: Path | str,
    *,
    repo_id: Optional[str] = None,
    repo_type: str = "model",
) -> bool:
    """Best-effort upload of every file in artifact_dir to HF Hub.

    Returns True on success, False on no-op or failure. Never raises.
    """
    repo_id = repo_id or os.environ.get("TRENDLINE_HF_REPO")
    if not repo_id:
        return False
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print(f"[hf_backup] no HF_TOKEN env; skipping backup of {artifact_dir}")
        return False

    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        return False

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[hf_backup] huggingface_hub not installed; skipping")
        return False

    try:
        api = HfApi(token=token)
        # Create repo if it doesn't exist (idempotent with exist_ok)
        try:
            create_repo(repo_id=repo_id, repo_type=repo_type,
                        exist_ok=True, token=token, private=True)
        except Exception as exc:
            print(f"[hf_backup] create_repo soft-fail: {exc}")

        # Upload every file in artifact_dir under <artifact_name>/<filename>
        artifact_name = artifact_dir.name
        n_uploaded = 0
        for f in artifact_dir.iterdir():
            if not f.is_file():
                continue
            target = f"{artifact_name}/{f.name}"
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=target,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
            )
            n_uploaded += 1
        print(f"[hf_backup] uploaded {n_uploaded} files to {repo_id}/{artifact_name}")
        return True
    except Exception as exc:
        print(f"[hf_backup] upload failed: {exc}")
        return False


def maybe_download(
    artifact_name: str,
    local_dir: Path | str,
    *,
    repo_id: Optional[str] = None,
    repo_type: str = "model",
) -> bool:
    """Pull a previously-uploaded artifact back from HF Hub.

    Useful for recovery: if the local checkpoint disappeared, this can
    restore it from the cloud backup.
    """
    repo_id = repo_id or os.environ.get("TRENDLINE_HF_REPO")
    if not repo_id:
        return False
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return False

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id, repo_type=repo_type,
            allow_patterns=[f"{artifact_name}/*"],
            local_dir=str(local_dir.parent),
            token=token,
        )
        print(f"[hf_backup] restored {artifact_name} from {repo_id}")
        return True
    except Exception as exc:
        print(f"[hf_backup] restore failed: {exc}")
        return False
