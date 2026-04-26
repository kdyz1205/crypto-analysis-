"""Phase 1.T3: Geometry consistency classifier.

Takes pairs of trendlines (line_a, line_b) from the same (symbol,
timeframe) window, classifies their joint structure into 6 categories:
  0 channel
  1 triangle
  2 wedge
  3 parallel_same
  4 diverging
  5 unrelated

Architecture:
  - Each line is encoded via build_feature_vector (36-dim)
  - Concat (a_feat, b_feat, a_feat * b_feat, a_feat - b_feat) = 144-dim
    (the elementwise interaction terms force the model to learn relations)
  - 2-layer MLP -> 6-class softmax

Loss: CE on the geometry label.

Why a small MLP not the full fusion model: at this stage we just need
the LABELS to be well-defined and a baseline classifier to validate
that the labels carry signal. The full Transformer-fusion version goes
in models/heads.py once T1+T2+T3 all work in isolation.

Run:
    python -m trendline_tokenizer.training.geometry_pretrain \\
        --symbols BTCUSDT --timeframes 5m --max-records 5000 --epochs 3
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..benchmarks.geometry_pairs import (
    LinePair, all_pairs_within_pair, label_distribution,
)
from ..features.vector import build_feature_vector, FEATURE_VECTOR_DIM
from ..registry.dataset import (
    DatasetManifest, SplitSpec, build_manifest_from_collect_records,
)
from ..registry.experiment import begin_experiment, finalize_experiment
from ..registry.paths import fusion_dir, ROOT as REGISTRY_ROOT
from ..registry.manifest import ArtifactManifest
from ..retrain.refresh_dataset import (
    collect_records, DEFAULT_PATTERNS, DEFAULT_PATTERNS_SWEEP,
    DEFAULT_MANUAL, DEFAULT_OUTCOMES,
)
from ..schemas.trendline import TrendlineRecord


GEOMETRY_LABEL_TO_IDX = {
    "channel": 0, "triangle": 1, "wedge": 2,
    "parallel_same": 3, "diverging": 4, "unrelated": 5,
}
N_GEOMETRY_CLASSES = len(GEOMETRY_LABEL_TO_IDX)


class GeometryPairDataset(Dataset):
    """Pair-of-trendlines dataset, with class index target."""

    def __init__(self, pairs: list[LinePair], records_by_id: dict[str, TrendlineRecord]):
        self.pairs = pairs
        self.records_by_id = records_by_id
        # pre-build feature vectors once
        self._feats: dict[str, np.ndarray] = {
            r.id: build_feature_vector(r) for r in records_by_id.values()
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        a = self._feats[p.line_a_id]
        b = self._feats[p.line_b_id]
        # Interaction features (key for relation learning)
        x = np.concatenate([a, b, a * b, a - b], axis=0).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = GEOMETRY_LABEL_TO_IDX[p.label]
        return {"x": torch.from_numpy(x),
                "y": torch.tensor(y, dtype=torch.long)}


class GeometryClassifier(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = FEATURE_VECTOR_DIM * 4   # a, b, a*b, a-b
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_GEOMETRY_CLASSES),
        )

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, batch):
        logits = self.forward(batch["x"])
        loss = F.cross_entropy(logits, batch["y"])
        # Track per-class accuracy
        pred = logits.argmax(dim=-1)
        acc = (pred == batch["y"]).float().mean().item()
        return loss, {"loss": loss.detach().item(), "acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--max-records", type=int, default=5000)
    ap.add_argument("--max-per-pair", type=int, default=200,
                    help="cap trendlines per (sym,tf) before pairing (n^2)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--manual-oversample", type=int, default=0,
                    help="0 = no oversample (pairs are already abundant)")
    ap.add_argument("--dataset-version", default="trendline-structure-v0.1.0")
    args = ap.parse_args()

    records, stats = collect_records(
        symbols=args.symbols, timeframes=args.timeframes,
        max_legacy_per_pair=None,
        manual_oversample=args.manual_oversample,
    )
    print(f"[geom] pool stats: {stats}")
    if args.max_records and len(records) > args.max_records:
        records = records[:args.max_records]

    by_id = {r.id: r for r in records}
    pairs = all_pairs_within_pair(records, max_per_pair=args.max_per_pair)
    print(f"[geom] generated {len(pairs)} pairs from {len(records)} records")
    print(f"[geom] label distribution: {label_distribution(pairs)}")

    if not pairs:
        raise SystemExit("no pairs generated")

    # ── Layer 0: provenance ────
    pretrain_run_dir = REGISTRY_ROOT / "experiments" / f"geom-{int(time.time())}"
    pretrain_run_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        "auto_patterns": DEFAULT_PATTERNS,
        "patterns_sweep": DEFAULT_PATTERNS_SWEEP,
        "manual_lines": DEFAULT_MANUAL,
        "user_outcomes": DEFAULT_OUTCOMES,
    }
    raw_paths = {k: v for k, v in raw_paths.items() if v.exists()}
    # Random pair-level split (pairs are already lots; tiny smoke can use random)
    n = len(pairs)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.15))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]

    split_spec = SplitSpec(
        policy="random", train_count=len(train_pairs),
        val_count=len(val_pairs), test_count=n_test,
        notes="random pair-level split; pair labels are deterministic so "
              "this is OK for the classifier itself, but downstream eval "
              "MUST use time/symbol/regime splits via record-level pools",
    )
    dataset_manifest = build_manifest_from_collect_records(
        dataset_name="trendline-geometry-pairs",
        dataset_version=args.dataset_version,
        raw_paths=raw_paths,
        symbols=args.symbols, timeframes=args.timeframes,
        stats=stats, split_spec=split_spec,
    )
    dm_path = pretrain_run_dir / "dataset_manifest.json"
    dataset_manifest.save(dm_path)
    print(f"[geom] dataset manifest: {dm_path}")

    exp_manifest = begin_experiment(
        experiment_kind="pretrain-geometry",
        cli_args=vars(args),
        model_cfg={"hidden": args.hidden, "n_classes": N_GEOMETRY_CLASSES,
                   "input_dim": FEATURE_VECTOR_DIM * 4},
        dataset_manifest_path=dm_path,
        tokenizer_version="raw_features.v1",
    )

    train_ds = GeometryPairDataset(train_pairs, by_id)
    val_ds = GeometryPairDataset(val_pairs, by_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[geom] device={device}")
    model = GeometryClassifier(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        n_ok, agg_loss, agg_acc = 0, 0.0, 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss, parts = model.compute_loss(batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n_ok += 1; agg_loss += parts["loss"]; agg_acc += parts["acc"]
        train_loss = agg_loss / max(1, n_ok)
        train_acc = agg_acc / max(1, n_ok)

        model.eval()
        n_v, agg_l, agg_a = 0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, parts = model.compute_loss(batch)
                n_v += 1; agg_l += parts["loss"]; agg_a += parts["acc"]
        val_loss = agg_l / max(1, n_v)
        val_acc = agg_a / max(1, n_v)
        print(f"[geom] epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({time.time()-t0:.1f}s)")
        exp_manifest.add_epoch(
            epoch=epoch, train_loss=float(train_loss), val_loss=float(val_loss),
            n_ok=n_ok, n_skipped_nan=0, seconds=float(time.time() - t0),
            extra={"train_acc": float(train_acc), "val_acc": float(val_acc)},
        )

    # save
    name = f"geometry.v0.1-{int(time.time())}"
    out_dir = fusion_dir(name)
    torch.save({"state_dict": model.state_dict(),
                "hidden": args.hidden,
                "label_to_idx": GEOMETRY_LABEL_TO_IDX},
               out_dir / "geometry.pt")
    (out_dir / "geometry_config.json").write_text(
        json.dumps({"hidden": args.hidden,
                    "n_classes": N_GEOMETRY_CLASSES,
                    "label_to_idx": GEOMETRY_LABEL_TO_IDX}, indent=2),
        encoding="utf-8")

    manifest = ArtifactManifest(
        artifact_name=name, model_kind="geometry-classifier",
        model_version="geometry.v0.1",
        tokenizer_version="raw_features.v1",
        training_dataset_version=args.dataset_version,
        feature_norm_version="none",
        ckpt_path=str(out_dir / "geometry.pt"),
        config_path=str(out_dir / "geometry_config.json"),
        metrics={"val_loss": float(val_loss), "val_acc": float(val_acc),
                 "n_pairs": n, "n_records": len(records)},
        created_at=int(time.time()),
    )
    manifest.save(out_dir / "manifest.json")
    print(f"[geom] saved {out_dir}")

    finalize_experiment(
        exp_manifest,
        checkpoint_path=out_dir / "geometry.pt",
        config_path=out_dir / "geometry_config.json",
        final_metrics={"val_loss": float(val_loss), "val_acc": float(val_acc),
                       "n_pairs": n, "label_distribution": label_distribution(pairs)},
        out_dir=out_dir,
    )

    try:
        from ..registry.hf_backup import maybe_upload
        maybe_upload(out_dir)
    except Exception as _e:
        print(f"[geom] hf backup failed: {_e}")


if __name__ == "__main__":
    main()
