"""Per-class evaluation for the T3 geometry classifier.

The user's spec: "金融数据高度不平衡，所以不能只看 accuracy".
overall accuracy is dominated by the parallel_same class. We need
per-class precision / recall / F1.

Usage:
    python -m trendline_tokenizer.benchmarks.geometry_per_class \\
        --manifest checkpoints/fusion/geometry.v0.1-XXX/manifest.json \\
        --symbols BTCUSDT --timeframes 5m --max-records 1000
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .geometry_pairs import all_pairs_within_pair, label_distribution
from ..registry.manifest import ArtifactManifest
from ..retrain.refresh_dataset import collect_records
from ..training.geometry_pretrain import (
    GeometryClassifier, GeometryPairDataset,
    GEOMETRY_LABEL_TO_IDX, N_GEOMETRY_CLASSES,
)


IDX_TO_LABEL = {v: k for k, v in GEOMETRY_LABEL_TO_IDX.items()}


def per_class_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Macro-averaged precision/recall/F1 per class + confusion matrix."""
    n_classes = N_GEOMETRY_CLASSES
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class = {}
    for c in range(n_classes):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-9, precision + recall)
        per_class[IDX_TO_LABEL[c]] = {
            "support": int(confusion[c, :].sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    macro_f1 = float(np.mean([m["f1"] for m in per_class.values()]))
    overall_acc = float(np.diag(confusion).sum() / max(1, confusion.sum()))
    return {
        "n_samples": int(confusion.sum()),
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="checkpoints/fusion/geometry.v0.1-*/manifest.json")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--max-records", type=int, default=2000)
    ap.add_argument("--max-per-pair", type=int, default=200)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    manifest = ArtifactManifest.load(args.manifest)
    cfg = json.loads(Path(manifest.config_path).read_text(encoding="utf-8"))
    model = GeometryClassifier(hidden=cfg["hidden"])
    state = torch.load(manifest.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    print(f"[geom-eval] loaded {manifest.artifact_name}")

    records, _ = collect_records(symbols=args.symbols, timeframes=args.timeframes,
                                  manual_oversample=0)
    if args.max_records and len(records) > args.max_records:
        records = records[:args.max_records]
    pairs = all_pairs_within_pair(records, max_per_pair=args.max_per_pair)
    print(f"[geom-eval] eval on {len(pairs)} pairs from {len(records)} records")
    print(f"[geom-eval] truth distribution: {label_distribution(pairs)}")

    by_id = {r.id: r for r in records}
    ds = GeometryPairDataset(pairs, by_id)
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x = s["x"].unsqueeze(0)
            y = int(s["y"].item())
            logits = model(x)
            pred = int(logits.argmax(dim=-1).item())
            y_true.append(y); y_pred.append(pred)

    metrics = per_class_metrics(y_true, y_pred)
    print(f"[geom-eval] overall_acc={metrics['overall_accuracy']:.4f}")
    print(f"[geom-eval] macro_f1={metrics['macro_f1']:.4f}")
    print("[geom-eval] per-class breakdown:")
    for label, m in metrics["per_class"].items():
        print(f"  {label:>15}: support={m['support']:>5} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "manifest": manifest.model_dump(),
            "n_records_eval": len(records),
            "n_pairs_eval": len(pairs),
            "truth_distribution": label_distribution(pairs),
            "metrics": metrics,
        }, indent=2), encoding="utf-8")
        print(f"[geom-eval] saved {args.output}")
    return metrics


if __name__ == "__main__":
    main()
