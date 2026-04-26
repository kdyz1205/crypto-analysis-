"""Per-class evaluation for the multi-task fusion model heads.

Per the user's research-grade spec, every head needs a per-class
report — overall accuracy on imbalanced data is meaningless. This
runs a forward pass over a held-out slice and emits:

  bounce / break / continuation:  P/R/F1 per class (binary)
  regime:        P/R/F1 across low/normal/high_vol
  invalidation:  P/R/F1 across 5 invalidation classes

  buffer_pct:    MAE + median absolute error
  next_coarse:   top-1 + top-5 + top-10 accuracy

Outputs a JSON report at data/logs/benchmarks/multi_task_<artifact>.json.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..adapters import iter_legacy_pattern_records
from ..evolve.draw import _ohlcv_dataframe
from ..models.config import FusionConfig
from ..models.full_model import TrendlineFusionModel
from ..registry.manifest import ArtifactManifest
from ..retrain.refresh_dataset import collect_records
from ..training.sequence_dataset import build_examples, SequenceDataset
from ..training.splits import split_records


REGIME_LABELS = ["low_vol", "normal_vol", "high_vol"]
INVALIDATION_LABELS = ["valid", "weak_pen", "confirmed_break",
                       "break_retest", "failed_breakout"]
BINARY_LABELS = ["no", "yes"]


def per_class_from_arrays(y_true: np.ndarray, y_pred: np.ndarray,
                          n_classes: int, label_names: list[str]) -> dict:
    """Returns precision/recall/F1 per class + macro F1 + confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        per_class[label_names[c]] = {
            "support": int(cm[c, :].sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {
        "n_samples": int(cm.sum()),
        "overall_accuracy": round(float(np.diag(cm).sum() / max(1, cm.sum())), 4),
        "macro_f1": round(float(np.mean([m["f1"] for m in per_class.values()])), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(
    *,
    manifest_path: Path,
    symbols: list[str], timeframes: list[str],
    max_records: int = 5000,
    horizon_bars: int = 20,
    output_path: Path | None = None,
    device: str | None = None,
) -> dict:
    manifest = ArtifactManifest.load(manifest_path)
    cfg_dict = json.loads(Path(manifest.config_path).read_text(encoding="utf-8"))
    cfg = FusionConfig(**cfg_dict)
    print(f"[mt-eval] manifest: {manifest.artifact_name}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TrendlineFusionModel(cfg).to(device).eval()
    state = torch.load(manifest.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    print(f"[mt-eval] loaded model on {device}")

    records, stats = collect_records(symbols=symbols, timeframes=timeframes,
                                      manual_oversample=0)
    if max_records and len(records) > max_records:
        records = records[:max_records]
    print(f"[mt-eval] eval pool: {len(records)} records")

    # Use TIME_FORWARD split — only evaluate on val + test
    split = split_records(records, policy="time_forward")
    eval_idx = np.concatenate([split.val_idx, split.test_idx])
    eval_records = [records[i] for i in eval_idx]
    print(f"[mt-eval] eval set: {len(eval_records)} records "
          f"(val={len(split.val_idx)} + test={len(split.test_idx)})")

    by_key = defaultdict(list)
    for r in eval_records:
        by_key[(r.symbol, r.timeframe)].append(r)
    all_examples = []
    for (sym, tf), recs in by_key.items():
        df = _ohlcv_dataframe(sym, tf)
        if df is None:
            continue
        ex = build_examples(df, recs, price_seq_len=cfg.price_seq_len,
                            token_seq_len=cfg.token_seq_len,
                            horizon_bars=horizon_bars,
                            raw_feat_dim=cfg.raw_feat_dim)
        all_examples.extend(ex)
    print(f"[mt-eval] eval examples: {len(all_examples)}")

    ds = SequenceDataset(all_examples)
    loader = DataLoader(ds, batch_size=64)

    # Collect predictions + targets for every head
    coll: dict[str, list] = {
        "next_coarse_logits": [], "next_fine_logits": [],
        "bounce_pred": [], "bounce_true": [],
        "break_pred": [], "break_true": [],
        "cont_pred": [], "cont_true": [],
        "buffer_pred": [], "buffer_true": [],
        "regime_pred": [], "regime_true": [],
        "invalidation_pred": [], "invalidation_true": [],
        "next_coarse_true": [],
    }

    target_keys = ("next_coarse", "next_fine", "bounce", "brk", "cont",
                   "buffer_pct", "regime", "invalidation")

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {k: batch.pop(k) for k in target_keys}
            for k, v in list(batch.items()):
                if v.dtype.is_floating_point:
                    batch[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            out = model(batch)

            if "next_coarse_logits" in out:
                coll["next_coarse_logits"].append(out["next_coarse_logits"].cpu())
                coll["next_coarse_true"].append(targets["next_coarse"].cpu())

            if "bounce_logits" in out:
                p = out["bounce_logits"].argmax(dim=-1).cpu().numpy()
                coll["bounce_pred"].extend(p.tolist())
                coll["bounce_true"].extend(targets["bounce"].cpu().numpy().tolist())
            if "break_logits" in out:
                p = out["break_logits"].argmax(dim=-1).cpu().numpy()
                coll["break_pred"].extend(p.tolist())
                coll["break_true"].extend(targets["brk"].cpu().numpy().tolist())
            if "continuation_logits" in out:
                p = out["continuation_logits"].argmax(dim=-1).cpu().numpy()
                coll["cont_pred"].extend(p.tolist())
                coll["cont_true"].extend(targets["cont"].cpu().numpy().tolist())
            if "buffer_pct" in out:
                coll["buffer_pred"].extend(out["buffer_pct"].cpu().numpy().tolist())
                coll["buffer_true"].extend(targets["buffer_pct"].cpu().numpy().tolist())
            if "regime_logits" in out:
                p = out["regime_logits"].argmax(dim=-1).cpu().numpy()
                coll["regime_pred"].extend(p.tolist())
                coll["regime_true"].extend(targets["regime"].cpu().numpy().tolist())
            if "invalidation_logits" in out:
                p = out["invalidation_logits"].argmax(dim=-1).cpu().numpy()
                coll["invalidation_pred"].extend(p.tolist())
                coll["invalidation_true"].extend(targets["invalidation"].cpu().numpy().tolist())

    report = {"manifest": manifest.model_dump(),
              "n_eval_examples": len(all_examples)}

    # Per-head reports
    if coll["bounce_true"]:
        report["bounce"] = per_class_from_arrays(
            np.array(coll["bounce_true"]), np.array(coll["bounce_pred"]),
            n_classes=2, label_names=BINARY_LABELS,
        )
    if coll["break_true"]:
        report["break"] = per_class_from_arrays(
            np.array(coll["break_true"]), np.array(coll["break_pred"]),
            n_classes=2, label_names=BINARY_LABELS,
        )
    if coll["cont_true"]:
        report["continuation"] = per_class_from_arrays(
            np.array(coll["cont_true"]), np.array(coll["cont_pred"]),
            n_classes=2, label_names=BINARY_LABELS,
        )
    if coll["regime_true"]:
        report["regime"] = per_class_from_arrays(
            np.array(coll["regime_true"]), np.array(coll["regime_pred"]),
            n_classes=3, label_names=REGIME_LABELS,
        )
    if coll["invalidation_true"]:
        report["invalidation"] = per_class_from_arrays(
            np.array(coll["invalidation_true"]), np.array(coll["invalidation_pred"]),
            n_classes=5, label_names=INVALIDATION_LABELS,
        )
    if coll["buffer_true"]:
        true = np.array(coll["buffer_true"]); pred = np.array(coll["buffer_pred"])
        err = np.abs(pred - true)
        report["buffer_pct"] = {
            "n_samples": int(len(err)),
            "mae": round(float(err.mean()), 6),
            "median_abs_err": round(float(np.median(err)), 6),
            "p95_abs_err": round(float(np.quantile(err, 0.95)), 6),
            "true_mean": round(float(true.mean()), 6),
            "pred_mean": round(float(pred.mean()), 6),
        }
    if coll["next_coarse_logits"]:
        logits = torch.cat(coll["next_coarse_logits"], dim=0)
        true = torch.cat(coll["next_coarse_true"], dim=0)
        top1 = (logits.argmax(dim=-1) == true).float().mean().item()
        topk = torch.topk(logits, k=10, dim=-1).indices
        top5 = (topk[:, :5] == true.unsqueeze(1)).any(dim=1).float().mean().item()
        top10 = (topk == true.unsqueeze(1)).any(dim=1).float().mean().item()
        report["next_coarse"] = {
            "n_samples": int(len(true)),
            "top1": round(top1, 4),
            "top5": round(top5, 4),
            "top10": round(top10, 4),
        }

    # Print summary
    print(f"\n[mt-eval] === REPORT for {manifest.artifact_name} ===")
    for head_name in ("bounce", "break", "continuation", "regime", "invalidation"):
        if head_name in report:
            r = report[head_name]
            print(f"  {head_name:>14}: n={r['n_samples']} "
                  f"acc={r['overall_accuracy']:.3f} macro_f1={r['macro_f1']:.3f}")
    if "buffer_pct" in report:
        b = report["buffer_pct"]
        print(f"     buffer_pct: n={b['n_samples']} mae={b['mae']:.4f} "
              f"median={b['median_abs_err']:.4f}")
    if "next_coarse" in report:
        n = report["next_coarse"]
        print(f"   next_coarse: n={n['n_samples']} top1={n['top1']:.3f} "
              f"top5={n['top5']:.3f} top10={n['top10']:.3f}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[mt-eval] saved {output_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--max-records", type=int, default=5000)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate_model(
        manifest_path=args.manifest,
        symbols=args.symbols, timeframes=args.timeframes,
        max_records=args.max_records,
        output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()
