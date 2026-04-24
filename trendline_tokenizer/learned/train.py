"""Train the hierarchical VQ-VAE.

Usage:
    python -m trendline_tokenizer.learned.train \
        --epochs 5 --batch-size 256 --limit 50000

With torch+cuda the model auto-moves to GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import build_dataset
from .vqvae import HierarchicalVQVAE, VQVAEConfig
from ..tokenizer.vocab import LINE_ROLES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATTERNS = PROJECT_ROOT / "data" / "patterns"
DEFAULT_MANUAL = PROJECT_ROOT / "data" / "manual_trendlines.json"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "trendline_tokenizer" / "vqvae_v0.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns-dir", default=str(DEFAULT_PATTERNS))
    ap.add_argument("--manual", default=str(DEFAULT_MANUAL))
    ap.add_argument("--limit", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--ckpt-out", default=str(DEFAULT_CKPT))
    ap.add_argument("--device", default=None, help="'cuda' / 'cpu' / auto")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    t0 = time.time()
    ds, recs = build_dataset(
        patterns_dir=Path(args.patterns_dir),
        manual_json=Path(args.manual),
        limit=args.limit,
    )
    print(f"[train] dataset: {len(ds)} records in {time.time()-t0:.1f}s")
    if len(ds) == 0:
        print("[train] no data — abort")
        return

    val_n = max(1, int(len(ds) * args.val_frac))
    train_n = len(ds) - val_n
    ds_train, ds_val = random_split(ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    cfg = VQVAEConfig()
    model = HierarchicalVQVAE(cfg).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-6,
    )

    hist = []
    ckpt_path = Path(args.ckpt_out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        n_batches = 0
        agg = {"loss": 0.0, "recon": 0.0, "role_ce": 0.0, "bounce_ce": 0.0, "break_ce": 0.0, "commit": 0.0}
        last_codes = (0, 0)
        for batch in dl_train:
            feat = batch["feat"].to(device)
            target = {
                "role_idx": batch["role_idx"].to(device),
                "bounce_lbl": batch["bounce_lbl"].to(device),
                "break_lbl": batch["break_lbl"].to(device),
            }
            total, metrics = model.compute_loss(feat, target)
            opt.zero_grad()
            total.backward()
            opt.step()
            for k in agg:
                agg[k] += metrics[k]
            last_codes = (metrics["coarse_codes_used"], metrics["fine_codes_used"])
            n_batches += 1
        for k in agg:
            agg[k] /= max(1, n_batches)

        # validation
        model.eval()
        val_loss = 0.0
        val_role_correct = 0
        val_bounce_correct = 0
        val_break_correct = 0
        val_n_total = 0
        with torch.no_grad():
            for batch in dl_val:
                feat = batch["feat"].to(device)
                target = {
                    "role_idx": batch["role_idx"].to(device),
                    "bounce_lbl": batch["bounce_lbl"].to(device),
                    "break_lbl": batch["break_lbl"].to(device),
                }
                total, metrics = model.compute_loss(feat, target)
                val_loss += metrics["loss"] * feat.shape[0]
                out = model.forward(feat)
                val_role_correct += (out["role_logits"].argmax(-1) == target["role_idx"]).sum().item()
                val_bounce_correct += (out["bounce_logits"].argmax(-1) == target["bounce_lbl"]).sum().item()
                val_break_correct += (out["break_logits"].argmax(-1) == target["break_lbl"]).sum().item()
                val_n_total += feat.shape[0]
        val_loss /= max(1, val_n_total)
        row = {
            "epoch": epoch,
            "train_loss": agg["loss"],
            "train_recon": agg["recon"],
            "train_role_ce": agg["role_ce"],
            "val_loss": val_loss,
            "val_role_acc": val_role_correct / max(1, val_n_total),
            "val_bounce_acc": val_bounce_correct / max(1, val_n_total),
            "val_break_acc": val_break_correct / max(1, val_n_total),
            "coarse_codes_used": last_codes[0],
            "fine_codes_used": last_codes[1],
            "seconds": round(time.time() - ep_t0, 1),
        }
        hist.append(row)
        print(f"[train] ep{epoch:>2}  train_loss={row['train_loss']:.4f}  recon={row['train_recon']:.4f}"
              f"  val_loss={row['val_loss']:.4f}  role_acc={row['val_role_acc']:.3f}"
              f"  bounce_acc={row['val_bounce_acc']:.3f}  break_acc={row['val_break_acc']:.3f}"
              f"  coarse_used={row['coarse_codes_used']}/256  fine_used={row['fine_codes_used']}/1024"
              f"  ({row['seconds']}s)")

        # save ckpt each epoch
        torch.save({
            "model_state": model.state_dict(),
            "cfg": vars(cfg),
            "history": hist,
            "epoch": epoch,
        }, ckpt_path)

    print(f"[train] done. ckpt at {ckpt_path}  ({sum(r['seconds'] for r in hist):.1f}s total)")
    print(f"[train] final history: {json.dumps(hist[-1], indent=2)}")


if __name__ == "__main__":
    main()
