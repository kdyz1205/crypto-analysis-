"""CLI demo: run the AutoregressiveGenerator on a real OHLCV window.

Pulls the freshest cached OHLCV for one (symbol, timeframe), detects the
existing trendlines on that window, builds the inference batch the same
way InferenceService does, then projects N future trendlines and prints
them with their bounce/break probabilities.

Usage:
    python -m trendline_tokenizer.generation.demo \\
        --manifest checkpoints/fusion/<artifact>/manifest.json \\
        --symbol HYPEUSDT --timeframe 1h --n-steps 10 --temperature 0.7 \\
        --top-k 50 --seed 42

This is the user-visible "show me what the model thinks happens next"
endpoint. It's NOT a benchmark — it's a dump for eyeball verification
that the trained checkpoint actually projects sensible structures
(e.g. an ascending support after a sustained uptrend, not random noise).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..evolve.draw import _ohlcv_dataframe
from ..generation.autoregressive import AutoregressiveGenerator
from ..inference.runtime_detector import detect_lines
from ..inference.runtime_tokenizer import RuntimeTokenizer
from ..inference.feature_cache import FeatureCache
from ..models.config import FusionConfig
from ..models.full_model import TrendlineFusionModel
from ..registry.manifest import ArtifactManifest


def _build_inference_batch(symbol: str, tf: str, cfg: FusionConfig,
                            tokenizer: RuntimeTokenizer, device: str | torch.device):
    """Mirrors InferenceService.predict's batch construction. Returns
    (batch_dict, lines, df) so the caller can also see what the input
    context actually was."""
    df = _ohlcv_dataframe(symbol, tf)
    if df is None or len(df) < cfg.price_seq_len:
        raise SystemExit(f"insufficient OHLCV cache for {symbol} {tf} "
                          f"(need >={cfg.price_seq_len} bars)")

    lines = detect_lines(df, symbol=symbol, timeframe=tf,
                          max_lines=cfg.token_seq_len)
    lines = sorted(lines, key=lambda r: r.end_bar_index)[-cfg.token_seq_len:]
    tok = tokenizer.encode_records(lines)

    T = cfg.token_seq_len
    rule_c = np.zeros(T, dtype=np.int64); rule_c[T - len(lines):] = tok["rule_coarse"]
    rule_f = np.zeros(T, dtype=np.int64); rule_f[T - len(lines):] = tok["rule_fine"]
    l_c = np.zeros(T, dtype=np.int64); l_c[T - len(lines):] = tok["learned_coarse"]
    l_f = np.zeros(T, dtype=np.int64); l_f[T - len(lines):] = tok["learned_fine"]
    raw = np.zeros((T, cfg.raw_feat_dim), dtype=np.float32)
    if len(lines) > 0:
        raw[T - len(lines):] = tok["raw_feat"]
    token_pad = np.ones(T, dtype=bool); token_pad[T - len(lines):] = False

    # Use FeatureCache to build the price window with training-parity feats
    fc = FeatureCache(capacity=max(512, cfg.price_seq_len * 2))
    for _, row in df.iterrows():
        bar = {
            "open_time": int(row["open_time"]),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        fc.push(symbol, tf, bar)
    price_window, price_pad = fc.price_window(symbol, tf, cfg.price_seq_len)

    batch = {
        "price": torch.from_numpy(price_window).unsqueeze(0).to(device),
        "price_pad": torch.from_numpy(price_pad).unsqueeze(0).to(device),
        "rule_coarse": torch.from_numpy(rule_c).unsqueeze(0).to(device),
        "rule_fine": torch.from_numpy(rule_f).unsqueeze(0).to(device),
        "learned_coarse": torch.from_numpy(l_c).unsqueeze(0).to(device),
        "learned_fine": torch.from_numpy(l_f).unsqueeze(0).to(device),
        "raw_feat": torch.from_numpy(raw).unsqueeze(0).to(device),
        "token_pad": torch.from_numpy(token_pad).unsqueeze(0).to(device),
    }
    return batch, lines, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help=">0 sample; <=0 greedy")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=None,
                    help="optional JSON output path")
    args = ap.parse_args()

    manifest = ArtifactManifest.load(args.manifest)
    cfg_dict = json.loads(Path(manifest.config_path).read_text(encoding="utf-8"))
    cfg = FusionConfig(**cfg_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo] manifest: {manifest.artifact_name} on {device}")

    model = TrendlineFusionModel(cfg).to(device).eval()
    state = torch.load(manifest.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)

    tokenizer = RuntimeTokenizer(
        use_rule=cfg.use_rule_tokens,
        use_learned=cfg.use_learned_tokens,
        use_raw=cfg.use_raw_features,
        vqvae_path=cfg.vqvae_checkpoint_path,
    )
    batch, lines, df = _build_inference_batch(args.symbol, args.timeframe,
                                                cfg, tokenizer, device)

    print(f"[demo] OHLCV window: {len(df)} bars; "
          f"detected {len(lines)} existing lines")
    print(f"[demo] sampling {args.n_steps} future trendlines "
          f"(T={args.temperature}, top_k={args.top_k}, seed={args.seed})")

    ref = lines[-1] if lines else None
    gen = AutoregressiveGenerator(model, device=device)
    steps = gen.generate(
        batch, n_steps=args.n_steps,
        temperature=args.temperature, top_k=args.top_k,
        seed=args.seed, reference_record=ref,
    )

    print("\n┌────┬──────────┬──────┬──────────┬──────┬──────┬──────┬────────┐")
    print("│ ## │   role   │ dir  │   slope  │ bnc% │ brk% │ con% │ buf %  │")
    print("├────┼──────────┼──────┼──────────┼──────┼──────┼──────┼────────┤")
    for s in steps:
        r = s.decoded_record
        print(f"│ {s.step:>2} │ {r.line_role:>8} │ {r.direction:>4} │ "
              f"{r.log_slope_per_bar():>+9.5f} │ "
              f"{s.bounce_prob*100:>4.1f} │ {s.break_prob*100:>4.1f} │ "
              f"{s.continuation_prob*100:>4.1f} │ "
              f"{s.suggested_buffer_pct*100:>5.3f} │")
    print("└────┴──────────┴──────┴──────────┴──────┴──────┴──────┴────────┘")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": manifest.model_dump(),
            "symbol": args.symbol, "timeframe": args.timeframe,
            "n_input_lines": len(lines),
            "n_bars": len(df),
            "sampling": {"temperature": args.temperature,
                         "top_k": args.top_k, "seed": args.seed},
            "steps": [{
                "step": s.step,
                "rule_coarse_id": s.rule_coarse_id,
                "rule_fine_id": s.rule_fine_id,
                "bounce_prob": s.bounce_prob,
                "break_prob": s.break_prob,
                "continuation_prob": s.continuation_prob,
                "suggested_buffer_pct": s.suggested_buffer_pct,
                "decoded": {
                    "role": s.decoded_record.line_role,
                    "direction": s.decoded_record.direction,
                    "log_slope_per_bar": s.decoded_record.log_slope_per_bar(),
                    "duration_bars": s.decoded_record.duration_bars(),
                    "start_price": s.decoded_record.start_price,
                    "end_price": s.decoded_record.end_price,
                },
                "top5_coarse_ids": s.top5_coarse_ids,
                "top5_coarse_probs": s.top5_coarse_probs,
            } for s in steps],
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[demo] saved {args.output}")


if __name__ == "__main__":
    main()
