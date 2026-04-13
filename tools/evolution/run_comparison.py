"""Run a matrix of variants side-by-side with optional harness overrides."""

from __future__ import annotations

import time
from pathlib import Path

from .evaluator import EvalConfig, evaluate_variant
from .harness import HarnessParams
from .orchestrator import round_dir


SYMBOL_POOL = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "HYPEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "DOGEUSDT", "PEPEUSDT", "SHIBUSDT", "WIFUSDT",
]
TIMEFRAMES = ["15m", "1h", "4h"]


def _fmt_metrics(m: dict) -> str:
    return (f"n_trig={m.get('n_setups_triggered',0):4d} "
            f"wr={m.get('fade_win_rate',0):.3f} "
            f"avg_R={m.get('avg_total_R',0):.3f} "
            f"tot_R={m.get('total_R',0):7.2f} "
            f"flip_wr={m.get('flip_win_rate',0):.3f}")


def main():
    config = EvalConfig(
        symbols=SYMBOL_POOL,
        timeframes=TIMEFRAMES,
        days=730,
        train_fraction=0.7,
        min_total_triggered=20,
    )

    default_harness = HarnessParams()  # enable_flip=True, max_setups=2
    fade_only_harness = HarnessParams(enable_flip=False)
    touch3_only_harness = HarnessParams(max_setups_per_line=1, enable_flip=False)

    out = round_dir(0)

    print("=" * 70)
    print("ROUND 2 COMPARISON (in-sample lookahead fixed, fitness formula fixed)")
    print("=" * 70)

    variants = [
        ("server.strategy.evolved.v0_baseline", "v0",         default_harness),
        ("server.strategy.evolved.v1_clean",    "v1",         default_harness),
        ("server.strategy.evolved.v1a_filtered", "v1a",       default_harness),
        ("server.strategy.evolved.v1a_filtered", "v1a_nof",   fade_only_harness),     # same detector, no flip
        ("server.strategy.evolved.v1a_filtered", "v1a_t3",    touch3_only_harness),   # touch-3-only, no flip
    ]

    results = {}
    for variant_path, tag, harness in variants:
        print(f"\n--- {tag} (harness: flip={harness.enable_flip}, max_setups={harness.max_setups_per_line}) ---")
        t0 = time.time()
        result = evaluate_variant(variant_path, config, out, harness_override=harness)
        dt = time.time() - t0
        print(f"  time: {dt:.1f}s")
        print(f"  fitness_train: {result.fitness_train:.3f}")
        print(f"  fitness_test:  {result.fitness_test:.3f}")
        print(f"  train: {_fmt_metrics(result.train)}")
        print(f"  test:  {_fmt_metrics(result.test)}")

        by_touch = result.test.get("by_setup_touch", {})
        print(f"  per-touch (test):")
        for k in sorted(by_touch.keys(), key=lambda x: int(x) if str(x).isdigit() else 99):
            t = by_touch[k]
            if t.get("n_setups_triggered", 0) == 0 and int(k) > 0:
                continue
            print(f"    touch={k}: {_fmt_metrics(t)}")

        results[tag] = result

    # Winner selection: ranked by fitness_test
    print("\n" + "=" * 70)
    print("WINNER RANKING (by fitness_test, with total_R and WR as primary)")
    print("=" * 70)
    ranked = sorted(results.items(), key=lambda kv: kv[1].fitness_test, reverse=True)
    for i, (tag, r) in enumerate(ranked):
        print(f"  {i+1}. {tag:10s}  fitness_test={r.fitness_test:8.2f}  "
              f"total_R={r.test.get('total_R',0):7.2f}  "
              f"wr={r.test.get('fade_win_rate',0):.3f}  "
              f"n_trig={r.test.get('n_setups_triggered',0)}")

    print(f"\nWinner: {ranked[0][0]}")


if __name__ == "__main__":
    main()
