"""Run v0_baseline and v1_clean side-by-side, dump a comparison report."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .evaluator import EvalConfig, evaluate_variant
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

    out = round_dir(0)

    print("=" * 70)
    print("V0 vs V1 COMPARISON")
    print("=" * 70)

    for variant_path, tag in [
        ("server.strategy.evolved.v0_baseline", "v0"),
        ("server.strategy.evolved.v1a_filtered", "v1a"),
        ("server.strategy.evolved.v1b_recency", "v1b"),
    ]:
        print(f"\n--- {tag} ({variant_path}) ---")
        t0 = time.time()
        result = evaluate_variant(variant_path, config, out)
        dt = time.time() - t0
        print(f"  time: {dt:.1f}s")
        print(f"  fitness_train: {result.fitness_train:.3f}")
        print(f"  fitness_test:  {result.fitness_test:.3f}")
        print(f"  train: {_fmt_metrics(result.train)}")
        print(f"  test:  {_fmt_metrics(result.test)}")

        print(f"  per-touch (test):")
        by_touch = result.test.get("by_setup_touch", {})
        for k in sorted(by_touch.keys(), key=lambda x: int(x) if str(x).isdigit() else 99):
            t = by_touch[k]
            if t.get("n_setups_triggered", 0) == 0 and int(k) > 0:
                continue
            print(f"    touch={k}: {_fmt_metrics(t)}")

    print("\nDone. Summaries written to:", out)


if __name__ == "__main__":
    main()
