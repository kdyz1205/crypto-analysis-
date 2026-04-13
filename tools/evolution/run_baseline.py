"""Run v0_baseline across the full symbol/TF matrix.

Usage:
    python -m tools.evolution.run_baseline

Outputs:
    data/evolution/rounds/round_00/v0_baseline_traces.jsonl
    data/evolution/rounds/round_00/v0_baseline_summary.json
    data/evolution/state.json (initial baseline snapshot)

This is Round 0 — the baseline everything else competes against.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .evaluator import EvalConfig, evaluate_variant
from .orchestrator import EvolutionState, round_dir, save_state


SYMBOL_POOL = [
    # majors
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    # mid-caps
    "HYPEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    # high-vol alts
    "DOGEUSDT", "PEPEUSDT", "SHIBUSDT", "WIFUSDT",
]

TIMEFRAMES = ["15m", "1h", "4h"]


def main():
    print("[evolution] starting baseline run (v0)")
    print(f"  symbols: {len(SYMBOL_POOL)}")
    print(f"  timeframes: {TIMEFRAMES}")

    t0 = time.time()
    config = EvalConfig(
        symbols=SYMBOL_POOL,
        timeframes=TIMEFRAMES,
        days=730,
        train_fraction=0.7,
        min_total_triggered=20,
    )

    out_dir = round_dir(0)
    result = evaluate_variant("server.strategy.evolved.v0_baseline", config, out_dir)

    elapsed = time.time() - t0
    print(f"\n[evolution] baseline complete in {elapsed:.1f}s")
    print(f"  fitness_train: {result.fitness_train:.3f}")
    print(f"  fitness_test:  {result.fitness_test:.3f}")
    print(f"  train: {result.train}")
    print(f"  test:  {result.test}")

    # Initialize state
    state = EvolutionState(
        current_baseline="v0_baseline",
        current_fitness_train=result.fitness_train,
        current_fitness_test=result.fitness_test,
    )
    save_state(state)

    print(f"\n[evolution] per-slice breakdown:")
    for s in result.per_slice:
        if "error" in s:
            print(f"  {s['symbol']:10s} {s.get('tf','?'):4s}  ERROR: {s['error']}")
            continue
        tr = s.get("train", {})
        te = s.get("test", {})
        print(f"  {s['symbol']:10s} {s['tf']:4s}  "
              f"train:{tr.get('n_setups_triggered',0):3d}/{tr.get('fade_win_rate',0):.2f}/R={tr.get('total_R',0):.2f}  "
              f"test:{te.get('n_setups_triggered',0):3d}/{te.get('fade_win_rate',0):.2f}/R={te.get('total_R',0):.2f}")

    # Per-touch-number summary (user's core question)
    print(f"\n[evolution] per-touch-number (test split):")
    by_touch = result.test.get("by_setup_touch", {})
    for k in sorted(by_touch.keys(), key=lambda x: int(x) if str(x).isdigit() else 99):
        t = by_touch[k]
        print(f"  touch={k}: n_trig={t.get('n_setups_triggered',0):3d}  "
              f"wr={t.get('fade_win_rate',0):.3f}  "
              f"avg_R={t.get('avg_total_R',0):.3f}  "
              f"total_R={t.get('total_R',0):.2f}")


if __name__ == "__main__":
    main()
