"""Self-evolving agent loop.

Run as:
    python -m trendline_tokenizer.agent.loop \\
        --symbols BTCUSDT ETHUSDT HYPEUSDT \\
        --timeframes 5m 15m 1h 4h \\
        --interval-seconds 1800 \\
        --feedback-threshold 5 \\
        --max-iterations 200

One iteration:
    1. Snapshot agent state (last seen ts, last artifact)
    2. Check feedback store for new events since last_run_ts
    3. Check manual_trendlines.json count delta
    4. If new_feedback >= threshold OR new_manual > 0:
         - run train_fusion subprocess
         - record new artifact name
    5. Load latest manifest -> InferenceService
    6. For each (symbol, tf) in watch_list:
         - load OHLCV
         - auto-draw lines (auto_drawer.auto_draw)
         - backtest the kept lines (backtest.run_backtest)
    7. Aggregate metrics into IterationReport, append to JSONL
    8. Save state, sleep until next interval

Stops on Ctrl-C; resumes from saved state.
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..feedback.store import FeedbackStore
from ..inference.inference_service import InferenceService
from ..registry.manifest import ArtifactManifest
from ..registry.paths import ROOT, latest_fusion
from .auto_drawer import auto_draw
from .report import IterationReport
from .state import AgentState


DEFAULT_FEEDBACK_PATH = ROOT / "data" / "trendline_feedback.jsonl"
DEFAULT_MANUAL_PATH = ROOT / "data" / "manual_trendlines.json"


def count_manual_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        for k in ("drawings", "lines", "items"):
            if isinstance(data.get(k), list):
                return len(data[k])
    except Exception:
        return 0
    return 0


def load_ohlcv(symbol: str, tf: str):
    from ..evolve.draw import _ohlcv_dataframe
    return _ohlcv_dataframe(symbol, tf)


def maybe_retrain(state: AgentState, *, threshold: int, symbols: list[str],
                  timeframes: list[str], force: bool = False) -> tuple[bool, str]:
    """Run train_fusion subprocess if conditions are met.

    Returns (retrained, new_artifact_or_empty).
    """
    fb_count = 0
    if DEFAULT_FEEDBACK_PATH.exists():
        store = FeedbackStore(DEFAULT_FEEDBACK_PATH)
        for ev in store:
            if getattr(ev, "created_at", 0) > state.last_train_ts:
                fb_count += 1
    manual_count = count_manual_lines(DEFAULT_MANUAL_PATH)
    new_manual = max(0, manual_count - state.last_seen_manual_count)

    print(f"[agent] new_feedback={fb_count} new_manual={new_manual} "
          f"threshold={threshold} force={force}")

    if not force and fb_count < threshold and new_manual == 0:
        return False, ""

    cmd = [
        sys.executable, "-u", "-m", "trendline_tokenizer.training.train_fusion",
        "--symbols", *symbols,
        "--timeframes", *timeframes,
        "--max-records", "20000",
        "--epochs", "2",
        "--batch-size", "32",
        "--d-model", "96",
        "--manual-oversample", "50",
    ]
    print(f"[agent] retrain cmd: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        return False, ""
    state.last_train_ts = int(time.time())
    state.last_seen_manual_count = manual_count
    state.n_retrain_triggered += 1
    art_dir = latest_fusion("fusion.v0.1")
    if art_dir is None:
        return True, ""
    return True, art_dir.name


def iteration_step(
    *,
    state: AgentState,
    symbols: list[str],
    timeframes: list[str],
    feedback_threshold: int,
    force_retrain: bool = False,
) -> IterationReport:
    rep = IterationReport(iteration=state.iteration, started_at=int(time.time()))

    # 1. (Maybe) retrain.
    retrained, new_artifact = False, ""
    try:
        retrained, new_artifact = maybe_retrain(
            state, threshold=feedback_threshold,
            symbols=symbols, timeframes=timeframes,
            force=force_retrain,
        )
    except Exception as exc:
        rep.errors.append(f"retrain failed: {exc}")
    rep.retrained = retrained
    rep.new_artifact = new_artifact

    # 2. Load the latest model.
    art_dir = latest_fusion("fusion.v0.1")
    if art_dir is None:
        rep.errors.append("no fusion artifact available")
        rep.append()
        return rep
    manifest_path = art_dir / "manifest.json"
    if not manifest_path.exists():
        rep.errors.append(f"manifest missing at {manifest_path}")
        rep.append()
        return rep
    state.last_train_artifact = art_dir.name

    try:
        service = InferenceService(manifest_path, device="cpu")
    except Exception as exc:
        rep.errors.append(f"InferenceService load failed: {exc}")
        rep.append()
        return rep

    # 3. Auto-draw + backtest per (symbol, tf).
    backtest_by_pair = {}
    for sym in symbols:
        for tf in timeframes:
            try:
                df = load_ohlcv(sym, tf)
                if df is None or len(df) < 200:
                    continue
                kept, out_path = auto_draw(
                    service=service, df=df, symbol=sym, timeframe=tf,
                    min_confidence=0.55, keep_top_k=10,
                )
                rep.n_lines_auto_drawn += len(kept)
                rep.n_symbols_processed += 1

                if kept:
                    bt_metrics = _quick_backtest_kept_lines(kept, df, tf)
                    backtest_by_pair[f"{sym}_{tf}"] = bt_metrics
            except Exception as exc:
                rep.errors.append(f"{sym} {tf}: {exc}")
    rep.backtest_summary = backtest_by_pair
    state.n_lines_auto_drawn_total += rep.n_lines_auto_drawn
    rep.append()
    return rep


def _quick_backtest_kept_lines(lines, df, tf) -> dict:
    """Cheap backtest for the agent's kept lines: forward 20 bars from
    each line's end, compute outcome (bounced/broke/no-edge)."""
    from ..training.sequence_dataset import _label_outcomes
    n = len(lines)
    bounce = brk = cont = 0
    for r in lines:
        out = _label_outcomes(r, df, horizon_bars=20)
        bounce += out["bounce"]; brk += out["brk"]; cont += out["cont"]
    return {
        "n_lines": n,
        "bounce_pct": bounce / max(1, n),
        "break_pct": brk / max(1, n),
        "continuation_pct": cont / max(1, n),
    }


def run_loop(
    *,
    symbols: list[str],
    timeframes: list[str],
    interval_seconds: int,
    feedback_threshold: int,
    max_iterations: int = 0,
    force_retrain_first: bool = False,
):
    state = AgentState.load()
    print(f"[agent] resuming. iteration={state.iteration} "
          f"last_train={state.last_train_artifact}")
    stop = {"flag": False}

    def on_sig(*_):
        stop["flag"] = True
        print("[agent] caught signal, will stop after current iteration")

    try:
        signal.signal(signal.SIGINT, on_sig)
        signal.signal(signal.SIGTERM, on_sig)
    except Exception:
        pass

    while not stop["flag"]:
        state.iteration += 1
        force = force_retrain_first and state.iteration == 1
        print(f"[agent] === iteration {state.iteration} ===")
        rep = iteration_step(
            state=state, symbols=symbols, timeframes=timeframes,
            feedback_threshold=feedback_threshold,
            force_retrain=force,
        )
        print(f"[agent] iter {state.iteration} done: "
              f"retrained={rep.retrained} drawn={rep.n_lines_auto_drawn} "
              f"errors={len(rep.errors)}")
        state.last_run_ts = int(time.time())
        state.save()

        if max_iterations and state.iteration >= max_iterations:
            print(f"[agent] reached max_iterations={max_iterations}")
            break
        if stop["flag"]:
            break

        # sleep in 1s slices so signals can interrupt
        slept = 0
        while slept < interval_seconds and not stop["flag"]:
            time.sleep(1)
            slept += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--interval-seconds", type=int, default=1800)
    ap.add_argument("--feedback-threshold", type=int, default=5)
    ap.add_argument("--max-iterations", type=int, default=0,
                    help="0 = run until killed")
    ap.add_argument("--force-retrain-first", action="store_true",
                    help="run a retrain on the very first iteration")
    args = ap.parse_args()
    run_loop(
        symbols=args.symbols, timeframes=args.timeframes,
        interval_seconds=args.interval_seconds,
        feedback_threshold=args.feedback_threshold,
        max_iterations=args.max_iterations,
        force_retrain_first=args.force_retrain_first,
    )


if __name__ == "__main__":
    main()
