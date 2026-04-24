"""Round orchestrator: one round = draw → backtest → select → save.

Subsequent rounds tune SRParams toward the configs that produced the
highest-ranked winners. Hand-off between rounds is via JSONL artefacts
under data/evolve_rounds/round_NN/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..adapters import iter_legacy_pattern_records
from .backtest import backtest_lines
from .draw import draw_lines_for_symbol, _ohlcv_dataframe
from .select import aggregate_per_line, select_winners, summarize_round
from .seeds import build_round0_seeds


ROOT = Path(__file__).resolve().parents[2]
ROUNDS_DIR = ROOT / "data" / "evolve_rounds"


def load_seeds() -> dict:
    return build_round0_seeds(
        params_path=ROOT / "data" / "trendline_params.json",
        labels_path=ROOT / "data" / "user_drawing_labels.jsonl",
    )


def _symbol_sr_params(seeds: dict, symbol: str) -> dict:
    base = dict(seeds["sr_params"])
    over = seeds.get("per_symbol_sr_overrides", {}).get(symbol, {})
    if isinstance(over, dict):
        base.update({k: v for k, v in over.items() if k in base})
    return base


def _load_legacy_lines(symbols: list[str], timeframes: list[str],
                       max_lines_per_sym_tf: int) -> list:
    """Use the 271k existing auto rows in data/patterns/*.jsonl as the
    round-0 draw output. Lets the backtest loop run today without
    waiting on a real sr_patterns re-draw."""
    patterns_dir = ROOT / "data" / "patterns"
    if not patterns_dir.exists():
        return []
    out = []
    wanted = {(s.upper(), t) for s in symbols for t in timeframes}
    for sym_upper, tf in wanted:
        fname = patterns_dir / f"{sym_upper}_{tf}.jsonl"
        if not fname.exists():
            continue
        n = 0
        for rec in iter_legacy_pattern_records(fname):
            out.append(rec)
            n += 1
            if n >= max_lines_per_sym_tf:
                break
    return out


def run_round(
    round_id: int,
    symbols: list[str],
    timeframes: list[str],
    *,
    seeds: dict | None = None,
    max_lines_per_sym_tf: int = 100,
    max_bars_forward: int = 80,
    draw_source: str = "legacy",   # "legacy" | "sr_patterns"
) -> dict:
    """Execute one evolve round. Writes artefacts and returns summary.

    draw_source:
      - "legacy": pull candidates from data/patterns/*.jsonl (fast, reuses
                  the 271k auto rows you already have)
      - "sr_patterns": re-run detect_patterns live (requires the adapter
                  in draw.py to match the installed sr_patterns shape)
    """
    seeds = seeds or load_seeds()
    round_dir = ROUNDS_DIR / f"round_{round_id:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # 1. Draw
    all_lines = []
    ohlcv_cache: dict[tuple[str, str], pd.DataFrame] = {}

    if draw_source == "legacy":
        all_lines = _load_legacy_lines(symbols, timeframes, max_lines_per_sym_tf)
        for symbol in symbols:
            for tf in timeframes:
                df = _ohlcv_dataframe(symbol, tf)
                if df is not None and len(df) >= 50:
                    ohlcv_cache[(symbol, tf)] = df
    else:
        for symbol in symbols:
            for tf in timeframes:
                df = _ohlcv_dataframe(symbol, tf)
                if df is None or len(df) < 50:
                    continue
                ohlcv_cache[(symbol, tf)] = df
                params = _symbol_sr_params(seeds, symbol)
                lines = draw_lines_for_symbol(symbol, tf, params,
                                              max_lines=max_lines_per_sym_tf)
                all_lines.extend(lines)
    t_draw = time.time() - t0

    # 2. Backtest
    t1 = time.time()
    outcomes = backtest_lines(all_lines, ohlcv_cache, seeds["trade_configs"],
                              max_bars_forward=max_bars_forward)
    t_bt = time.time() - t1

    # 3. Select
    winners, stats = select_winners(outcomes, top_frac=0.2, min_score=0.3)

    # 4. Persist
    (round_dir / "config.json").write_text(
        json.dumps({"seeds": seeds, "symbols": symbols, "timeframes": timeframes}, indent=2),
        encoding="utf-8",
    )
    with (round_dir / "lines.jsonl").open("w", encoding="utf-8") as fh:
        for line in all_lines:
            fh.write(line.model_dump_json() + "\n")
    with (round_dir / "outcomes.jsonl").open("w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(json.dumps(o.as_dict()) + "\n")
    with (round_dir / "selected.jsonl").open("w", encoding="utf-8") as fh:
        for line in all_lines:
            if line.id in set(winners):
                fh.write(line.model_dump_json() + "\n")

    summary = {
        "round_id": round_id,
        "symbols": symbols,
        "timeframes": timeframes,
        "n_lines": len(all_lines),
        "n_outcomes": len(outcomes),
        "n_winners": len(winners),
        "t_draw_s": round(t_draw, 2),
        "t_backtest_s": round(t_bt, 2),
        **summarize_round(outcomes, stats),
    }
    (round_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
