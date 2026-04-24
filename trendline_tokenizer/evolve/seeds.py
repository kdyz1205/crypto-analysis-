"""Seed configurations from the user's own tuning history.

Pulls:
  - per-symbol SRParams overrides from data/trendline_params.json
  - per-line best trade configs from data/user_drawing_labels.jsonl
    (best_buffer_pct, best_rr, best_trailing_enabled)

These form the starting population for round 0 so we don't start from
factory defaults.
"""
from __future__ import annotations

import json
from pathlib import Path


DEFAULT_SEEDS = {
    "sr_params": {
        "atr_multiplier": 0.5,
        "line_break_factor": 0.3,
        "tolerance_percent": 0.005,
        "min_touches": 2,
        "zone_merge_factor": 1.5,
        "window_left": 1,
        "window_right": 5,
    },
    "trade_configs": [
        # Each dict is one buffer/rr combo to backtest every drawn line against.
        {"buffer_pct": 0.001, "rr": 2.0, "sl_tick_pct": 1e-5, "trailing": False},
        {"buffer_pct": 0.002, "rr": 2.0, "sl_tick_pct": 1e-5, "trailing": False},
        {"buffer_pct": 0.003, "rr": 3.0, "sl_tick_pct": 1e-5, "trailing": True},
        {"buffer_pct": 0.005, "rr": 4.0, "sl_tick_pct": 1e-5, "trailing": True},
    ],
}


def load_user_sr_params(path: Path) -> dict:
    """data/trendline_params.json → per-symbol SRParams override map."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_user_best_configs(labels_path: Path) -> list[dict]:
    """data/user_drawing_labels.jsonl → list of distinct best trade configs."""
    out: list[dict] = []
    seen = set()
    if not labels_path.exists():
        return out
    with labels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            cfg = (
                float(row.get("best_buffer_pct") or 0.0),
                float(row.get("best_rr") or 0.0),
                bool(row.get("best_trailing_enabled")),
            )
            if cfg in seen or cfg[1] <= 0:
                continue
            seen.add(cfg)
            out.append({
                "buffer_pct": cfg[0],
                "rr": cfg[1],
                "sl_tick_pct": 1e-5,
                "trailing": cfg[2],
                "source": "user_label",
            })
    return out


def build_round0_seeds(
    *,
    params_path: Path | None = None,
    labels_path: Path | None = None,
) -> dict:
    """Merge defaults with whatever the user already tuned."""
    seeds = {
        "sr_params": dict(DEFAULT_SEEDS["sr_params"]),
        "trade_configs": list(DEFAULT_SEEDS["trade_configs"]),
        "per_symbol_sr_overrides": {},
    }
    if params_path:
        seeds["per_symbol_sr_overrides"] = load_user_sr_params(params_path)
    if labels_path:
        user_cfgs = load_user_best_configs(labels_path)
        # Put user-sourced configs FIRST so they're tried as-is
        if user_cfgs:
            seeds["trade_configs"] = user_cfgs + seeds["trade_configs"]
    return seeds
