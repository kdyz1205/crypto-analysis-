"""MA Ribbon EMA21 backtest web panel.

A standalone FastAPI app that exposes:
  - GET  /                    -> the HTML panel
  - GET  /api/cache           -> list cached (symbol, TF) pairs in CSV cache
  - GET  /api/universe        -> recommended symbols from config.phase1.json
  - POST /api/fetch           -> prefetch CSVs for the given (symbols, TFs)
  - POST /api/scan            -> run the formation-event scan + cohort report
                                  on already-cached symbols and return results

Run:
    python -m backtests.ma_ribbon_ema21.web_app --port 8765

Or via the launcher .bat:
    scripts/launch_ma_ribbon_panel.bat
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backtests.ma_ribbon_ema21.data_loader import (
    DataLoaderConfig, csv_path, load_or_fetch, load_ohlcv_from_csv,
)
from backtests.ma_ribbon_ema21.ma_alignment import AlignmentConfig
from backtests.ma_ribbon_ema21.phase1_engine import (
    UniverseConfig, scan_universe, scan_symbol_tf, Phase1Event,
)
from backtests.ma_ribbon_ema21.cohort_report import (
    aggregate_cohorts, CohortStats,
)
from backtests.ma_ribbon_ema21.acceptance_gate import evaluate_phase1_gate


_LOG = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent
_WEB_DIR = _HERE / "web"
_DEFAULT_CONFIG = _HERE / "config.phase1.json"
_TIMEFRAMES = ["5m", "15m", "1h", "4h"]


def _load_default_config() -> dict[str, Any]:
    if _DEFAULT_CONFIG.exists():
        return json.loads(_DEFAULT_CONFIG.read_text())
    return {}


app = FastAPI(title="MA Ribbon EMA21 Backtest Panel")


# ──────────────────────────── request/response models ────────────────────────────


class FetchRequest(BaseModel):
    symbols: list[str]
    tfs: list[str] = Field(default_factory=lambda: list(_TIMEFRAMES))
    pages: int = 30


class FetchResponse(BaseModel):
    fetched: int
    skipped: int
    errors: list[str]


class ScanRequest(BaseModel):
    symbols: list[str]
    tfs: list[str] = Field(default_factory=lambda: list(_TIMEFRAMES))
    forward_horizons: list[int] = Field(default_factory=lambda: [5, 10, 20, 50])
    horizon_for_gate: int = 20
    train_pct: float = 0.70
    fee_per_side: float = 0.0005
    slippage_per_fill: float = 0.0001
    gate_threshold_pct: float = 0.01
    gate_min_symbol_pct: float = 0.30
    alignment: dict[str, bool] | None = None  # mirrors AlignmentConfig fields


class CohortRow(BaseModel):
    symbol: str
    tf: str
    bucket: str
    split: str
    count: int
    mean_post_fee: float
    median_post_fee: float
    win_rate: float
    worst_post_fee: float


class GateBlock(BaseModel):
    passed: bool
    threshold_pct: float
    min_symbol_pct: float
    horizon: int
    symbols_total: int
    symbols_passing: int
    passing_symbols: list[str]
    failing_symbols: list[str]
    reason: str


class ScanResponse(BaseModel):
    total_events: int
    cohorts: list[CohortRow]
    gate: GateBlock
    horizon: int
    skipped_pairs: list[tuple[str, str]]


# ──────────────────────────── routes ────────────────────────────


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    html = (_WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/cache")
def api_cache() -> dict:
    """List cached (symbol, TF) pairs in the CSV cache."""
    cfg = DataLoaderConfig()
    cache_dir = Path(cfg.cache_dir)
    if not cache_dir.exists():
        return {"available": []}
    out = []
    for p in sorted(cache_dir.glob("*.csv")):
        name = p.stem  # SYMBOL_TF
        if "_" not in name:
            continue
        sym, tf = name.rsplit("_", 1)
        if tf not in _TIMEFRAMES:
            continue
        try:
            with p.open("r") as f:
                bar_count = sum(1 for _ in f) - 1
        except OSError:
            bar_count = 0
        out.append({
            "symbol": sym,
            "tf": tf,
            "bars": bar_count,
            "size_kb": p.stat().st_size // 1024,
        })
    return {"available": out}


@app.get("/api/universe")
def api_universe() -> dict:
    """Return the recommended universe + TF list from config.phase1.json."""
    cfg_data = _load_default_config()
    return {
        "universe":   cfg_data.get("universe", []),
        "timeframes": cfg_data.get("timeframes", _TIMEFRAMES),
        "buckets":    cfg_data.get("distance_buckets", [
            [0.0, 0.005], [0.005, 0.01], [0.01, 0.02],
            [0.02, 0.04], [0.04, 0.07], [0.07, 1.0]
        ]),
    }


@app.post("/api/fetch", response_model=FetchResponse)
def api_fetch(req: FetchRequest) -> FetchResponse:
    cfg = DataLoaderConfig(bitget_pages_per_symbol=int(req.pages))
    fetched = 0
    skipped = 0
    errors: list[str] = []
    for sym in req.symbols:
        for tf in req.tfs:
            cp = csv_path(sym, tf, cfg)
            if cp.exists() and cp.stat().st_size > 100:
                skipped += 1
                continue
            try:
                df = load_or_fetch(sym, tf, cfg)
                if df.empty:
                    errors.append(f"{sym} {tf}: empty result")
                else:
                    fetched += 1
            except Exception as exc:  # noqa: BLE001 — surface to caller per PRINCIPLES P10
                errors.append(f"{sym} {tf}: {exc}")
    return FetchResponse(fetched=fetched, skipped=skipped, errors=errors)


@app.post("/api/scan", response_model=ScanResponse)
def api_scan(req: ScanRequest) -> ScanResponse:
    cfg = DataLoaderConfig()
    align_cfg = AlignmentConfig.from_dict(req.alignment) if req.alignment else AlignmentConfig.default()

    all_events: list[Phase1Event] = []
    skipped: list[tuple[str, str]] = []
    for sym in req.symbols:
        for tf in req.tfs:
            df = load_ohlcv_from_csv(sym, tf, cfg)
            if df.empty:
                skipped.append((sym, tf))
                continue
            events = scan_symbol_tf(
                df, symbol=sym, tf=tf,
                alignment_cfg=align_cfg,
                forward_horizons=tuple(req.forward_horizons),
                fee_per_side=req.fee_per_side,
                slippage_per_fill=req.slippage_per_fill,
                train_pct=req.train_pct,
            )
            all_events.extend(events)

    cohorts: list[CohortStats] = aggregate_cohorts(all_events, horizon=req.horizon_for_gate)
    gate = evaluate_phase1_gate(
        cohorts, horizon=req.horizon_for_gate,
        threshold_pct=req.gate_threshold_pct,
        min_symbol_pct=req.gate_min_symbol_pct,
    )

    return ScanResponse(
        total_events=len(all_events),
        cohorts=[CohortRow(
            symbol=c.symbol, tf=c.tf, bucket=c.bucket, split=c.split,
            count=c.count,
            mean_post_fee=c.mean_return_post_fee,
            median_post_fee=c.median_return_post_fee,
            win_rate=c.win_rate,
            worst_post_fee=c.worst_return_post_fee,
        ) for c in cohorts],
        gate=GateBlock(
            passed=gate.passed,
            threshold_pct=gate.threshold_pct,
            min_symbol_pct=gate.min_symbol_pct,
            horizon=gate.horizon,
            symbols_total=gate.symbols_total,
            symbols_passing=gate.symbols_passing,
            passing_symbols=gate.passing_symbols,
            failing_symbols=gate.failing_symbols,
            reason=gate.reason,
        ),
        horizon=req.horizon_for_gate,
        skipped_pairs=skipped,
    )


# ──────────────────────────── entrypoint ────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"\n  MA Ribbon EMA21 Panel: http://{args.host}:{args.port}/\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
