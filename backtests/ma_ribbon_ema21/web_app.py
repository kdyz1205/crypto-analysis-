"""MA Ribbon EMA21 backtest web panel — LIVE async version.

Endpoints:
  - GET  /                    -> the HTML panel
  - GET  /api/symbols         -> live Bitget USDT-perp universe (filtered by 24h vol)
  - POST /api/scan            -> live-fetch selected (symbols, TFs), run scan,
                                  return cohorts + gate result. Takes ~2-5 min for
                                  100+ symbols on first call; in-process cache (5 min)
                                  makes immediate re-runs near-instant.
  - GET  /api/cache           -> [legacy] list any CSV-cached pairs from prior runs
  - GET  /api/universe        -> [legacy] hardcoded recommended set from config

Run:
    python -m backtests.ma_ribbon_ema21.web_app --port 8765

Or via the launcher .bat / desktop shortcut.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backtests.ma_ribbon_ema21.data_loader import (
    DataLoaderConfig, csv_path, load_ohlcv_from_csv,
)
from backtests.ma_ribbon_ema21.data_loader_async import (
    AsyncLoaderConfig, FetchProgress,
    fetch_all_usdt_perp_symbols, fetch_universe_async,
)
from backtests.ma_ribbon_ema21.ma_alignment import AlignmentConfig
from backtests.ma_ribbon_ema21.phase1_engine import (
    scan_symbol_tf, Phase1Event,
)
from backtests.ma_ribbon_ema21.cohort_report import (
    aggregate_cohorts, CohortStats,
)
from backtests.ma_ribbon_ema21.acceptance_gate import evaluate_phase1_gate
from backtests.ma_ribbon_ema21.phase2_engine import (
    scan_universe_phase2, PortfolioSummary,
)
from backtests.ma_ribbon_ema21.trailing_backtest import TrailingMetrics


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
    pages_per_symbol: int = 5
    concurrency: int = 10
    alignment: dict[str, bool] | None = None


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
    fetch_seconds: float
    scan_seconds: float
    total_seconds: float


# ──────────────────────────── routes ────────────────────────────


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    html = (_WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/symbols")
async def api_symbols(
    min_volume: float = 1_000_000.0,
    product_types: str = "USDT-FUTURES,USDC-FUTURES,COIN-FUTURES",
) -> dict:
    """Live: full Bitget perp universe across the requested product types,
    filtered by 24h quote volume. product_types is a comma-separated list."""
    pt_tuple = tuple(s.strip() for s in product_types.split(",") if s.strip())
    cfg = AsyncLoaderConfig()
    async with httpx.AsyncClient() as client:
        syms = await fetch_all_usdt_perp_symbols(
            client, cfg,
            min_quote_volume_24h=min_volume,
            product_types=pt_tuple,
        )
    return {
        "symbols": syms,
        "count": len(syms),
        "min_volume_usd": min_volume,
        "product_types": list(pt_tuple),
    }


@app.get("/api/universe")
def api_universe() -> dict:
    """[Legacy] Recommended universe + TF list from config.phase1.json."""
    cfg_data = _load_default_config()
    return {
        "universe":   cfg_data.get("universe", []),
        "timeframes": cfg_data.get("timeframes", _TIMEFRAMES),
        "buckets":    cfg_data.get("distance_buckets", [
            [0.0, 0.005], [0.005, 0.01], [0.01, 0.02],
            [0.02, 0.04], [0.04, 0.07], [0.07, 1.0]
        ]),
    }


@app.get("/api/cache")
def api_cache() -> dict:
    """[Legacy] List CSV-cached (symbol, TF) pairs from earlier runs."""
    cfg = DataLoaderConfig()
    cache_dir = Path(cfg.cache_dir)
    if not cache_dir.exists():
        return {"available": []}
    out = []
    for p in sorted(cache_dir.glob("*.csv")):
        name = p.stem
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
            "symbol": sym, "tf": tf,
            "bars": bar_count, "size_kb": p.stat().st_size // 1024,
        })
    return {"available": out}


@app.post("/api/scan", response_model=ScanResponse)
async def api_scan(req: ScanRequest) -> ScanResponse:
    """Live fetch + scan + cohort + gate. The 'big one'.

    For 100+ symbols on 4 TFs at 5 pages each, expect 2-5 min on first call,
    then ~10s on repeat (in-process 5-min cache). 5-min cache means clicking
    'rescan' with different alignment / gate thresholds is fast — only the
    aggregate + gate re-runs, not the data fetch.
    """
    align_cfg = AlignmentConfig.from_dict(req.alignment) if req.alignment else AlignmentConfig.default()

    fetch_cfg = AsyncLoaderConfig(
        pages_per_symbol=int(req.pages_per_symbol),
        concurrency=int(req.concurrency),
    )

    # --- live fetch -------------------------------------------------------------
    fetch_start = time.time()
    progress = FetchProgress()
    data = await fetch_universe_async(
        symbols=req.symbols,
        tfs=req.tfs,
        cfg=fetch_cfg,
        progress=progress,
    )
    fetch_seconds = time.time() - fetch_start
    _LOG.info("fetch done: %d pairs, %.2fs (errors: %d)",
              len(data), fetch_seconds, len(progress.errors))

    # --- scan -------------------------------------------------------------------
    scan_start = time.time()
    all_events: list[Phase1Event] = []
    skipped: list[tuple[str, str]] = []
    for (sym, tf), df in data.items():
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
    scan_seconds = time.time() - scan_start

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
        fetch_seconds=round(fetch_seconds, 2),
        scan_seconds=round(scan_seconds, 2),
        total_seconds=round(fetch_seconds + scan_seconds, 2),
    )


# ──────────────────────────── Phase 2 (trailing SL) ────────────────────────────


class Phase2Request(BaseModel):
    symbols: list[str]
    tfs: list[str] = Field(default_factory=lambda: list(_TIMEFRAMES))
    buffer_pct: float = 0.015                  # 1.5% default
    train_pct: float = 0.70
    fee_per_side: float = 0.0005
    slippage_per_fill: float = 0.0001
    pages_per_symbol: int = 30
    concurrency: int = 12
    alignment: dict[str, bool] | None = None


class PairMetricsRow(BaseModel):
    symbol: str
    tf: str
    trades: int
    win_rate: float
    total_return_post_fee: float
    mean_return_post_fee: float
    profit_factor: float
    max_drawdown: float
    sharpe_per_trade: float
    avg_holding_bars: float
    avg_mae: float
    avg_mfe: float
    train_count: int
    test_count: int
    train_mean_return: float
    test_mean_return: float


class PortfolioBlock(BaseModel):
    total_trades: int
    total_wins: int
    win_rate: float
    portfolio_total_return_post_fee: float
    portfolio_max_drawdown: float
    portfolio_sharpe_per_trade: float
    portfolio_profit_factor: float
    avg_holding_bars: float
    symbols_with_positive_test_return: int
    symbols_total: int
    pairs_evaluated: int


class Phase2Response(BaseModel):
    portfolio: PortfolioBlock
    pairs: list[PairMetricsRow]
    skipped_pairs: list[tuple[str, str]]
    fetch_seconds: float
    scan_seconds: float
    total_seconds: float


@app.post("/api/phase2_scan", response_model=Phase2Response)
async def api_phase2_scan(req: Phase2Request) -> Phase2Response:
    """Live fetch + EMA21-buffer trailing SL backtest. Real trades, real PnL."""
    align_cfg = AlignmentConfig.from_dict(req.alignment) if req.alignment else AlignmentConfig.default()
    fetch_cfg = AsyncLoaderConfig(
        pages_per_symbol=int(req.pages_per_symbol),
        concurrency=int(req.concurrency),
    )

    fetch_start = time.time()
    progress = FetchProgress()
    data = await fetch_universe_async(
        symbols=req.symbols, tfs=req.tfs, cfg=fetch_cfg, progress=progress,
    )
    fetch_seconds = time.time() - fetch_start

    scan_start = time.time()
    skipped: list[tuple[str, str]] = [k for k, df in data.items() if df.empty]
    non_empty = {k: v for k, v in data.items() if not v.empty}
    per_pair, summary = scan_universe_phase2(
        data=non_empty,
        buffer_pct=float(req.buffer_pct),
        alignment_cfg=align_cfg,
        fee_per_side=req.fee_per_side,
        slippage_per_fill=req.slippage_per_fill,
        train_pct=req.train_pct,
    )
    scan_seconds = time.time() - scan_start

    rows = [PairMetricsRow(
        symbol=m.symbol, tf=m.tf, trades=m.trades,
        win_rate=m.win_rate,
        total_return_post_fee=m.total_return_post_fee,
        mean_return_post_fee=m.mean_return_post_fee,
        profit_factor=m.profit_factor if m.profit_factor != float("inf") else 999.0,
        max_drawdown=m.max_drawdown,
        sharpe_per_trade=m.sharpe_per_trade,
        avg_holding_bars=m.avg_holding_bars,
        avg_mae=m.avg_mae, avg_mfe=m.avg_mfe,
        train_count=m.train_count, test_count=m.test_count,
        train_mean_return=m.train_mean_return_post_fee,
        test_mean_return=m.test_mean_return_post_fee,
    ) for m in per_pair if m.trades > 0]

    return Phase2Response(
        portfolio=PortfolioBlock(
            total_trades=summary.total_trades,
            total_wins=summary.total_wins,
            win_rate=summary.win_rate,
            portfolio_total_return_post_fee=summary.portfolio_total_return_post_fee,
            portfolio_max_drawdown=summary.portfolio_max_drawdown,
            portfolio_sharpe_per_trade=summary.portfolio_sharpe_per_trade,
            portfolio_profit_factor=summary.portfolio_profit_factor
                if summary.portfolio_profit_factor != float("inf") else 999.0,
            avg_holding_bars=summary.avg_holding_bars,
            symbols_with_positive_test_return=summary.symbols_with_positive_test_return,
            symbols_total=summary.symbols_total,
            pairs_evaluated=summary.pairs_evaluated,
        ),
        pairs=rows,
        skipped_pairs=skipped,
        fetch_seconds=round(fetch_seconds, 2),
        scan_seconds=round(scan_seconds, 2),
        total_seconds=round(fetch_seconds + scan_seconds, 2),
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
