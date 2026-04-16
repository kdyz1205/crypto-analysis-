"""Tool 5: Backtest / Simulation — run strategies on historical data.
No silent failures. Every error is recorded."""

from __future__ import annotations
import asyncio, json, time, traceback
from pathlib import Path
from dataclasses import asdict
from .types import SimulationJob, SimulationResult, new_id
from .audit import write_audit
from . import market

RESULTS_DIR = Path(__file__).parent.parent / "data" / "strategies" / "simulated"
FAILED_DIR = Path(__file__).parent.parent / "data" / "strategies" / "failed"


async def run_simulation(job: SimulationJob) -> SimulationJob:
    """Run a simulation job with full error tracking."""
    from server.strategy.config import StrategyConfig, apply_strategy_overrides
    from server.strategy.replay import build_latest_snapshot
    import numpy as np

    job.status = "running"
    results = []
    failed_items = []

    write_audit("system", "simulation_started", "simulation", job.id,
                {"strategy_count": len(job.strategy_ids), "generation": job.generation})

    for sid in job.strategy_ids:
        draft_path = Path(__file__).parent.parent / "data" / "strategies" / "drafts" / f"{sid}.json"
        if not draft_path.exists():
            failed_items.append({"strategy_id": sid, "symbol": "-", "timeframe": "-", "stage": "draft_load", "error": "file not found"})
            continue

        try:
            draft = json.loads(draft_path.read_text(encoding="utf-8"))
        except Exception as e:
            failed_items.append({"strategy_id": sid, "symbol": "-", "timeframe": "-", "stage": "draft_parse", "error": str(e)})
            continue

        for symbol in draft.get("symbols", ["BTCUSDT"]):
            for tf in draft.get("timeframes", ["4h"]):
                try:
                    df = await market.get_ohlcv(symbol, tf, 90)
                    if df.empty or len(df) < 50:
                        failed_items.append({"strategy_id": sid, "symbol": symbol, "timeframe": tf, "stage": "market_data", "error": f"insufficient data ({len(df)} bars)"})
                        continue
                except Exception as e:
                    failed_items.append({"strategy_id": sid, "symbol": symbol, "timeframe": tf, "stage": "market_data", "error": str(e)[:100]})
                    continue

                try:
                    params = draft.get("params", {})
                    cfg = apply_strategy_overrides(
                        StrategyConfig(),
                        lookback_bars=params.get("lookback_bars"),
                        min_touches=params.get("min_touches"),
                        rr_target=params.get("rr_target"),
                    )

                    trades, equity = _walk_forward(df, cfg, symbol, tf, draft, job.capital)

                    if len(trades) < 3:
                        failed_items.append({"strategy_id": sid, "symbol": symbol, "timeframe": tf, "stage": "insufficient_trades", "error": f"only {len(trades)} trades"})
                        continue

                    result = _compute_metrics(trades, equity, job, sid, symbol, tf)
                    if result:
                        results.append(result)

                except Exception as e:
                    failed_items.append({"strategy_id": sid, "symbol": symbol, "timeframe": tf, "stage": "backtest_runtime", "error": str(e)[:100]})

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        (RESULTS_DIR / f"{r.id}.json").write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")

    # Save failed items to structured storage
    if failed_items:
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        for fi in failed_items:
            fi["job_id"] = job.id
            fi["generation"] = job.generation
            fi["batch_id"] = job.batch_id
            fi["failed_at"] = time.time()
            fid = new_id()
            (FAILED_DIR / f"{fid}.json").write_text(json.dumps(fi, indent=2), encoding="utf-8")
        write_audit("system", "simulation_failures_recorded", "simulation", job.id, {
            "count": len(failed_items),
            "stages": list(set(fi.get("stage", "") for fi in failed_items)),
            "generation": job.generation,
        })

    job.status = "completed"
    job.completed_at = time.time()
    job.results = [r.id for r in results]
    job.failed_items = failed_items

    profitable = sum(1 for r in results if r.return_pct > 0)
    write_audit("system", "simulation_completed", "simulation", job.id, {
        "results": len(results), "failed": len(failed_items), "profitable": profitable,
        "generation": job.generation,
    })

    return job


def _walk_forward(df, cfg, symbol, tf, draft, capital):
    """Walk-forward backtest with train/val/test split.

    Data split: 60% train | 20% validation | 20% test
    Returns (all_trades, final_equity, split_info).
    Each trade is tagged with its split phase.
    """
    from server.strategy.replay import build_latest_snapshot

    total_bars = len(df)
    train_end = int(total_bars * 0.6)
    val_end = int(total_bars * 0.8)

    trades = []
    equity = capital
    risk_per = draft.get("params", {}).get("risk_per_trade", 0.003)
    open_trade = None
    triggers = tuple(draft.get("trigger_modes", ["rejection", "failed_breakout"]))

    start_bar = max(80, int(total_bars * 0.1))  # need lookback

    for bar_end in range(start_bar, total_bars, 3):
        # Tag which split phase this bar belongs to
        if bar_end < train_end:
            phase = "train"
        elif bar_end < val_end:
            phase = "val"
        else:
            phase = "test"

        window = df.iloc[:bar_end + 1].reset_index(drop=True)
        high = float(window.iloc[-1]["high"])
        low = float(window.iloc[-1]["low"])

        if open_trade:
            if open_trade["dir"] == "long":
                if low <= open_trade["stop"]:
                    equity -= open_trade["risk"]; trades.append({"won": False, "pnl": -open_trade["risk"], "phase": phase}); open_trade = None
                elif high >= open_trade["tp"]:
                    rr = abs(open_trade["tp"] - open_trade["entry"]) / max(abs(open_trade["entry"] - open_trade["stop"]), 1e-10)
                    gain = open_trade["risk"] * rr; equity += gain; trades.append({"won": True, "pnl": gain, "rr": rr, "phase": phase}); open_trade = None
            else:
                if high >= open_trade["stop"]:
                    equity -= open_trade["risk"]; trades.append({"won": False, "pnl": -open_trade["risk"], "phase": phase}); open_trade = None
                elif low <= open_trade["tp"]:
                    rr = abs(open_trade["entry"] - open_trade["tp"]) / max(abs(open_trade["stop"] - open_trade["entry"]), 1e-10)
                    gain = open_trade["risk"] * rr; equity += gain; trades.append({"won": True, "pnl": gain, "rr": rr, "phase": phase}); open_trade = None

        if not open_trade and equity > 0:
            snapshot = build_latest_snapshot(window, cfg, symbol=symbol, timeframe=tf, enabled_trigger_modes=triggers)
            if snapshot.signals:
                sig = snapshot.signals[0]
                open_trade = {"entry": sig.entry_price, "stop": sig.stop_price, "tp": sig.tp_price, "dir": sig.direction, "risk": equity * risk_per}

        if equity <= 0:
            break

    return trades, equity


def _compute_metrics(trades, equity, job, sid, symbol, tf):
    """Compute performance metrics with train/val/test breakdown."""
    import numpy as np
    if not trades:
        return None

    # Overall metrics
    wins = sum(1 for t in trades if t["won"])
    total_return = (equity - job.capital) / job.capital * 100
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0) or 0.01
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)) or 0.01
    pf = gp / gl
    rets = [t["pnl"] / job.capital for t in trades]
    sharpe = (np.mean(rets) / max(np.std(rets), 1e-10)) * np.sqrt(len(trades))
    peak, max_dd, eq = job.capital, 0.0, job.capital
    for t in trades:
        eq += t["pnl"]; peak = max(peak, eq); max_dd = max(max_dd, (peak - eq) / peak * 100)
    avg_rr = np.mean([t.get("rr", 0) for t in trades if t.get("rr", 0) > 0]) if any(t.get("rr", 0) > 0 for t in trades) else 0
    score = 0.25 * min(sharpe / 2, 1) + 0.25 * min(pf / 3, 1) + 0.20 * (wins / len(trades)) + 0.15 * min(avg_rr / 5, 1) + 0.15 * max(0, 1 - max_dd / 20)

    # Per-split metrics
    def split_score(phase_trades):
        if len(phase_trades) < 2:
            return 0.0, 0.0
        w = sum(1 for t in phase_trades if t["won"])
        wr = w / len(phase_trades)
        ret = sum(t["pnl"] for t in phase_trades) / job.capital * 100
        r = [t["pnl"] / job.capital for t in phase_trades]
        sh = (np.mean(r) / max(np.std(r), 1e-10)) * np.sqrt(len(r))
        g = sum(t["pnl"] for t in phase_trades if t["pnl"] > 0) or 0.01
        l = abs(sum(t["pnl"] for t in phase_trades if t["pnl"] < 0)) or 0.01
        pf_s = g / l
        ar = np.mean([t.get("rr", 0) for t in phase_trades if t.get("rr", 0) > 0]) if any(t.get("rr") for t in phase_trades) else 0
        sc = 0.25 * min(sh / 2, 1) + 0.25 * min(pf_s / 3, 1) + 0.20 * wr + 0.15 * min(ar / 5, 1) + 0.15 * max(0, 1 - 5 / 20)
        return round(sc, 4), round(ret, 2)

    train_trades = [t for t in trades if t.get("phase") == "train"]
    val_trades = [t for t in trades if t.get("phase") == "val"]
    test_trades = [t for t in trades if t.get("phase") == "test"]

    ts, tr = split_score(train_trades)
    vs, vr = split_score(val_trades)
    tes, ter = split_score(test_trades)

    # Overfit detection
    overfit_flag = ""
    if ts > 0 and tes > 0:
        if ts > tes * 1.5 and ts > 0.3:
            overfit_flag = "overfit"
        elif abs(ts - tes) < ts * 0.2:
            overfit_flag = "stable"
        elif tes > ts:
            overfit_flag = "stable"
        else:
            overfit_flag = "degraded"

    return SimulationResult(
        job_id=job.id, strategy_id=sid, symbol=symbol, timeframe=tf,
        return_pct=round(total_return, 2), win_rate=round(wins / len(trades) * 100, 1),
        profit_factor=round(pf, 2), sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd, 2), trade_count=len(trades),
        avg_rr=round(float(avg_rr), 2), score=round(score, 4),
        train_score=ts, val_score=vs, test_score=tes,
        train_return=tr, val_return=vr, test_return=ter,
        overfit_flag=overfit_flag,
        generation=job.generation, batch_id=job.batch_id,
    )


def list_failures(generation: int = None, limit: int = 50) -> list[dict]:
    """List stored failure records. Optionally filter by generation."""
    if not FAILED_DIR.exists():
        return []
    failures = []
    for f in FAILED_DIR.glob("*.json"):
        try:
            item = json.loads(f.read_text(encoding="utf-8"))
            if generation is not None and item.get("generation") != generation:
                continue
            failures.append(item)
        except Exception:
            pass
    failures.sort(key=lambda x: x.get("failed_at", 0), reverse=True)
    return failures[:limit]


def run_simulation_sync(job: SimulationJob) -> SimulationJob:
    return asyncio.run(run_simulation(job))
