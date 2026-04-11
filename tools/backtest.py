"""Tool 5: Backtest / Simulation — run strategies on historical data."""

from __future__ import annotations
import asyncio, json, time
from pathlib import Path
from dataclasses import asdict
from .types import SimulationJob, SimulationResult, new_id
from . import market

RESULTS_DIR = Path(__file__).parent.parent / "data" / "strategies" / "simulated"


async def run_simulation(job: SimulationJob) -> SimulationJob:
    """Run a simulation job — backtest all strategies in the job."""
    from server.strategy.config import StrategyConfig, apply_strategy_overrides
    from server.strategy.replay import build_latest_snapshot

    job.status = "running"
    results = []

    for sid in job.strategy_ids:
        # Load strategy draft
        draft_path = Path(__file__).parent.parent / "data" / "strategies" / "drafts" / f"{sid}.json"
        if not draft_path.exists():
            continue
        draft = json.loads(draft_path.read_text(encoding="utf-8"))

        for symbol in draft.get("symbols", ["BTCUSDT"]):
            for tf in draft.get("timeframes", ["4h"]):
                try:
                    df = await market.get_ohlcv(symbol, tf, 90)
                    if df.empty or len(df) < 50:
                        continue

                    params = draft.get("params", {})
                    cfg = apply_strategy_overrides(
                        StrategyConfig(),
                        lookback_bars=params.get("lookback_bars"),
                        min_touches=params.get("min_touches"),
                        rr_target=params.get("rr_target"),
                    )

                    # Walk-forward backtest
                    trades = []
                    equity = job.capital
                    risk_per = params.get("risk_per_trade", 0.003)
                    open_trade = None
                    triggers = tuple(draft.get("trigger_modes", ["rejection", "failed_breakout"]))

                    for bar_end in range(80, len(df), 3):
                        window = df.iloc[:bar_end + 1].reset_index(drop=True)
                        close = float(window.iloc[-1]["close"])
                        high = float(window.iloc[-1]["high"])
                        low = float(window.iloc[-1]["low"])

                        if open_trade:
                            if open_trade["dir"] == "long":
                                if low <= open_trade["stop"]:
                                    equity -= open_trade["risk"]
                                    trades.append({"won": False, "pnl": -open_trade["risk"]})
                                    open_trade = None
                                elif high >= open_trade["tp"]:
                                    rr = abs(open_trade["tp"] - open_trade["entry"]) / max(abs(open_trade["entry"] - open_trade["stop"]), 1e-10)
                                    gain = open_trade["risk"] * rr
                                    equity += gain
                                    trades.append({"won": True, "pnl": gain, "rr": rr})
                                    open_trade = None
                            else:
                                if high >= open_trade["stop"]:
                                    equity -= open_trade["risk"]
                                    trades.append({"won": False, "pnl": -open_trade["risk"]})
                                    open_trade = None
                                elif low <= open_trade["tp"]:
                                    rr = abs(open_trade["entry"] - open_trade["tp"]) / max(abs(open_trade["stop"] - open_trade["entry"]), 1e-10)
                                    gain = open_trade["risk"] * rr
                                    equity += gain
                                    trades.append({"won": True, "pnl": gain, "rr": rr})
                                    open_trade = None

                        if not open_trade and equity > 0:
                            try:
                                snapshot = build_latest_snapshot(window, cfg, symbol=symbol, timeframe=tf, enabled_trigger_modes=triggers)
                                if snapshot.signals:
                                    sig = snapshot.signals[0]
                                    open_trade = {
                                        "entry": sig.entry_price, "stop": sig.stop_price, "tp": sig.tp_price,
                                        "dir": sig.direction, "risk": equity * risk_per,
                                    }
                            except Exception:
                                pass

                        if equity <= 0:
                            break

                    # Calculate metrics
                    if len(trades) >= 3:
                        import numpy as np
                        wins = sum(1 for t in trades if t["won"])
                        total_return = (equity - job.capital) / job.capital * 100
                        gp = sum(t["pnl"] for t in trades if t["pnl"] > 0) or 0.01
                        gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)) or 0.01
                        pf = gp / gl
                        rets = [t["pnl"] / job.capital for t in trades]
                        sharpe = (np.mean(rets) / max(np.std(rets), 1e-10)) * np.sqrt(len(trades))
                        peak, max_dd, eq = job.capital, 0.0, job.capital
                        for t in trades:
                            eq += t["pnl"]; peak = max(peak, eq); max_dd = max(max_dd, (peak-eq)/peak*100)

                        score = 0.25*min(sharpe/2,1) + 0.25*min(pf/3,1) + 0.20*(wins/len(trades)) + 0.15*min(2.0/5,1) + 0.15*max(0,1-max_dd/20)

                        result = SimulationResult(
                            job_id=job.id, strategy_id=sid, symbol=symbol, timeframe=tf,
                            return_pct=round(total_return, 2), win_rate=round(wins/len(trades)*100, 1),
                            profit_factor=round(pf, 2), sharpe_ratio=round(sharpe, 2),
                            max_drawdown_pct=round(max_dd, 2), trade_count=len(trades),
                            avg_rr=round(np.mean([t.get("rr",0) for t in trades if t.get("rr",0)>0]) if any(t.get("rr",0)>0 for t in trades) else 0, 2),
                            score=round(score, 4),
                        )
                        results.append(result)

                except Exception:
                    continue

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        (RESULTS_DIR / f"{r.id}.json").write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")

    job.status = "completed"
    job.completed_at = time.time()
    job.results = [r.id for r in results]
    return job


def run_simulation_sync(job: SimulationJob) -> SimulationJob:
    return asyncio.run(run_simulation(job))
