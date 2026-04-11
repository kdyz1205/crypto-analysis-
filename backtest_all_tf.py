"""
Backtest S/R zone strategy across ALL timeframes and symbols.
Goal: find which TFs are profitable so they can be added to BACKTEST_CALIBRATION.
"""
import asyncio
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from server.strategy.pivots import detect_pivots
from server.strategy.zones import detect_horizontal_zones
from server.strategy.zone_signals import generate_zone_signals
from server.strategy.config import StrategyConfig
from server.data_service import get_ohlcv_with_df
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
except ImportError:
    pass

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HYPEUSDT",
    "DOGEUSDT", "SUIUSDT", "ADAUSDT", "PEPEUSDT", "ZECUSDT",
    "ENAUSDT", "DASHUSDT", "RIVERUSDT", "ARIAUSDT", "TAOUSDT",
]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


async def run():
    cfg = StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=500, min_rr_ratio=3.0)
    results_by_tf = {}

    MIN_SUCCESSFUL_SYMBOLS = 10  # require at least 10 symbols to trust calibration
    MIN_TRADES_FOR_CALIBRATION = 15  # require at least 15 trades

    for tf in TIMEFRAMES:
        trades = []
        stops = []
        successful_symbols = 0
        failed_symbols = []
        for sym in SYMBOLS:
            try:
                df, _ = await get_ohlcv_with_df(sym, tf, None, 365, history_mode="fast_window")
                pdf = df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
                pdf = pdf.rename(columns={"open_time": "timestamp"})
                pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(pd.Timestamp(v).timestamp()))
                for c in ("open", "high", "low", "close", "volume"):
                    pdf[c] = pd.to_numeric(pdf[c])
                pdf = pdf.reset_index(drop=True)
                n = len(pdf)
                step = max(1, n // 150)

                for bar in range(200, n, step):
                    w = pdf.iloc[:bar + 1].reset_index(drop=True)
                    pivots = detect_pivots(w, cfg)
                    zones = detect_horizontal_zones(w, pivots, cfg, symbol=sym, timeframe=tf, max_zones_per_side=3)
                    sigs = generate_zone_signals(w, zones, cfg, symbol=sym, timeframe=tf)

                    for sig in sigs:
                        stop_pct = abs(sig.entry_price - sig.stop_price) / sig.entry_price
                        eh = False
                        for fb in range(bar + 1, min(bar + 60, n)):
                            lo = float(pdf.iloc[fb]["low"])
                            hi = float(pdf.iloc[fb]["high"])
                            if not eh:
                                if sig.direction == "long" and lo <= sig.entry_price:
                                    eh = True
                                elif sig.direction == "short" and hi >= sig.entry_price:
                                    eh = True
                                continue
                            if sig.direction == "long":
                                if lo <= sig.stop_price:
                                    trades.append(("L", sig.risk_reward))
                                    stops.append(stop_pct)
                                    break
                                if hi >= sig.tp_price:
                                    trades.append(("W", sig.risk_reward))
                                    stops.append(stop_pct)
                                    break
                            else:
                                if hi >= sig.stop_price:
                                    trades.append(("L", sig.risk_reward))
                                    stops.append(stop_pct)
                                    break
                                if lo <= sig.tp_price:
                                    trades.append(("W", sig.risk_reward))
                                    stops.append(stop_pct)
                                    break

                successful_symbols += 1
                print(f"  {sym:12s} {tf:4s} done ({n} bars)", flush=True)
            except Exception as e:
                failed_symbols.append(sym)
                print(f"  {sym:12s} {tf:4s} ERROR: {e}", flush=True)

        w = sum(1 for t in trades if t[0] == "W")
        l = sum(1 for t in trades if t[0] == "L")
        total = w + l

        if total == 0:
            print(f"\n--- {tf}: 0 trades ---\n", flush=True)
            results_by_tf[tf] = {"trades": 0, "verdict": "NO_DATA"}
            continue

        wr = w / total
        avg_rr = np.mean([t[1] for t in trades])
        ev = wr * avg_rr - (1 - wr)
        hk = max(0, (wr * avg_rr - (1 - wr)) / avg_rr) * 0.5 if avg_rr > 0 else 0
        med_stop = float(np.median(stops))
        avg_stop = float(np.mean(stops))

        # Sample completeness check
        sample_complete = (
            successful_symbols >= MIN_SUCCESSFUL_SYMBOLS
            and total >= MIN_TRADES_FOR_CALIBRATION
        )
        if ev > 0 and sample_complete:
            verdict = "PROFITABLE"
        elif ev > 0 and not sample_complete:
            verdict = "PROFITABLE_BUT_INCOMPLETE_SAMPLE"
        else:
            verdict = "NOT_PROFITABLE"

        print(f"\n--- {tf} ---", flush=True)
        print(f"  Symbols: {successful_symbols}/{len(SYMBOLS)} ok, {len(failed_symbols)} failed: {failed_symbols}", flush=True)
        print(f"  Trades: {total} | W: {w} | L: {l}", flush=True)
        print(f"  WR: {wr*100:.1f}% | RR: {avg_rr:.2f} | EV: {ev:.3f}R", flush=True)
        print(f"  Kelly: {hk*100:.2f}% | Med stop: {med_stop*100:.3f}% | Avg stop: {avg_stop*100:.3f}%", flush=True)
        print(f"  Sample complete: {sample_complete} (need >={MIN_SUCCESSFUL_SYMBOLS} symbols, >={MIN_TRADES_FOR_CALIBRATION} trades)", flush=True)
        print(f"  Verdict: {verdict}", flush=True)

        results_by_tf[tf] = {
            "trades": total, "wins": w, "losses": l,
            "successful_symbols": successful_symbols,
            "failed_symbols": failed_symbols,
            "sample_complete": sample_complete,
            "win_rate": round(wr, 4), "avg_rr": round(avg_rr, 2),
            "ev": round(ev, 4), "half_kelly": round(hk, 4),
            "median_stop_pct": round(med_stop, 6), "avg_stop_pct": round(avg_stop, 6),
            "verdict": verdict,
        }

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("CALIBRATION TABLE (only profitable TFs):", flush=True)
    print("BACKTEST_CALIBRATION = {", flush=True)
    for tf, r in results_by_tf.items():
        if r.get("verdict") == "PROFITABLE":
            print(f'    "{tf}": ({r["win_rate"]}, {r["avg_rr"]}, {r["ev"]}, {r["half_kelly"]}, {r["median_stop_pct"]}),', flush=True)
    print("}", flush=True)

    with open(os.path.join(os.path.dirname(__file__), "backtest_all_tf_results.json"), "w") as f:
        json.dump(results_by_tf, f, indent=2)
    print("\nSaved to backtest_all_tf_results.json", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
