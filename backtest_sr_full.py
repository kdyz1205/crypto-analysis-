"""
Full S/R Zone Strategy Backtest — Real data, real stats.
Walk-forward: at each bar, detect zones, generate signals, simulate fills.
Output: per-timeframe stats for calibrating stop sizes, Kelly, and leverage.
"""
import asyncio
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataclasses import dataclass
from server.strategy.pivots import detect_pivots
from server.strategy.zones import detect_horizontal_zones
from server.strategy.zone_signals import generate_zone_signals
from server.strategy.config import StrategyConfig, calculate_atr
from server.data_service import get_ohlcv_with_df
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
except ImportError:
    pass

SYMBOLS = ["HYPEUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "ADAUSDT"]
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "backtest_sr_results.json")


@dataclass
class Trade:
    symbol: str
    timeframe: str
    direction: str
    entry_price: float
    stop_price: float
    tp_price: float
    exit_price: float
    result: str  # "TP" or "SL"
    rr: float
    stop_pct: float
    zone_touches: int
    zone_strength: float
    pnl_pct: float


async def run_backtest():
    cfg = StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=500, min_rr_ratio=3.0)
    all_trades: list[Trade] = []

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                df, _ = await get_ohlcv_with_df(sym, tf, None, 365, history_mode="fast_window")
                pdf = df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
                pdf = pdf.rename(columns={"open_time": "timestamp"})
                pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(pd.Timestamp(v).timestamp()))
                for c in ("open", "high", "low", "close", "volume"):
                    pdf[c] = pd.to_numeric(pdf[c])
                pdf = pdf.reset_index(drop=True)

                n_bars = len(pdf)
                step = max(1, n_bars // 200)  # ~200 checkpoints per series
                sym_trades = 0

                for bar in range(200, n_bars, step):
                    window = pdf.iloc[:bar + 1].reset_index(drop=True)
                    pivots = detect_pivots(window, cfg)
                    zones = detect_horizontal_zones(window, pivots, cfg, symbol=sym, timeframe=tf, max_zones_per_side=3)
                    sigs = generate_zone_signals(window, zones, cfg, symbol=sym, timeframe=tf)

                    for sig in sigs:
                        zone = next((z for z in zones if z.zone_id == sig.line_id), None)
                        entry_hit = False
                        for fb in range(bar + 1, min(bar + 60, n_bars)):
                            lo = float(pdf.iloc[fb]["low"])
                            hi = float(pdf.iloc[fb]["high"])

                            if not entry_hit:
                                if sig.direction == "long" and lo <= sig.entry_price:
                                    entry_hit = True
                                elif sig.direction == "short" and hi >= sig.entry_price:
                                    entry_hit = True
                                continue

                            if sig.direction == "long":
                                if lo <= sig.stop_price:
                                    pnl_pct = (sig.stop_price - sig.entry_price) / sig.entry_price
                                    stop_pct = abs(sig.entry_price - sig.stop_price) / sig.entry_price
                                    all_trades.append(Trade(sym, tf, "long", sig.entry_price, sig.stop_price, sig.tp_price,
                                                           sig.stop_price, "SL", sig.risk_reward, stop_pct,
                                                           zone.touches if zone else 0, zone.strength if zone else 0, pnl_pct))
                                    sym_trades += 1
                                    break
                                if hi >= sig.tp_price:
                                    pnl_pct = (sig.tp_price - sig.entry_price) / sig.entry_price
                                    stop_pct = abs(sig.entry_price - sig.stop_price) / sig.entry_price
                                    all_trades.append(Trade(sym, tf, "long", sig.entry_price, sig.stop_price, sig.tp_price,
                                                           sig.tp_price, "TP", sig.risk_reward, stop_pct,
                                                           zone.touches if zone else 0, zone.strength if zone else 0, pnl_pct))
                                    sym_trades += 1
                                    break
                            else:
                                if hi >= sig.stop_price:
                                    pnl_pct = (sig.entry_price - sig.stop_price) / sig.entry_price
                                    stop_pct = abs(sig.stop_price - sig.entry_price) / sig.entry_price
                                    all_trades.append(Trade(sym, tf, "short", sig.entry_price, sig.stop_price, sig.tp_price,
                                                           sig.stop_price, "SL", sig.risk_reward, stop_pct,
                                                           zone.touches if zone else 0, zone.strength if zone else 0, pnl_pct))
                                    sym_trades += 1
                                    break
                                if lo <= sig.tp_price:
                                    pnl_pct = (sig.entry_price - sig.tp_price) / sig.entry_price
                                    stop_pct = abs(sig.stop_price - sig.entry_price) / sig.entry_price
                                    all_trades.append(Trade(sym, tf, "short", sig.entry_price, sig.stop_price, sig.tp_price,
                                                           sig.tp_price, "TP", sig.risk_reward, stop_pct,
                                                           zone.touches if zone else 0, zone.strength if zone else 0, pnl_pct))
                                    sym_trades += 1
                                    break

                print(f"  {sym:10s} {tf:4s}: {n_bars} bars, {sym_trades} trades", flush=True)
            except Exception as e:
                print(f"  {sym:10s} {tf:4s}: ERROR {e}", flush=True)

    # Analyze results
    print(f"\n{'='*80}")
    print(f"TOTAL TRADES: {len(all_trades)}")

    if not all_trades:
        print("NO TRADES")
        return

    # Per-timeframe breakdown
    report = {"total_trades": len(all_trades), "timeframes": {}, "symbols": {}}

    for tf in TIMEFRAMES:
        tf_trades = [t for t in all_trades if t.timeframe == tf]
        if not tf_trades:
            continue
        wins = [t for t in tf_trades if t.result == "TP"]
        losses = [t for t in tf_trades if t.result == "SL"]
        wr = len(wins) / len(tf_trades)
        avg_rr = np.mean([t.rr for t in tf_trades])
        avg_stop = np.mean([t.stop_pct for t in tf_trades]) * 100
        median_stop = np.median([t.stop_pct for t in tf_trades]) * 100
        avg_win_pnl = np.mean([t.pnl_pct for t in wins]) * 100 if wins else 0
        avg_loss_pnl = np.mean([t.pnl_pct for t in losses]) * 100 if losses else 0
        ev = wr * avg_rr - (1 - wr)
        kelly = max(0, (wr * avg_rr - (1 - wr)) / avg_rr) * 0.5  # half kelly

        print(f"\n--- {tf} ---")
        print(f"  Trades: {len(tf_trades)} | Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"  Win rate: {wr*100:.1f}%")
        print(f"  Avg RR: {avg_rr:.2f}")
        print(f"  Avg stop: {avg_stop:.3f}% | Median stop: {median_stop:.3f}%")
        print(f"  Avg win: +{avg_win_pnl:.3f}% | Avg loss: {avg_loss_pnl:.3f}%")
        print(f"  Expected value: {ev:.3f}R per trade")
        print(f"  Half-Kelly: {kelly*100:.2f}%")
        print(f"  Verdict: {'PROFITABLE' if ev > 0 else 'NOT PROFITABLE'}")

        report["timeframes"][tf] = {
            "trades": len(tf_trades), "wins": len(wins), "losses": len(losses),
            "win_rate": round(wr, 4), "avg_rr": round(avg_rr, 2),
            "avg_stop_pct": round(avg_stop, 4), "median_stop_pct": round(median_stop, 4),
            "ev_per_trade": round(ev, 4), "half_kelly": round(kelly, 4),
        }

    # By zone touches
    print(f"\n--- BY ZONE TOUCHES ---")
    for min_t in [2, 3, 4, 5]:
        filtered = [t for t in all_trades if t.zone_touches >= min_t]
        if not filtered:
            continue
        w = sum(1 for t in filtered if t.result == "TP")
        wr = w / len(filtered)
        rr = np.mean([t.rr for t in filtered])
        ev = wr * rr - (1 - wr)
        print(f"  touches >= {min_t}: {len(filtered)} trades, WR={wr*100:.1f}%, RR={rr:.1f}, EV={ev:.3f}R {'OK' if ev>0 else 'BAD'}")

    # Save report
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run_backtest())
