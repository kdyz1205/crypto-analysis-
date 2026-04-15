"""
Trendline Bounce Strategy - Backtest + Param Sweep
Run: python -m research.trendline_backtest
"""

import gzip, csv, sys, os, numpy as np
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.trendline_strategy import backtest, DEFAULT_CONFIG

def load_ohlcv(symbol, tf):
    fn = f"data/{symbol}_{tf}.csv.gz"
    if not os.path.exists(fn): return None
    with gzip.open(fn, "rt") as f:
        reader = csv.DictReader(f); rows = list(reader)
    if len(rows) < 500: return None
    return (
        np.array([float(r["open"]) for r in rows]),
        np.array([float(r["high"]) for r in rows]),
        np.array([float(r["low"]) for r in rows]),
        np.array([float(r["close"]) for r in rows]),
        np.array([float(r["volume"]) for r in rows]),
        [r["open_time"] for r in rows],
    )

def find_symbols(tf):
    import glob
    return sorted(set(
        os.path.basename(f).replace(f"_{tf}.csv.gz", "")
        for f in glob.glob(f"data/*_{tf}.csv.gz")
    ))

# ═══════════════════════════════════════════════════════════════
# PARAM SETS
# ═══════════════════════════════════════════════════════════════

PARAM_SETS = [
    {"label": "A_default",  "swing_lookback":10, "buffer_pct":0.15, "sl_pct":0.3, "rr":3.0, "max_hold_bars":50,
     "approach_pct":1.0, "min_bars_between":20, "max_bars_between":500, "max_penetrations":2},
    {"label": "B_tight",    "swing_lookback":5,  "buffer_pct":0.10, "sl_pct":0.2, "rr":4.0, "max_hold_bars":30,
     "approach_pct":0.5, "min_bars_between":15, "max_bars_between":300, "max_penetrations":1},
    {"label": "C_wide",     "swing_lookback":15, "buffer_pct":0.20, "sl_pct":0.5, "rr":2.5, "max_hold_bars":80,
     "approach_pct":1.5, "min_bars_between":30, "max_bars_between":800, "max_penetrations":3},
    {"label": "D_aggressive","swing_lookback":7, "buffer_pct":0.05, "sl_pct":0.15,"rr":5.0, "max_hold_bars":20,
     "approach_pct":0.3, "min_bars_between":10, "max_bars_between":200, "max_penetrations":1},
    {"label": "E_conservative","swing_lookback":20,"buffer_pct":0.25,"sl_pct":0.4,"rr":2.0, "max_hold_bars":100,
     "approach_pct":2.0, "min_bars_between":40, "max_bars_between":1000,"max_penetrations":2},
]

SYMBOLS = ["btcusdt","ethusdt","solusdt","dogeusdt","nearusdt","hypeusdt",
           "adausdt","bnbusdt","linkusdt","pepeusdt","shibusdt","suiusdt",
           "xrpusdt","avaxusdt","dotusdt","wifusdt"]

TIMEFRAMES = ["15m", "1h", "4h"]


def main():
    total = len(SYMBOLS) * len(TIMEFRAMES) * len(PARAM_SETS)
    print("=" * 100)
    print("TRENDLINE BOUNCE STRATEGY - BACKTEST + PARAM SWEEP")
    print(f"  {len(SYMBOLS)} symbols x {len(TIMEFRAMES)} TFs x {len(PARAM_SETS)} param sets = {total}")
    print("=" * 100)

    results = []
    done = 0

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            data = load_ohlcv(sym, tf)
            if data is None:
                done += len(PARAM_SETS)
                continue
            o, h, l, c, v, ts = data

            for ps in PARAM_SETS:
                cfg = {**DEFAULT_CONFIG, **{k:v for k,v in ps.items() if k != "label"}}
                bt = backtest(o, h, l, c, v, cfg)
                results.append({
                    "sym": sym.upper(), "tf": tf, "par": ps["label"],
                    "bars": len(c), **{k:v for k,v in bt.items() if k != "trade_log"},
                })
                done += 1

            sys.stdout.write(f"\r  {done:4d}/{total} | {sym.upper():>10} {tf}  ")
            sys.stdout.flush()

    print(f"\n\n  Total: {done} backtests")

    # ═══════════════════════════════════════════════════════════
    # TOP 30 by Sharpe
    # ═══════════════════════════════════════════════════════════
    good = [r for r in results if r["trades"] >= 10 and r["sharpe"] > 0]
    good.sort(key=lambda r: r["sharpe"], reverse=True)

    print("\n" + "=" * 100)
    print(f"TOP 30 BY SHARPE (min 10 trades) — {len(good)} profitable / {len(results)} total")
    print("=" * 100)
    print(f"{'#':>3} {'sym':<10} {'TF':<4} {'params':<16} {'lines':>5} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>9} {'DD%':>6} {'avg_hold':>8}")
    print("-" * 90)
    for i, r in enumerate(good[:30], 1):
        print(f"{i:3d} {r['sym']:<10} {r['tf']:<4} {r['par']:<16} {r.get('lines_found',0):5d} {r['trades']:6d} "
              f"{r['winrate']:5.1f}% {r['sharpe']:+6.2f} {r['net_pct']:+8.1f}% {r['max_dd']:5.1f}% {r.get('avg_hold',0):7.1f}")

    # ═══════════════════════════════════════════════════════════
    # AGGREGATE BY PARAM SET
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("AGGREGATE BY PARAM SET")
    print("=" * 100)
    print(f"{'params':<16} {'avg_trades':>10} {'avg_win%':>9} {'avg_sharpe':>11} {'avg_net%':>10} {'avg_DD%':>9} {'pos/total':>10}")
    print("-" * 80)

    for ps in PARAM_SETS:
        pr = [r for r in results if r["par"] == ps["label"] and r["trades"] >= 2]
        if not pr:
            print(f"{ps['label']:<16}  (no data)")
            continue
        n_pos = sum(1 for r in pr if r["net_pct"] > 0)
        print(f"{ps['label']:<16} {np.mean([r['trades'] for r in pr]):10.1f} "
              f"{np.mean([r['winrate'] for r in pr]):8.1f}% "
              f"{np.mean([r['sharpe'] for r in pr]):+10.2f} "
              f"{np.mean([r['net_pct'] for r in pr]):+9.1f}% "
              f"{np.mean([r['max_dd'] for r in pr]):8.1f}% "
              f"{n_pos:>5}/{len(pr)}")

    # ═══════════════════════════════════════════════════════════
    # AGGREGATE BY TIMEFRAME
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("AGGREGATE BY TIMEFRAME (best param per symbol)")
    print("=" * 100)

    for tf in TIMEFRAMES:
        tfr = [r for r in results if r["tf"] == tf and r["trades"] >= 5]
        if not tfr:
            print(f"  {tf}: no data")
            continue
        # Best param per symbol
        best_per_sym = {}
        for r in tfr:
            if r["sym"] not in best_per_sym or r["sharpe"] > best_per_sym[r["sym"]]["sharpe"]:
                best_per_sym[r["sym"]] = r
        bests = list(best_per_sym.values())
        n_pos = sum(1 for r in bests if r["net_pct"] > 0)
        print(f"  {tf:>4}: {len(bests):2d} symbols | "
              f"avg Sharpe {np.mean([r['sharpe'] for r in bests]):+.2f} | "
              f"avg net {np.mean([r['net_pct'] for r in bests]):+.1f}% | "
              f"avg win% {np.mean([r['winrate'] for r in bests]):.1f}% | "
              f"profitable {n_pos}/{len(bests)}")

    # ═══════════════════════════════════════════════════════════
    # BEST PER SYMBOL
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("BEST COMBO PER SYMBOL (min 10 trades)")
    print("=" * 100)
    print(f"{'sym':<10} {'TF':<4} {'params':<16} {'lines':>5} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>9} {'DD%':>6}")
    print("-" * 80)

    for sym in SYMBOLS:
        sr = [r for r in results if r["sym"] == sym.upper() and r["trades"] >= 10 and r["sharpe"] > 0]
        if not sr:
            print(f"{sym.upper():<10} (no profitable combo)")
            continue
        sr.sort(key=lambda r: r["sharpe"], reverse=True)
        b = sr[0]
        print(f"{b['sym']:<10} {b['tf']:<4} {b['par']:<16} {b.get('lines_found',0):5d} {b['trades']:6d} "
              f"{b['winrate']:5.1f}% {b['sharpe']:+6.2f} {b['net_pct']:+8.1f}% {b['max_dd']:5.1f}%")

    # ═══════════════════════════════════════════════════════════
    # TRADE ANALYSIS on best combo
    # ═══════════════════════════════════════════════════════════
    if good:
        best = good[0]
        print(f"\n" + "=" * 100)
        print(f"TRADE ANALYSIS: {best['sym']} {best['tf']} {best['par']}")
        print("=" * 100)

        # Re-run to get trade log
        data = load_ohlcv(best["sym"].lower(), best["tf"])
        if data:
            o, h, l, c, v, ts = data
            ps = [p for p in PARAM_SETS if p["label"] == best["par"]][0]
            cfg = {**DEFAULT_CONFIG, **{k:v for k,v in ps.items() if k != "label"}}
            bt = backtest(o, h, l, c, v, cfg)

            print(f"  {'#':>3} {'side':<6} {'entry':>10} {'exit':>10} {'SL':>10} {'TP':>10} {'pnl%':>8} {'type':<8} {'hold':>5}")
            print("-" * 80)
            for i, t in enumerate(bt["trade_log"][:20], 1):
                print(f"  {i:3d} {t['side']:<6} {t['entry']:10.4f} {t['exit']:10.4f} "
                      f"{t['sl']:10.4f} {t['tp']:10.4f} {t['pnl_pct']:+7.2f}% {t['exit_type']:<8} {t['hold_bars']:5d}")
            if len(bt["trade_log"]) > 20:
                print(f"  ... and {len(bt['trade_log'])-20} more trades")

            # Exit type distribution
            types = {}
            for t in bt["trade_log"]:
                types[t["exit_type"]] = types.get(t["exit_type"], 0) + 1
            print(f"\n  Exit types: {types}")

    print(f"\nCompleted: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
