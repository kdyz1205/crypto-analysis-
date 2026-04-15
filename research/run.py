"""
MA Ribbon Strategy - Production Validation + Signal Scanner
Run: python -m research.run
"""

import gzip, csv, sys, os, numpy as np
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.strategy import Strategy, DEFAULT_CONFIG

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_ohlcv(symbol, tf):
    fn = f"data/{symbol}_{tf}.csv.gz"
    if not os.path.exists(fn): return None
    with gzip.open(fn, "rt") as f:
        reader = csv.DictReader(f); rows = list(reader)
    if len(rows) < 200: return None
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
    syms = []
    for f in glob.glob(f"data/*_{tf}.csv.gz"):
        sym = os.path.basename(f).replace(f"_{tf}.csv.gz", "")
        syms.append(sym)
    return sorted(syms)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    s = Strategy()
    tfs = ["15m", "1h", "4h"]

    # ── Part 1: Validation Backtest ──
    print("=" * 100)
    print("MA RIBBON + BB EXIT — PRODUCTION VALIDATION")
    print(f"Config: MA {DEFAULT_CONFIG['ma5_n']}/{DEFAULT_CONFIG['ma8_n']}/EMA{DEFAULT_CONFIG['ema21_n']}/{DEFAULT_CONFIG['ma55_n']}")
    print(f"        ADX>{DEFAULT_CONFIG['adx_min']} Vol>{DEFAULT_CONFIG['vol_mult']}x  ATR x{DEFAULT_CONFIG['atr_mult']}  BB({DEFAULT_CONFIG['bb_period']},{DEFAULT_CONFIG['bb_std']})")
    print(f"        Slope>{DEFAULT_CONFIG['slope_threshold']}%  Fanning>{DEFAULT_CONFIG['fanning_min_pct']}%")
    print("=" * 100)

    all_results = []
    for tf in tfs:
        syms = find_symbols(tf)
        for sym in syms:
            data = load_ohlcv(sym, tf)
            if data is None: continue
            o, h, l, c, v, ts = data
            bt = s.backtest(o, h, l, c, v)
            bt["sym"] = sym.upper()
            bt["tf"] = tf
            bt["bars"] = len(c)
            bt["period"] = f"{ts[0][:10]} to {ts[-1][:10]}"
            all_results.append(bt)

    # Print results table
    print(f"\n{'sym':<10} {'TF':<4} {'bars':>5} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>9} {'DD%':>6} {'avg_win':>8} {'avg_loss':>9} {'period'}")
    print("-" * 100)

    for r in sorted(all_results, key=lambda x: (x["tf"], x["sym"])):
        if r["trades"] < 2: continue
        print(f"{r['sym']:<10} {r['tf']:<4} {r['bars']:5d} {r['trades']:6d} {r['winrate']:5.1f}% "
              f"{r['sharpe']:+6.2f} {r['net_pct']:+8.1f}% {r['max_dd']:5.1f}% "
              f"{r['avg_win']:+7.2f}% {r['avg_loss']:+8.2f}% {r['period']}")

    # Aggregate by TF
    print("\n" + "=" * 100)
    print("AGGREGATE BY TIMEFRAME")
    print("=" * 100)
    for tf in tfs:
        tfr = [r for r in all_results if r["tf"] == tf and r["trades"] >= 2]
        if not tfr: continue
        n_pos = sum(1 for r in tfr if r["net_pct"] > 0)
        print(f"  {tf:>4}: {len(tfr):2d} symbols | "
              f"avg Sharpe {np.mean([r['sharpe'] for r in tfr]):+.2f} | "
              f"avg net {np.mean([r['net_pct'] for r in tfr]):+.1f}% | "
              f"avg win% {np.mean([r['winrate'] for r in tfr]):.1f}% | "
              f"avg DD {np.mean([r['max_dd'] for r in tfr]):.1f}% | "
              f"profitable {n_pos}/{len(tfr)}")

    # ── Part 2: Current Signal Scanner ──
    print("\n" + "=" * 100)
    print("CURRENT SIGNAL SCANNER")
    print("=" * 100)

    scanner_results = []
    for tf in tfs:
        syms = find_symbols(tf)
        for sym in syms:
            data = load_ohlcv(sym, tf)
            if data is None: continue
            o, h, l, c, v, ts = data
            state = s.current_state(o, h, l, c, v)
            state["sym"] = sym.upper()
            state["tf"] = tf
            state["last_bar"] = ts[-1]
            scanner_results.append(state)

    # Active signals (crossover on latest bar)
    active = [r for r in scanner_results if r["signal"] != 0]
    if active:
        print(f"\n  ACTIVE SIGNALS ({len(active)}):")
        for r in active:
            direction = "LONG" if r["signal"] == 1 else "SHORT"
            print(f"    {r['sym']:<10} {r['tf']:<4} {direction:<6} @ {r['close']:.4f}  "
                  f"SL={r['close'] - r['atr']*3:.4f if r['signal']==1 else r['close'] + r['atr']*3:.4f}  "
                  f"TP(BB)={r['bb_upper'] if r['signal']==1 else r['bb_lower']}  "
                  f"ADX={r['adx']:.1f}  fan={r['fanning']:.2f}%  [{r['last_bar']}]")
    else:
        print(f"\n  No active signals on latest bar")

    # Approaching signals (3/4 ribbon conditions met)
    approaching = [r for r in scanner_results if r.get("bull_conditions_met", 0) == 3 or r.get("bear_conditions_met", 0) == 3]
    if approaching:
        print(f"\n  APPROACHING ALIGNMENT ({len(approaching)} — 3/4 conditions met):")
        for r in approaching:
            if r.get("bull_conditions_met", 0) == 3:
                direction = "BULL"
                # Which condition is missing?
                ma5, ma8, e21, ma55 = r.get("ma5",0), r.get("ma8",0), r.get("ema21",0), r.get("ma55",0)
                missing = []
                if not (r["close"] > ma5): missing.append(f"close({r['close']:.2f}) > MA5({ma5:.2f})")
                if not (ma5 > ma8): missing.append(f"MA5({ma5:.2f}) > MA8({ma8:.2f})")
                if not (ma8 > e21): missing.append(f"MA8({ma8:.2f}) > EMA21({e21:.2f})")
                if not (e21 > ma55): missing.append(f"EMA21({e21:.2f}) > MA55({ma55:.2f})")
            else:
                direction = "BEAR"
                ma5, ma8, e21, ma55 = r.get("ma5",0), r.get("ma8",0), r.get("ema21",0), r.get("ma55",0)
                missing = []
                if not (r["close"] < ma5): missing.append(f"close({r['close']:.2f}) < MA5({ma5:.2f})")
                if not (ma5 < ma8): missing.append(f"MA5({ma5:.2f}) < MA8({ma8:.2f})")
                if not (ma8 < e21): missing.append(f"MA8({ma8:.2f}) < EMA21({e21:.2f})")
                if not (e21 < ma55): missing.append(f"EMA21({e21:.2f}) < MA55({ma55:.2f})")

            adx_str = f"ADX={r['adx']:.1f}" if r['adx'] else "ADX=N/A"
            fan_str = f"fan={r['fanning']:.2f}%" if r['fanning'] else "fan=N/A"
            print(f"    {r['sym']:<10} {r['tf']:<4} {direction:<5} {adx_str}  {fan_str}  "
                  f"missing: {', '.join(missing)}  [{r['last_bar']}]")

    # Ribbon already aligned (potential re-entry)
    aligned = [r for r in scanner_results
               if (r["bull_ribbon"] or r["bear_ribbon"]) and r["signal"] == 0 and r["adx_ok"]]
    if aligned:
        print(f"\n  RIBBON ALIGNED but no crossover ({len(aligned)} — watch for re-entry):")
        for r in aligned[:15]:
            direction = "BULL" if r["bull_ribbon"] else "BEAR"
            fan_str = f"fan={r['fanning']:.2f}%" if r['fanning'] else "fan=N/A"
            print(f"    {r['sym']:<10} {r['tf']:<4} {direction:<5} close={r['close']:.4f}  "
                  f"ADX={r['adx']:.1f}  {fan_str}  [{r['last_bar']}]")
        if len(aligned) > 15:
            print(f"    ... and {len(aligned)-15} more")

    print(f"\nCompleted: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
