"""
MA Ribbon Strategy - Extended Ablation + Param Sweep + Combined Filters
7 symbols × 3 TFs × 10 variants × 6 param sets = 1260 backtests
Run: python research_backtest_v2.py
"""

import gzip, csv, sys, numpy as np
from datetime import datetime, timezone
from itertools import product

sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_ohlcv(symbol, tf):
    fn = f"data/{symbol}_{tf}.csv.gz"
    with gzip.open(fn, "rt") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    o = np.array([float(r["open"]) for r in rows])
    h = np.array([float(r["high"]) for r in rows])
    l = np.array([float(r["low"]) for r in rows])
    c = np.array([float(r["close"]) for r in rows])
    v = np.array([float(r["volume"]) for r in rows])
    return o, h, l, c, v

# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════

def sma(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if np.any(np.isnan(w)):
            continue
        out[i] = np.mean(w)
    return out

def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.full(len(x), np.nan)
    out[n-1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i-1]
    return out

def atr(h, l, c, n=14):
    prev_c = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    return sma(tr, n)

def adx_calc(h, l, c, n=14):
    prev_h, prev_l, prev_c = np.roll(h, 1), np.roll(l, 1), np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    dmp = np.where((h - prev_h) > (prev_l - l), np.maximum(h - prev_h, 0), 0.0)
    dmn = np.where((prev_l - l) > (h - prev_h), np.maximum(prev_l - l, 0), 0.0)
    dmp[0] = 0; dmn[0] = 0
    atr_n = sma(tr, n)
    dip = 100 * sma(dmp, n) / (atr_n + 1e-12)
    din = 100 * sma(dmn, n) / (atr_n + 1e-12)
    dx = 100 * np.abs(dip - din) / (dip + din + 1e-12)
    return sma(dx, n)

def bollinger_bands(c, n=20, std=2.0):
    mid = sma(c, n)
    s = np.full(len(c), np.nan)
    for i in range(n-1, len(c)):
        s[i] = np.std(c[i-n+1:i+1], ddof=0)
    return mid - std * s, mid, mid + std * s

def ma_slope(ma, lookback=5):
    slope = np.full(len(ma), np.nan)
    for i in range(lookback, len(ma)):
        if not np.isnan(ma[i]) and not np.isnan(ma[i-lookback]) and ma[i-lookback] != 0:
            slope[i] = (ma[i] - ma[i-lookback]) / ma[i-lookback] * 100
    return slope

def fanning_distance(ma5, ma8, ema21, ma55, c):
    dist = np.full(len(c), np.nan)
    for i in range(len(c)):
        if any(np.isnan([ma5[i], ma8[i], ema21[i], ma55[i]])):
            continue
        dist[i] = (abs(ma5[i]-ma8[i]) + abs(ma8[i]-ema21[i]) + abs(ema21[i]-ma55[i])) / (3*c[i]) * 100
    return dist

# ═══════════════════════════════════════════════════════════════
# PRE-COMPUTE ALL INDICATORS ONCE PER (symbol, tf, param_set)
# ═══════════════════════════════════════════════════════════════

def compute_indicators(o, h, l, c, v, p):
    ma5 = sma(c, p["ma5_n"])
    ma8 = sma(c, p["ma8_n"])
    e21 = ema(c, p["ema21_n"])
    ma55 = sma(c, p["ma55_n"])
    atr14 = atr(h, l, c, 14)
    adx14 = adx_calc(h, l, c, 14)
    vol_avg = sma(v, 20)

    bull = (c > ma5) & (ma5 > ma8) & (ma8 > e21) & (e21 > ma55)
    bear = (c < ma5) & (ma5 < ma8) & (ma8 < e21) & (e21 < ma55)
    adx_ok = adx14 > p["adx_min"]
    vol_ok = v > p["vol_mult"] * vol_avg

    # V0 signals
    n = len(c)
    sig_v0 = np.zeros(n)
    for i in range(1, n):
        if bull[i] and not bull[i-1] and adx_ok[i] and vol_ok[i]:
            sig_v0[i] = 1
        elif bear[i] and not bear[i-1] and adx_ok[i] and vol_ok[i]:
            sig_v0[i] = -1

    # Filter arrays
    slope5 = ma_slope(ma5, 5)
    slope8 = ma_slope(ma8, 5)
    slope21 = ma_slope(e21, 5)
    bb_lo, bb_mid, bb_up = bollinger_bands(c, 20, 2.0)
    fan = fanning_distance(ma5, ma8, e21, ma55, c)

    return {
        "sig_v0": sig_v0, "atr14": atr14,
        "slope5": slope5, "slope8": slope8, "slope21": slope21,
        "bb_lo": bb_lo, "bb_up": bb_up,
        "fan": fan,
    }

# ═══════════════════════════════════════════════════════════════
# FILTER APPLICATION (composable)
# ═══════════════════════════════════════════════════════════════

def apply_filters(sig_v0, c, ind, filters, p):
    """Apply a set of filters to V0 signals. filters is a set of strings."""
    sig = sig_v0.copy()
    n = len(c)

    for i in range(n):
        if sig[i] == 0:
            continue

        # Slope filter
        if "slope" in filters:
            thr = p.get("slope_threshold", 0.0)
            s5, s8, s21 = ind["slope5"][i], ind["slope8"][i], ind["slope21"][i]
            if sig[i] == 1:
                if np.isnan(s5) or np.isnan(s8) or np.isnan(s21) or s5 <= thr or s8 <= thr or s21 <= thr:
                    sig[i] = 0; continue
            elif sig[i] == -1:
                if np.isnan(s5) or np.isnan(s8) or np.isnan(s21) or s5 >= -thr or s8 >= -thr or s21 >= -thr:
                    sig[i] = 0; continue

        # BB overextension filter
        if "bb" in filters and sig[i] != 0:
            bb_max = p.get("bb_max_ext", 1.0)
            bl, bu = ind["bb_lo"][i], ind["bb_up"][i]
            if np.isnan(bl) or np.isnan(bu) or (bu - bl) <= 0:
                sig[i] = 0; continue
            pos = (c[i] - bl) / (bu - bl)
            if sig[i] == 1 and pos > bb_max:
                sig[i] = 0; continue
            elif sig[i] == -1 and pos < (1 - bb_max):
                sig[i] = 0; continue

        # Fanning filter
        if "fanning" in filters and sig[i] != 0:
            fan_min = p.get("fan_min_pct", 0.5)
            if np.isnan(ind["fan"][i]) or ind["fan"][i] < fan_min:
                sig[i] = 0; continue

    return sig

# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def backtest(signals, h, l, c, atr14, atr_mult=2.0, rr=2.0, fee=0.0005):
    pos = 0; entry = 0.0; sl = 0.0; tp = 0.0
    equity = 1.0; peak = 1.0; max_dd = 0.0
    returns = []; wins = 0; trades = 0
    n = len(c)

    for i in range(1, n):
        if pos != 0:
            if pos == 1:
                hit_sl = l[i] <= sl; hit_tp = h[i] >= tp
            else:
                hit_sl = h[i] >= sl; hit_tp = l[i] <= tp

            if hit_tp and hit_sl:
                if (pos == 1 and c[i] < entry) or (pos == -1 and c[i] > entry):
                    hit_tp = False
                else:
                    hit_sl = False

            if hit_tp:
                net = abs(tp - entry) / entry - fee * 2
                equity *= (1 + net); returns.append(net); wins += 1; trades += 1; pos = 0
            elif hit_sl:
                net = -abs(sl - entry) / entry - fee * 2
                equity *= (1 + net); returns.append(net); trades += 1; pos = 0

            if pos == 1 and not np.isnan(atr14[i]):
                sl = max(sl, c[i] - atr_mult * atr14[i])
            elif pos == -1 and not np.isnan(atr14[i]):
                sl = min(sl, c[i] + atr_mult * atr14[i])

            peak = max(peak, equity); max_dd = max(max_dd, (peak - equity) / peak)

        if pos == 0 and signals[i] != 0 and not np.isnan(atr14[i]):
            pos = int(signals[i]); entry = c[i]
            sd = atr_mult * atr14[i]
            sl = entry - sd if pos == 1 else entry + sd
            tp = entry + rr * sd if pos == 1 else entry - rr * sd

    if trades < 2:
        return {"net_pct": 0, "sharpe": 0, "winrate": 0, "trades": trades, "max_dd": 0}
    r = np.array(returns)
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(len(r))
    return {"net_pct": (equity-1)*100, "sharpe": sharpe,
            "winrate": wins/trades*100, "trades": trades, "max_dd": max_dd*100}

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "dogeusdt", "nearusdt", "hypeusdt", "myxusdt"]
TIMEFRAMES = ["15m", "1h", "4h"]

# Variant definitions: name -> set of filter names
VARIANTS = {
    "V0_core":          set(),
    "V1_slope":         {"slope"},
    "V2_bb":            {"bb"},
    "V3_fanning":       {"fanning"},
    "V1+V2":            {"slope", "bb"},
    "V1+V3":            {"slope", "fanning"},
    "V2+V3":            {"bb", "fanning"},
    "V1+V2+V3":         {"slope", "bb", "fanning"},
}

# Parameter sets to sweep
PARAM_SETS = [
    {   # P0: Default
        "label": "Default",
        "ma5_n": 5, "ma8_n": 8, "ema21_n": 21, "ma55_n": 55,
        "adx_min": 20, "vol_mult": 1.2, "atr_mult": 2.0, "rr": 2.0,
        "slope_threshold": 0.0, "bb_max_ext": 1.0, "fan_min_pct": 0.5,
    },
    {   # P1: Looser ADX + no vol filter
        "label": "LooseADX15+NoVol",
        "ma5_n": 5, "ma8_n": 8, "ema21_n": 21, "ma55_n": 55,
        "adx_min": 15, "vol_mult": 0.0, "atr_mult": 2.0, "rr": 2.0,
        "slope_threshold": 0.0, "bb_max_ext": 1.0, "fan_min_pct": 0.5,
    },
    {   # P2: Tighter SL + higher RR
        "label": "ATR1.5+RR3",
        "ma5_n": 5, "ma8_n": 8, "ema21_n": 21, "ma55_n": 55,
        "adx_min": 20, "vol_mult": 1.2, "atr_mult": 1.5, "rr": 3.0,
        "slope_threshold": 0.0, "bb_max_ext": 1.0, "fan_min_pct": 0.5,
    },
    {   # P3: Wider SL + lower RR (more wins, smaller wins)
        "label": "ATR3.0+RR1.5",
        "ma5_n": 5, "ma8_n": 8, "ema21_n": 21, "ma55_n": 55,
        "adx_min": 20, "vol_mult": 1.2, "atr_mult": 3.0, "rr": 1.5,
        "slope_threshold": 0.0, "bb_max_ext": 1.0, "fan_min_pct": 0.5,
    },
    {   # P4: Fib MAs
        "label": "FibMA_5/8/13/34",
        "ma5_n": 5, "ma8_n": 8, "ema21_n": 13, "ma55_n": 34,
        "adx_min": 20, "vol_mult": 1.2, "atr_mult": 2.0, "rr": 2.0,
        "slope_threshold": 0.0, "bb_max_ext": 1.0, "fan_min_pct": 0.5,
    },
    {   # P5: Strict ADX + strict fanning
        "label": "ADX25+Fan0.8",
        "ma5_n": 5, "ma8_n": 8, "ema21_n": 21, "ma55_n": 55,
        "adx_min": 25, "vol_mult": 1.2, "atr_mult": 2.0, "rr": 2.0,
        "slope_threshold": 0.1, "bb_max_ext": 0.9, "fan_min_pct": 0.8,
    },
]

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    total_combos = len(SYMBOLS) * len(TIMEFRAMES) * len(VARIANTS) * len(PARAM_SETS)
    print("=" * 95)
    print("MA RIBBON STRATEGY - EXTENDED SWEEP")
    print(f"  {len(SYMBOLS)} symbols x {len(TIMEFRAMES)} TFs x {len(VARIANTS)} variants x {len(PARAM_SETS)} param sets = {total_combos} backtests")
    print("=" * 95)

    all_results = []
    done = 0

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                o, h, l, c, v = load_ohlcv(sym, tf)
            except Exception as e:
                print(f"  SKIP {sym}/{tf}: {e}")
                continue

            for pi, p in enumerate(PARAM_SETS):
                # Compute indicators once per (sym, tf, param_set)
                ind = compute_indicators(o, h, l, c, v, p)

                for vname, filters in VARIANTS.items():
                    sig = apply_filters(ind["sig_v0"], c, ind, filters, p)
                    bt = backtest(sig, h, l, c, ind["atr14"], p["atr_mult"], p["rr"])
                    all_results.append({
                        "symbol": sym.upper(), "tf": tf,
                        "variant": vname, "params": p["label"],
                        "param_idx": pi, **bt
                    })
                    done += 1

            print(f"  {sym.upper():>10} {tf}  done ({done}/{total_combos})")

    # ═══════════════════════════════════════════════════════════
    # TOP 30 by Sharpe
    # ═══════════════════════════════════════════════════════════
    profitable = [r for r in all_results if r["trades"] >= 5 and r["sharpe"] > 0]
    profitable.sort(key=lambda r: r["sharpe"], reverse=True)

    print("\n" + "=" * 95)
    print(f"TOP 30 COMBOS BY SHARPE (out of {len(profitable)} profitable, {len(all_results)} total)")
    print("=" * 95)
    print(f"{'#':>3} {'symbol':<10} {'TF':<4} {'variant':<12} {'params':<18} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>8} {'maxDD%':>7}")
    print("-" * 95)
    for i, r in enumerate(profitable[:30], 1):
        print(f"{i:3d} {r['symbol']:<10} {r['tf']:<4} {r['variant']:<12} {r['params']:<18} "
              f"{r['trades']:6d} {r['winrate']:5.1f}% {r['sharpe']:+6.2f} {r['net_pct']:+7.1f}% {r['max_dd']:6.1f}%")

    # ═══════════════════════════════════════════════════════════
    # AGGREGATE: Best variant across all symbols/TFs
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("AGGREGATE BY VARIANT (avg across all symbols x TFs, default params only)")
    print("=" * 95)
    print(f"{'variant':<12} {'avg_trades':>10} {'win%':>6} {'sharpe':>7} {'net%':>8} {'maxDD%':>7} {'profitable':>11}")
    print("-" * 75)

    for vname in VARIANTS:
        vr = [r for r in all_results if r["variant"] == vname and r["param_idx"] == 0 and r["trades"] >= 2]
        if not vr:
            print(f"{vname:<12}  (no data)")
            continue
        n_pos = sum(1 for r in vr if r["net_pct"] > 0)
        print(f"{vname:<12} {np.mean([r['trades'] for r in vr]):10.1f} "
              f"{np.mean([r['winrate'] for r in vr]):5.1f}% "
              f"{np.mean([r['sharpe'] for r in vr]):+6.2f} "
              f"{np.mean([r['net_pct'] for r in vr]):+7.1f}% "
              f"{np.mean([r['max_dd'] for r in vr]):6.1f}% "
              f"{n_pos:>5}/{len(vr)}")

    # ═══════════════════════════════════════════════════════════
    # AGGREGATE: Best param set across all symbols/TFs
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("AGGREGATE BY PARAM SET (avg across all symbols x TFs, V0 only)")
    print("=" * 95)
    print(f"{'params':<18} {'avg_trades':>10} {'win%':>6} {'sharpe':>7} {'net%':>8} {'maxDD%':>7} {'profitable':>11}")
    print("-" * 80)

    for pi, p in enumerate(PARAM_SETS):
        pr = [r for r in all_results if r["param_idx"] == pi and r["variant"] == "V0_core" and r["trades"] >= 2]
        if not pr:
            print(f"{p['label']:<18}  (no data)")
            continue
        n_pos = sum(1 for r in pr if r["net_pct"] > 0)
        print(f"{p['label']:<18} {np.mean([r['trades'] for r in pr]):10.1f} "
              f"{np.mean([r['winrate'] for r in pr]):5.1f}% "
              f"{np.mean([r['sharpe'] for r in pr]):+6.2f} "
              f"{np.mean([r['net_pct'] for r in pr]):+7.1f}% "
              f"{np.mean([r['max_dd'] for r in pr]):6.1f}% "
              f"{n_pos:>5}/{len(pr)}")

    # ═══════════════════════════════════════════════════════════
    # BEST COMBO PER SYMBOL
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("BEST COMBO PER SYMBOL (highest Sharpe, min 5 trades)")
    print("=" * 95)
    print(f"{'symbol':<10} {'TF':<4} {'variant':<12} {'params':<18} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>8} {'maxDD%':>7}")
    print("-" * 90)

    for sym in SYMBOLS:
        sym_r = [r for r in all_results if r["symbol"] == sym.upper() and r["trades"] >= 5 and r["sharpe"] > 0]
        if not sym_r:
            print(f"{sym.upper():<10}  (no profitable combo)")
            continue
        sym_r.sort(key=lambda r: r["sharpe"], reverse=True)
        best = sym_r[0]
        print(f"{best['symbol']:<10} {best['tf']:<4} {best['variant']:<12} {best['params']:<18} "
              f"{best['trades']:6d} {best['winrate']:5.1f}% {best['sharpe']:+6.2f} {best['net_pct']:+7.1f}% {best['max_dd']:6.1f}%")

    # ═══════════════════════════════════════════════════════════
    # BEST COMBO PER TIMEFRAME
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("BEST COMBO PER TIMEFRAME (highest avg Sharpe across symbols, min 5 trades)")
    print("=" * 95)

    for tf in TIMEFRAMES:
        print(f"\n  --- {tf} ---")
        # For each (variant, param), compute avg sharpe across symbols
        combos = {}
        for vname in VARIANTS:
            for pi, p in enumerate(PARAM_SETS):
                tf_r = [r for r in all_results if r["tf"] == tf and r["variant"] == vname
                        and r["param_idx"] == pi and r["trades"] >= 5]
                if len(tf_r) < 3:  # need at least 3 symbols
                    continue
                avg_sharpe = np.mean([r["sharpe"] for r in tf_r])
                avg_net = np.mean([r["net_pct"] for r in tf_r])
                avg_wr = np.mean([r["winrate"] for r in tf_r])
                avg_dd = np.mean([r["max_dd"] for r in tf_r])
                avg_trades = np.mean([r["trades"] for r in tf_r])
                n_pos = sum(1 for r in tf_r if r["net_pct"] > 0)
                combos[(vname, p["label"])] = {
                    "sharpe": avg_sharpe, "net": avg_net, "wr": avg_wr,
                    "dd": avg_dd, "trades": avg_trades, "n_pos": n_pos, "n": len(tf_r)
                }

        if not combos:
            print("    (no combos with enough data)")
            continue

        top = sorted(combos.items(), key=lambda x: x[1]["sharpe"], reverse=True)[:5]
        print(f"  {'variant':<12} {'params':<18} {'avg_sharpe':>10} {'avg_net%':>9} {'avg_win%':>9} {'avg_DD%':>8} {'pos/total':>9}")
        for (vn, pl), d in top:
            print(f"  {vn:<12} {pl:<18} {d['sharpe']:+9.2f} {d['net']:+8.1f}% {d['wr']:8.1f}% {d['dd']:7.1f}% {d['n_pos']:>4}/{d['n']}")

    # ═══════════════════════════════════════════════════════════
    # FILTER IMPACT HEATMAP
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("FILTER IMPACT: avg Sharpe delta vs V0 (default params, across all symbols x TFs)")
    print("=" * 95)
    print(f"{'variant':<12} {'sharpe_delta':>13} {'net_delta':>11} {'trade_reduction':>16}")
    print("-" * 60)

    for vname in VARIANTS:
        if vname == "V0_core":
            continue
        deltas_s, deltas_n, deltas_t = [], [], []
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                v0 = [r for r in all_results if r["symbol"]==sym.upper() and r["tf"]==tf
                      and r["variant"]=="V0_core" and r["param_idx"]==0]
                vx = [r for r in all_results if r["symbol"]==sym.upper() and r["tf"]==tf
                      and r["variant"]==vname and r["param_idx"]==0]
                if v0 and vx and v0[0]["trades"] >= 2:
                    deltas_s.append(vx[0]["sharpe"] - v0[0]["sharpe"])
                    deltas_n.append(vx[0]["net_pct"] - v0[0]["net_pct"])
                    if v0[0]["trades"] > 0:
                        deltas_t.append((v0[0]["trades"] - vx[0]["trades"]) / v0[0]["trades"] * 100)

        if deltas_s:
            print(f"{vname:<12} {np.mean(deltas_s):+12.2f} {np.mean(deltas_n):+10.1f}% {np.mean(deltas_t):+15.1f}%")

    print(f"\nCompleted: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 95)


if __name__ == "__main__":
    main()
