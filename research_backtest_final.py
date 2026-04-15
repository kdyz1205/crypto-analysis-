"""
MA Ribbon Strategy - Final Extended Sweep
25 symbols × 4 TFs × best filter combos × best param sets
Focus on top-performing combinations from round 2
Run: python research_backtest_final.py
"""

import gzip, csv, sys, os, numpy as np
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_ohlcv(symbol, tf):
    fn = f"data/{symbol}_{tf}.csv.gz"
    if not os.path.exists(fn):
        return None
    with gzip.open(fn, "rt") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) < 200:
        return None
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
    # Fast path: if no NaN in input, use cumsum
    if not np.any(np.isnan(x)):
        cs = np.cumsum(x)
        out[n-1:] = (cs[n-1:] - np.concatenate([[0], cs[:-n]])) / n
        return out
    # Slow path: NaN-aware
    nan_mask = np.isnan(x)
    nan_cs = np.cumsum(nan_mask)
    x_filled = np.where(nan_mask, 0, x)
    cs = np.cumsum(x_filled)
    for i in range(n-1, len(x)):
        nans_in_window = nan_cs[i] - (nan_cs[i-n] if i >= n else 0)
        if nans_in_window == 0:
            out[i] = (cs[i] - (cs[i-n] if i >= n else 0)) / n
    return out

def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.full(len(x), np.nan)
    out[n-1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i-1]
    return out

def atr(h, l, c, n=14):
    prev_c = np.roll(c, 1); tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]; return sma(tr, n)

def adx_calc(h, l, c, n=14):
    ph, pl, pc = np.roll(h,1), np.roll(l,1), np.roll(c,1)
    tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc))); tr[0]=h[0]-l[0]
    dmp = np.where((h-ph)>(pl-l), np.maximum(h-ph,0), 0.0); dmp[0]=0
    dmn = np.where((pl-l)>(h-ph), np.maximum(pl-l,0), 0.0); dmn[0]=0
    an = sma(tr,n); dip = 100*sma(dmp,n)/(an+1e-12); din = 100*sma(dmn,n)/(an+1e-12)
    dx = 100*np.abs(dip-din)/(dip+din+1e-12); return sma(dx,n)

def bollinger_bands(c, n=20, std_mult=2.0):
    mid = sma(c, n)
    # Vectorized std: var = E[x^2] - (E[x])^2
    cs = np.cumsum(c); cs2 = np.cumsum(c**2)
    s = np.full(len(c), np.nan)
    for i in range(n-1, len(c)):
        sm = cs[i] - (cs[i-n] if i>=n else 0)
        sm2 = cs2[i] - (cs2[i-n] if i>=n else 0)
        var = sm2/n - (sm/n)**2
        s[i] = np.sqrt(max(var, 0))
    return mid-std_mult*s, mid, mid+std_mult*s

def ma_slope(ma, lb=5):
    s = np.full(len(ma), np.nan)
    shifted = np.roll(ma, lb)
    shifted[:lb] = np.nan
    valid = ~np.isnan(ma) & ~np.isnan(shifted) & (shifted != 0)
    s[valid] = (ma[valid] - shifted[valid]) / shifted[valid] * 100
    return s

def fanning_dist(ma5, ma8, e21, ma55, c):
    valid = ~(np.isnan(ma5)|np.isnan(ma8)|np.isnan(e21)|np.isnan(ma55))
    d = np.full(len(c), np.nan)
    d[valid] = (np.abs(ma5[valid]-ma8[valid])+np.abs(ma8[valid]-e21[valid])+np.abs(e21[valid]-ma55[valid]))/(3*c[valid])*100
    return d

# ═══════════════════════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════════════════════

def compute_and_backtest(o, h, l, c, v, p, filters):
    """Compute indicators + apply filters + backtest in one call"""
    ma5 = sma(c, p["ma5_n"]); ma8 = sma(c, p["ma8_n"])
    e21 = ema(c, p["ema21_n"]); ma55 = sma(c, p["ma55_n"])
    atr14 = atr(h, l, c, 14); adx14 = adx_calc(h, l, c, 14); vol_avg = sma(v, 20)

    bull = (c>ma5)&(ma5>ma8)&(ma8>e21)&(e21>ma55)
    bear = (c<ma5)&(ma5<ma8)&(ma8<e21)&(e21<ma55)
    adx_ok = adx14 > p["adx_min"]
    vol_ok = (v > p["vol_mult"]*vol_avg) if p["vol_mult"] > 0 else np.ones(len(c), bool)

    n = len(c)
    sig = np.zeros(n)
    for i in range(1, n):
        if bull[i] and not bull[i-1] and adx_ok[i] and vol_ok[i]: sig[i] = 1
        elif bear[i] and not bear[i-1] and adx_ok[i] and vol_ok[i]: sig[i] = -1

    # Apply filters
    if filters:
        sl5 = sl8 = sl21 = bb_lo = bb_up = fan = None
        if "slope" in filters:
            sl5, sl8, sl21 = ma_slope(ma5), ma_slope(ma8), ma_slope(e21)
        if "bb" in filters:
            bb_lo, _, bb_up = bollinger_bands(c)
        if "fanning" in filters:
            fan = fanning_dist(ma5, ma8, e21, ma55, c)

        for i in range(n):
            if sig[i] == 0: continue
            if "slope" in filters:
                thr = p.get("slope_thr", 0.0)
                if sig[i]==1:
                    if np.isnan(sl5[i]) or sl5[i]<=thr or np.isnan(sl8[i]) or sl8[i]<=thr or np.isnan(sl21[i]) or sl21[i]<=thr:
                        sig[i]=0; continue
                else:
                    if np.isnan(sl5[i]) or sl5[i]>=-thr or np.isnan(sl8[i]) or sl8[i]>=-thr or np.isnan(sl21[i]) or sl21[i]>=-thr:
                        sig[i]=0; continue
            if sig[i]!=0 and "bb" in filters:
                bm = p.get("bb_max", 1.0)
                if np.isnan(bb_lo[i]) or np.isnan(bb_up[i]) or (bb_up[i]-bb_lo[i])<=0:
                    sig[i]=0; continue
                pos = (c[i]-bb_lo[i])/(bb_up[i]-bb_lo[i])
                if sig[i]==1 and pos>bm: sig[i]=0; continue
                elif sig[i]==-1 and pos<(1-bm): sig[i]=0; continue
            if sig[i]!=0 and "fanning" in filters:
                fm = p.get("fan_min", 0.5)
                if np.isnan(fan[i]) or fan[i]<fm: sig[i]=0; continue

    # Backtest
    pos=0; entry=0.0; sl_=0.0; tp_=0.0
    equity=1.0; peak=1.0; max_dd=0.0; returns=[]; wins=0; trades=0
    am, rr = p["atr_mult"], p["rr"]; fee=0.0005

    for i in range(1, n):
        if pos!=0:
            hs = (l[i]<=sl_) if pos==1 else (h[i]>=sl_)
            ht = (h[i]>=tp_) if pos==1 else (l[i]<=tp_)
            if ht and hs:
                if (pos==1 and c[i]<entry) or (pos==-1 and c[i]>entry): ht=False
                else: hs=False
            if ht:
                net=abs(tp_-entry)/entry-fee*2; equity*=(1+net); returns.append(net); wins+=1; trades+=1; pos=0
            elif hs:
                net=-abs(sl_-entry)/entry-fee*2; equity*=(1+net); returns.append(net); trades+=1; pos=0
            if pos==1 and not np.isnan(atr14[i]): sl_=max(sl_,c[i]-am*atr14[i])
            elif pos==-1 and not np.isnan(atr14[i]): sl_=min(sl_,c[i]+am*atr14[i])
            peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak)
        if pos==0 and sig[i]!=0 and not np.isnan(atr14[i]):
            pos=int(sig[i]); entry=c[i]; sd=am*atr14[i]
            sl_=entry-sd if pos==1 else entry+sd; tp_=entry+rr*sd if pos==1 else entry-rr*sd

    if trades<2: return {"net":0,"sharpe":0,"wr":0,"trades":trades,"dd":0}
    r=np.array(returns); sharpe=np.mean(r)/(np.std(r)+1e-12)*np.sqrt(len(r))
    return {"net":(equity-1)*100,"sharpe":sharpe,"wr":wins/trades*100,"trades":trades,"dd":max_dd*100}

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

SYMBOLS = [
    "btcusdt","ethusdt","solusdt","dogeusdt","nearusdt","hypeusdt","myxusdt",
    "adausdt","avaxusdt","bnbusdt","dotusdt","enausdt","linkusdt","pepeusdt",
    "shibusdt","suiusdt","taousdt","wifusdt","xrpusdt","ariausdt",
    "1inchusdt","aptusdt","arbusdt",
]

TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# Top filter combos from round 2
FILTER_COMBOS = {
    "V0":       set(),
    "V1":       {"slope"},
    "V3":       {"fanning"},
    "V1+V3":    {"slope", "fanning"},
    "V2+V3":    {"bb", "fanning"},
    "V1+V2":    {"slope", "bb"},
    "V1+V2+V3": {"slope", "bb", "fanning"},
}

# Top param sets from round 2 + new combinations
PARAM_SETS = [
    {"label": "A_default",     "ma5_n":5,"ma8_n":8,"ema21_n":21,"ma55_n":55,"adx_min":20,"vol_mult":1.2,"atr_mult":2.0,"rr":2.0,"slope_thr":0.0,"bb_max":1.0,"fan_min":0.5},
    {"label": "B_wide+lowRR",  "ma5_n":5,"ma8_n":8,"ema21_n":21,"ma55_n":55,"adx_min":20,"vol_mult":1.2,"atr_mult":3.0,"rr":1.5,"slope_thr":0.0,"bb_max":1.0,"fan_min":0.5},
    {"label": "C_strict",      "ma5_n":5,"ma8_n":8,"ema21_n":21,"ma55_n":55,"adx_min":25,"vol_mult":1.2,"atr_mult":2.0,"rr":2.0,"slope_thr":0.1,"bb_max":0.9,"fan_min":0.8},
    {"label": "D_fib",         "ma5_n":5,"ma8_n":8,"ema21_n":13,"ma55_n":34,"adx_min":20,"vol_mult":1.2,"atr_mult":2.0,"rr":2.0,"slope_thr":0.0,"bb_max":1.0,"fan_min":0.5},
    {"label": "E_wide+strict", "ma5_n":5,"ma8_n":8,"ema21_n":21,"ma55_n":55,"adx_min":25,"vol_mult":1.2,"atr_mult":3.0,"rr":1.5,"slope_thr":0.1,"bb_max":0.9,"fan_min":0.8},
    {"label": "F_noVol+loose", "ma5_n":5,"ma8_n":8,"ema21_n":21,"ma55_n":55,"adx_min":15,"vol_mult":0.0,"atr_mult":2.5,"rr":2.0,"slope_thr":0.0,"bb_max":1.0,"fan_min":0.3},
    {"label": "G_fib+wide",    "ma5_n":5,"ma8_n":8,"ema21_n":13,"ma55_n":34,"adx_min":20,"vol_mult":1.2,"atr_mult":3.0,"rr":1.5,"slope_thr":0.0,"bb_max":1.0,"fan_min":0.5},
]

def main():
    total = len(SYMBOLS)*len(TIMEFRAMES)*len(FILTER_COMBOS)*len(PARAM_SETS)
    print("="*100)
    print("MA RIBBON - FINAL EXTENDED SWEEP")
    print(f"  {len(SYMBOLS)} symbols x {len(TIMEFRAMES)} TFs x {len(FILTER_COMBOS)} filter combos x {len(PARAM_SETS)} param sets")
    print(f"  Max {total} backtests (skipping missing data)")
    print("="*100)

    results = []
    done = 0; skipped = 0

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            data = load_ohlcv(sym, tf)
            if data is None:
                skipped += len(FILTER_COMBOS)*len(PARAM_SETS)
                continue
            o, h, l, c, v = data

            for pi, p in enumerate(PARAM_SETS):
                for fname, filters in FILTER_COMBOS.items():
                    bt = compute_and_backtest(o, h, l, c, v, p, filters)
                    results.append({
                        "sym": sym.upper(), "tf": tf, "flt": fname,
                        "par": p["label"], "pi": pi, **bt
                    })
                    done += 1

            sys.stdout.write(f"\r  {done:5d} done | {sym.upper():>10} {tf}")
            sys.stdout.flush()

    print(f"\n\n  Total: {done} backtests run, {skipped} skipped (missing data)")

    # ═══════════════════════════════════════════════════════════
    # TOP 50 by Sharpe (min 10 trades)
    # ═══════════════════════════════════════════════════════════
    good = [r for r in results if r["trades"]>=10 and r["sharpe"]>0]
    good.sort(key=lambda r: r["sharpe"], reverse=True)

    print("\n"+"="*100)
    print(f"TOP 50 BY SHARPE (min 10 trades) — {len(good)} profitable out of {done}")
    print("="*100)
    print(f"{'#':>3} {'symbol':<10} {'TF':<4} {'filters':<10} {'params':<16} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>9} {'DD%':>6}")
    print("-"*85)
    for i, r in enumerate(good[:50], 1):
        print(f"{i:3d} {r['sym']:<10} {r['tf']:<4} {r['flt']:<10} {r['par']:<16} "
              f"{r['trades']:6d} {r['wr']:5.1f}% {r['sharpe']:+6.2f} {r['net']:+8.1f}% {r['dd']:5.1f}%")

    # ═══════════════════════════════════════════════════════════
    # BEST PER SYMBOL (all TFs, min 10 trades)
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*100)
    print("BEST COMBO PER SYMBOL (min 10 trades, by Sharpe)")
    print("="*100)
    print(f"{'symbol':<10} {'TF':<4} {'filters':<10} {'params':<16} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>9} {'DD%':>6}")
    print("-"*85)
    for sym in SYMBOLS:
        sr = [r for r in results if r["sym"]==sym.upper() and r["trades"]>=10 and r["sharpe"]>0]
        if not sr:
            print(f"{sym.upper():<10} (no profitable combo with 10+ trades)")
            continue
        sr.sort(key=lambda r: r["sharpe"], reverse=True)
        b = sr[0]
        print(f"{b['sym']:<10} {b['tf']:<4} {b['flt']:<10} {b['par']:<16} "
              f"{b['trades']:6d} {b['wr']:5.1f}% {b['sharpe']:+6.2f} {b['net']:+8.1f}% {b['dd']:5.1f}%")

    # ═══════════════════════════════════════════════════════════
    # AGGREGATE: filter x params (avg sharpe across all sym x tf)
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*100)
    print("HEATMAP: avg Sharpe by (filter combo x param set) — across all symbols x TFs")
    print("="*100)

    # Header
    plabels = [p["label"] for p in PARAM_SETS]
    print(f"{'':>10}", end="")
    for pl in plabels:
        print(f" {pl:>14}", end="")
    print(f" {'BEST_PAR':>14}")
    print("-"*(10 + 15*len(plabels) + 15))

    best_overall_sharpe = -999
    best_overall_combo = None

    for fname in FILTER_COMBOS:
        print(f"{fname:>10}", end="")
        best_s = -999; best_p = ""
        for pi, p in enumerate(PARAM_SETS):
            fr = [r for r in results if r["flt"]==fname and r["pi"]==pi and r["trades"]>=5]
            if fr:
                avg_s = np.mean([r["sharpe"] for r in fr])
                print(f" {avg_s:+13.2f}", end="")
                if avg_s > best_s:
                    best_s = avg_s; best_p = p["label"]
                if avg_s > best_overall_sharpe:
                    best_overall_sharpe = avg_s
                    best_overall_combo = (fname, p["label"])
            else:
                print(f" {'N/A':>14}", end="")
        print(f" {best_p:>14}")

    print(f"\n  >>> Best overall cell: {best_overall_combo[0]} x {best_overall_combo[1]} = Sharpe {best_overall_sharpe:+.3f}")

    # ═══════════════════════════════════════════════════════════
    # AGGREGATE: by TF
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*100)
    print("BEST (filter x param) PER TIMEFRAME (avg Sharpe, min 3 symbols contributing)")
    print("="*100)

    for tf in TIMEFRAMES:
        combos = []
        for fname in FILTER_COMBOS:
            for pi, p in enumerate(PARAM_SETS):
                fr = [r for r in results if r["tf"]==tf and r["flt"]==fname and r["pi"]==pi and r["trades"]>=5]
                if len(fr) >= 3:
                    avg_s = np.mean([r["sharpe"] for r in fr])
                    avg_n = np.mean([r["net"] for r in fr])
                    avg_w = np.mean([r["wr"] for r in fr])
                    avg_d = np.mean([r["dd"] for r in fr])
                    npos = sum(1 for r in fr if r["net"]>0)
                    combos.append({"f":fname,"p":p["label"],"s":avg_s,"n":avg_n,"w":avg_w,"d":avg_d,"pos":npos,"tot":len(fr)})
        combos.sort(key=lambda x: x["s"], reverse=True)
        print(f"\n  --- {tf} (top 5) ---")
        print(f"  {'filters':<10} {'params':<16} {'avg_sharpe':>10} {'avg_net%':>9} {'avg_win%':>9} {'avg_DD%':>8} {'pos/n':>7}")
        for c in combos[:5]:
            print(f"  {c['f']:<10} {c['p']:<16} {c['s']:+9.2f} {c['n']:+8.1f}% {c['w']:8.1f}% {c['d']:7.1f}% {c['pos']:>3}/{c['tot']}")

    # ═══════════════════════════════════════════════════════════
    # CONSISTENCY: combos profitable across most symbols
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*100)
    print("MOST CONSISTENT: combos profitable on most symbols (any TF, min 10 trades)")
    print("="*100)

    consistency = []
    for fname in FILTER_COMBOS:
        for pi, p in enumerate(PARAM_SETS):
            # For each symbol, is the best TF for this combo profitable?
            sym_profitable = 0; sym_total = 0
            for sym in SYMBOLS:
                sr = [r for r in results if r["sym"]==sym.upper() and r["flt"]==fname
                      and r["pi"]==pi and r["trades"]>=10]
                if sr:
                    sym_total += 1
                    best = max(sr, key=lambda r: r["sharpe"])
                    if best["net"] > 0:
                        sym_profitable += 1
            if sym_total >= 5:
                consistency.append({"f":fname,"p":p["label"],"pos":sym_profitable,"tot":sym_total,
                                   "pct":sym_profitable/sym_total*100})

    consistency.sort(key=lambda x: x["pct"], reverse=True)
    print(f"  {'filters':<10} {'params':<16} {'profitable_symbols':>20} {'pct':>6}")
    print("-"*60)
    for c in consistency[:15]:
        print(f"  {c['f']:<10} {c['p']:<16} {c['pos']:>8}/{c['tot']:<8} {c['pct']:5.1f}%")

    print(f"\nCompleted: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*100)


if __name__ == "__main__":
    main()
