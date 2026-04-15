"""
MA Ribbon V3 - MTF Confirmation + BB Exit + Ribbon Exit + Walk-Forward
Run: python research_backtest_v3.py
"""

import gzip, csv, sys, os, numpy as np
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_ohlcv(symbol, tf):
    fn = f"data/{symbol}_{tf}.csv.gz"
    if not os.path.exists(fn): return None
    with gzip.open(fn, "rt") as f:
        reader = csv.DictReader(f); rows = list(reader)
    if len(rows) < 200: return None
    return {
        "o": np.array([float(r["open"]) for r in rows]),
        "h": np.array([float(r["high"]) for r in rows]),
        "l": np.array([float(r["low"]) for r in rows]),
        "c": np.array([float(r["close"]) for r in rows]),
        "v": np.array([float(r["volume"]) for r in rows]),
        "ts": [r["open_time"] for r in rows],
    }

def resample_ohlcv(d, factor):
    """Resample OHLCV by factor (e.g., 4 = 4x higher TF). Returns dict like load_ohlcv."""
    n = len(d["c"])
    n_bars = n // factor
    if n_bars < 100: return None
    trim = n_bars * factor
    o = d["o"][:trim].reshape(n_bars, factor)
    h = d["h"][:trim].reshape(n_bars, factor)
    l = d["l"][:trim].reshape(n_bars, factor)
    c = d["c"][:trim].reshape(n_bars, factor)
    v = d["v"][:trim].reshape(n_bars, factor)
    return {
        "o": o[:, 0], "h": h.max(axis=1), "l": l.min(axis=1),
        "c": c[:, -1], "v": v.sum(axis=1),
    }

# ═══════════════════════════════════════════════════════════════
# INDICATORS (vectorized)
# ═══════════════════════════════════════════════════════════════

def sma(x, n):
    out = np.full(len(x), np.nan)
    if not np.any(np.isnan(x)):
        cs = np.cumsum(x); out[n-1:] = (cs[n-1:] - np.concatenate([[0], cs[:-n]])) / n
        return out
    nm = np.isnan(x); ncs = np.cumsum(nm); xf = np.where(nm, 0, x); cs = np.cumsum(xf)
    for i in range(n-1, len(x)):
        nw = ncs[i] - (ncs[i-n] if i>=n else 0)
        if nw == 0: out[i] = (cs[i] - (cs[i-n] if i>=n else 0)) / n
    return out

def ema(x, n):
    a = 2.0/(n+1); out = np.full(len(x), np.nan); out[n-1] = np.mean(x[:n])
    for i in range(n, len(x)): out[i] = a*x[i] + (1-a)*out[i-1]
    return out

def atr_calc(h, l, c, n=14):
    pc = np.roll(c,1); tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc))); tr[0]=h[0]-l[0]
    return sma(tr, n)

def adx_calc(h, l, c, n=14):
    ph,pl,pc = np.roll(h,1),np.roll(l,1),np.roll(c,1)
    tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc))); tr[0]=h[0]-l[0]
    dmp = np.where((h-ph)>(pl-l), np.maximum(h-ph,0), 0.0); dmp[0]=0
    dmn = np.where((pl-l)>(h-ph), np.maximum(pl-l,0), 0.0); dmn[0]=0
    an=sma(tr,n); dip=100*sma(dmp,n)/(an+1e-12); din=100*sma(dmn,n)/(an+1e-12)
    dx=100*np.abs(dip-din)/(dip+din+1e-12); return sma(dx,n)

def bb_calc(c, n=20, std=2.0):
    mid = sma(c, n); cs=np.cumsum(c); cs2=np.cumsum(c**2); s=np.full(len(c), np.nan)
    for i in range(n-1, len(c)):
        sm=cs[i]-(cs[i-n] if i>=n else 0); sm2=cs2[i]-(cs2[i-n] if i>=n else 0)
        s[i] = np.sqrt(max(sm2/n-(sm/n)**2, 0))
    return mid-std*s, mid, mid+std*s

def ma_slope(ma, lb=5):
    s=np.full(len(ma),np.nan); sh=np.roll(ma,lb); sh[:lb]=np.nan
    v=~np.isnan(ma)&~np.isnan(sh)&(sh!=0); s[v]=(ma[v]-sh[v])/sh[v]*100; return s

def fanning_dist(ma5, ma8, e21, ma55, c):
    v=~(np.isnan(ma5)|np.isnan(ma8)|np.isnan(e21)|np.isnan(ma55)); d=np.full(len(c),np.nan)
    d[v]=(np.abs(ma5[v]-ma8[v])+np.abs(ma8[v]-e21[v])+np.abs(e21[v]-ma55[v]))/(3*c[v])*100; return d

def ribbon_state(c, p):
    ma5=sma(c,p["ma5_n"]); ma8=sma(c,p["ma8_n"]); e21=ema(c,p["ema21_n"]); ma55=sma(c,p["ma55_n"])
    bull=(c>ma5)&(ma5>ma8)&(ma8>e21)&(e21>ma55)
    bear=(c<ma5)&(ma5<ma8)&(ma8<e21)&(e21<ma55)
    return bull, bear, ma5, ma8, e21, ma55

# ═══════════════════════════════════════════════════════════════
# MTF CONFIRMATION
# ═══════════════════════════════════════════════════════════════

# Pine: 15m->4H, 1H->1D, 4H->3D, 5m->1H
MTF_MAP = {"5m": 12, "15m": 16, "1h": 24, "4h": 18}  # resample factors

def compute_htf_ribbon(d, tf, p):
    """Compute HTF ribbon state, expanded back to base TF resolution"""
    factor = MTF_MAP.get(tf)
    if factor is None: return None, None
    htf = resample_ohlcv(d, factor)
    if htf is None: return None, None
    htf_bull, htf_bear, _, _, _, _ = ribbon_state(htf["c"], p)
    # Expand back: each HTF bar covers `factor` base bars
    n = len(d["c"])
    n_htf = len(htf_bull)
    bull_exp = np.zeros(n, dtype=bool)
    bear_exp = np.zeros(n, dtype=bool)
    for i in range(n_htf):
        start = i * factor; end = min((i+1)*factor, n)
        bull_exp[start:end] = htf_bull[i]
        bear_exp[start:end] = htf_bear[i]
    return bull_exp, bear_exp

# ═══════════════════════════════════════════════════════════════
# UNIFIED BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(d, tf, p, filters, exit_mode, bb_exit_n=20, bb_exit_std=2.0, use_mtf=False):
    """
    exit_mode: "atr_rr" (fixed SL/TP), "bb_exit" (BB as TP), "ribbon_exit" (exit on reverse ribbon),
               "bb+trail" (BB TP with trailing SL), "ribbon+trail" (ribbon exit with trailing SL)
    """
    o,h,l,c,v = d["o"],d["h"],d["l"],d["c"],d["v"]
    n = len(c)

    # Base indicators
    bull, bear, ma5, ma8, e21, ma55 = ribbon_state(c, p)
    atr14 = atr_calc(h, l, c, 14)
    adx14 = adx_calc(h, l, c, 14)
    vol_avg = sma(v, 20)
    adx_ok = adx14 > p["adx_min"]
    vol_ok = (v > p["vol_mult"]*vol_avg) if p["vol_mult"] > 0 else np.ones(n, bool)

    # MTF
    if use_mtf:
        htf_bull, htf_bear = compute_htf_ribbon(d, tf, p)
        if htf_bull is None: use_mtf = False

    # Filters
    sl5 = sl8 = sl21 = fan = None
    if "slope" in filters: sl5,sl8,sl21 = ma_slope(ma5),ma_slope(ma8),ma_slope(e21)
    if "fanning" in filters: fan = fanning_dist(ma5, ma8, e21, ma55, c)

    # BB for exit
    bb_lo = bb_up = None
    if "bb" in exit_mode:
        bb_lo, _, bb_up = bb_calc(c, bb_exit_n, bb_exit_std)

    # Generate signals
    sig = np.zeros(n)
    for i in range(1, n):
        if not (adx_ok[i] and vol_ok[i]): continue
        if bull[i] and not bull[i-1]:
            direction = 1
        elif bear[i] and not bear[i-1]:
            direction = -1
        else:
            continue

        # MTF confirmation
        if use_mtf:
            if direction == 1 and not htf_bull[i]: continue
            if direction == -1 and not htf_bear[i]: continue

        # Slope filter
        if "slope" in filters:
            thr = p.get("slope_thr", 0.0)
            if direction == 1:
                if np.isnan(sl5[i]) or sl5[i]<=thr or np.isnan(sl8[i]) or sl8[i]<=thr or np.isnan(sl21[i]) or sl21[i]<=thr: continue
            else:
                if np.isnan(sl5[i]) or sl5[i]>=-thr or np.isnan(sl8[i]) or sl8[i]>=-thr or np.isnan(sl21[i]) or sl21[i]>=-thr: continue

        # Fanning filter
        if "fanning" in filters:
            fm = p.get("fan_min", 0.5)
            if np.isnan(fan[i]) or fan[i]<fm: continue

        sig[i] = direction

    # ── Backtest ──
    pos=0; entry=0.0; sl_=0.0; tp_=0.0
    equity=1.0; peak=1.0; max_dd=0.0; returns=[]; wins=0; trades=0
    am=p["atr_mult"]; rr=p["rr"]; fee=0.0005

    for i in range(1, n):
        if pos != 0:
            exit_price = None; is_win = False

            # Check SL hit
            sl_hit = (pos==1 and l[i]<=sl_) or (pos==-1 and h[i]>=sl_)

            # Check TP/exit conditions
            tp_hit = False
            if exit_mode == "atr_rr":
                tp_hit = (pos==1 and h[i]>=tp_) or (pos==-1 and l[i]<=tp_)
            elif "bb" in exit_mode and bb_up is not None and not np.isnan(bb_up[i]):
                tp_hit = (pos==1 and h[i]>=bb_up[i]) or (pos==-1 and l[i]<=bb_lo[i])
                if tp_hit:
                    tp_ = bb_up[i] if pos==1 else bb_lo[i]
            elif "ribbon" in exit_mode:
                # Exit when opposite ribbon forms
                if pos==1 and bear[i]: tp_hit=True; tp_=c[i]
                elif pos==-1 and bull[i]: tp_hit=True; tp_=c[i]

            if tp_hit and sl_hit:
                if (pos==1 and c[i]<entry) or (pos==-1 and c[i]>entry): tp_hit=False
                else: sl_hit=False

            if tp_hit:
                net=abs(tp_-entry)/entry-fee*2; equity*=(1+net); returns.append(net); wins+=1; trades+=1; pos=0
            elif sl_hit:
                net=-abs(sl_-entry)/entry-fee*2; equity*=(1+net); returns.append(net); trades+=1; pos=0

            # Trail SL
            if pos!=0 and "trail" in exit_mode or exit_mode=="atr_rr":
                if pos==1 and not np.isnan(atr14[i]): sl_=max(sl_,c[i]-am*atr14[i])
                elif pos==-1 and not np.isnan(atr14[i]): sl_=min(sl_,c[i]+am*atr14[i])

            peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak)

        # New entry
        if pos==0 and sig[i]!=0 and not np.isnan(atr14[i]):
            pos=int(sig[i]); entry=c[i]; sd=am*atr14[i]
            sl_ = entry-sd if pos==1 else entry+sd
            if exit_mode == "atr_rr":
                tp_ = entry+rr*sd if pos==1 else entry-rr*sd
            elif "bb" in exit_mode and bb_up is not None and not np.isnan(bb_up[i]):
                tp_ = bb_up[i] if pos==1 else bb_lo[i]
            else:
                tp_ = entry+rr*sd if pos==1 else entry-rr*sd  # fallback

    if trades<2: return {"net":0,"sharpe":0,"wr":0,"trades":trades,"dd":0,"signals":int(np.sum(sig!=0))}
    r=np.array(returns); sharpe=np.mean(r)/(np.std(r)+1e-12)*np.sqrt(len(r))
    return {"net":(equity-1)*100,"sharpe":sharpe,"wr":wins/trades*100,"trades":trades,"dd":max_dd*100,"signals":int(np.sum(sig!=0))}

# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD SPLIT
# ═══════════════════════════════════════════════════════════════

def walk_forward(d, tf, p, filters, exit_mode, bb_n=20, bb_std=2.0, use_mtf=False, n_folds=3):
    """Split data into n_folds, train on fold[i], test on fold[i+1]"""
    n = len(d["c"])
    fold_size = n // n_folds
    oos_results = []

    for fold in range(n_folds - 1):
        # Out-of-sample = next fold
        oos_start = (fold + 1) * fold_size
        oos_end = min((fold + 2) * fold_size, n)

        oos_d = {k: v[oos_start:oos_end] if isinstance(v, np.ndarray) else v[oos_start:oos_end] for k, v in d.items()}
        bt = run_backtest(oos_d, tf, p, filters, exit_mode, bb_n, bb_std, use_mtf)
        oos_results.append(bt)

    # Aggregate OOS
    total_trades = sum(r["trades"] for r in oos_results)
    if total_trades < 2:
        return {"net":0,"sharpe":0,"wr":0,"trades":0,"dd":0}
    avg_sharpe = np.mean([r["sharpe"] for r in oos_results if r["trades"]>=2]) if any(r["trades"]>=2 for r in oos_results) else 0
    total_net = 1.0
    for r in oos_results:
        total_net *= (1 + r["net"]/100)
    total_net = (total_net - 1) * 100
    max_dd = max(r["dd"] for r in oos_results)
    avg_wr = np.mean([r["wr"] for r in oos_results if r["trades"]>=2]) if any(r["trades"]>=2 for r in oos_results) else 0
    return {"net":total_net,"sharpe":avg_sharpe,"wr":avg_wr,"trades":total_trades,"dd":max_dd}

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

SYMBOLS = ["btcusdt","ethusdt","solusdt","dogeusdt","nearusdt","hypeusdt",
           "adausdt","bnbusdt","enausdt","linkusdt","pepeusdt","shibusdt",
           "suiusdt","xrpusdt","ariausdt","avaxusdt","dotusdt","wifusdt","myxusdt"]

TIMEFRAMES = ["15m", "1h", "4h"]

# Best params from round 2
PARAMS = {
    "E_strict": {"ma5_n":5,"ma8_n":8,"ema21_n":21,"ma55_n":55,"adx_min":25,"vol_mult":1.2,
                 "atr_mult":3.0,"rr":1.5,"slope_thr":0.1,"fan_min":0.8},
    "B_wide":   {"ma5_n":5,"ma8_n":8,"ema21_n":21,"ma55_n":55,"adx_min":20,"vol_mult":1.2,
                 "atr_mult":3.0,"rr":1.5,"slope_thr":0.0,"fan_min":0.5},
    "A_default":{"ma5_n":5,"ma8_n":8,"ema21_n":21,"ma55_n":55,"adx_min":20,"vol_mult":1.2,
                 "atr_mult":2.0,"rr":2.0,"slope_thr":0.0,"fan_min":0.5},
}

# Best filter combo from round 2
FILTER_SETS = {
    "V1+V3":    {"slope", "fanning"},
    "V1":       {"slope"},
    "V0":       set(),
}

# Exit modes to test
EXIT_MODES = [
    ("atr_rr",       20, 2.0, "ATR_trail+RR"),
    ("bb_exit",      20, 2.0, "BB(20,2.0)"),
    ("bb_exit",      20, 1.5, "BB(20,1.5)"),
    ("bb_exit",      30, 2.0, "BB(30,2.0)"),
    ("bb_exit",      14, 2.0, "BB(14,2.0)"),
    ("bb+trail",     20, 2.0, "BB(20,2)+trail"),
    ("bb+trail",     20, 1.5, "BB(20,1.5)+trail"),
    ("bb+trail",     30, 2.0, "BB(30,2)+trail"),
    ("ribbon_exit",  20, 2.0, "Ribbon_reversal"),
    ("ribbon+trail", 20, 2.0, "Ribbon+trail"),
]

# MTF options
MTF_OPTIONS = [False, True]

def main():
    total = len(SYMBOLS)*len(TIMEFRAMES)*len(PARAMS)*len(FILTER_SETS)*len(EXIT_MODES)*len(MTF_OPTIONS)
    print("="*105)
    print("MA RIBBON V3 - MTF + EXIT LOGIC + BB PARAMS + WALK-FORWARD")
    print(f"  {len(SYMBOLS)} sym x {len(TIMEFRAMES)} TF x {len(PARAMS)} params x {len(FILTER_SETS)} filters x {len(EXIT_MODES)} exits x {len(MTF_OPTIONS)} MTF")
    print(f"  Max {total} backtests")
    print("="*105)

    results = []
    done = 0

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            d = load_ohlcv(sym, tf)
            if d is None: continue

            for pname, p in PARAMS.items():
                for fname, filters in FILTER_SETS.items():
                    for exit_mode, bb_n, bb_std, exit_label in EXIT_MODES:
                        for use_mtf in MTF_OPTIONS:
                            bt = run_backtest(d, tf, p, filters, exit_mode, bb_n, bb_std, use_mtf)
                            mtf_label = "+MTF" if use_mtf else ""
                            results.append({
                                "sym":sym.upper(),"tf":tf,"par":pname,"flt":fname,
                                "exit":exit_label,"mtf":use_mtf,
                                "label":f"{fname}{mtf_label}|{exit_label}|{pname}",
                                **bt
                            })
                            done += 1

            sys.stdout.write(f"\r  {done:5d} done | {sym.upper():>10} {tf}  ")
            sys.stdout.flush()

    print(f"\n\n  Total: {done} backtests")

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: TOP 40 by Sharpe
    # ═══════════════════════════════════════════════════════════
    good = [r for r in results if r["trades"]>=10 and r["sharpe"]>0]
    good.sort(key=lambda r: r["sharpe"], reverse=True)

    print("\n"+"="*105)
    print(f"TOP 40 BY SHARPE (min 10 trades) — {len(good)} profitable / {done} total")
    print("="*105)
    print(f"{'#':>3} {'sym':<8} {'TF':<4} {'flt':<6} {'MTF':<4} {'exit':<17} {'par':<10} {'tr':>4} {'win%':>6} {'shrp':>6} {'net%':>8} {'DD%':>6}")
    print("-"*90)
    for i, r in enumerate(good[:40], 1):
        mtf = "Y" if r["mtf"] else ""
        print(f"{i:3d} {r['sym']:<8} {r['tf']:<4} {r['flt']:<6} {mtf:<4} {r['exit']:<17} {r['par']:<10} "
              f"{r['trades']:4d} {r['wr']:5.1f}% {r['sharpe']:+5.2f} {r['net']:+7.1f}% {r['dd']:5.1f}%")

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: MTF impact
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*105)
    print("MTF IMPACT: avg Sharpe with vs without MTF confirmation")
    print("="*105)
    print(f"{'TF':<5} {'filter':<8} {'exit':<17} {'params':<10} {'noMTF_shrp':>10} {'MTF_shrp':>10} {'delta':>7} {'noMTF_tr':>9} {'MTF_tr':>7}")
    print("-"*90)

    # Group by (tf, flt, exit, par) and compare MTF on/off
    from collections import defaultdict
    groups = defaultdict(lambda: {"mtf": [], "no_mtf": []})
    for r in results:
        if r["trades"] < 5: continue
        key = (r["tf"], r["flt"], r["exit"], r["par"])
        if r["mtf"]: groups[key]["mtf"].append(r)
        else: groups[key]["no_mtf"].append(r)

    mtf_deltas = []
    for key, g in sorted(groups.items()):
        if len(g["mtf"]) >= 3 and len(g["no_mtf"]) >= 3:
            s_no = np.mean([r["sharpe"] for r in g["no_mtf"]])
            s_mtf = np.mean([r["sharpe"] for r in g["mtf"]])
            t_no = np.mean([r["trades"] for r in g["no_mtf"]])
            t_mtf = np.mean([r["trades"] for r in g["mtf"]])
            delta = s_mtf - s_no
            mtf_deltas.append((key, s_no, s_mtf, delta, t_no, t_mtf))

    mtf_deltas.sort(key=lambda x: x[3], reverse=True)
    for key, s_no, s_mtf, delta, t_no, t_mtf in mtf_deltas[:20]:
        tf, flt, exit_, par = key
        print(f"{tf:<5} {flt:<8} {exit_:<17} {par:<10} {s_no:+9.2f} {s_mtf:+9.2f} {delta:+6.2f} {t_no:8.0f} {t_mtf:6.0f}")

    # Overall MTF summary
    all_no = [r["sharpe"] for r in results if not r["mtf"] and r["trades"]>=5]
    all_mtf = [r["sharpe"] for r in results if r["mtf"] and r["trades"]>=5]
    if all_no and all_mtf:
        print(f"\n  Overall avg Sharpe: no MTF = {np.mean(all_no):+.3f}, MTF = {np.mean(all_mtf):+.3f}, delta = {np.mean(all_mtf)-np.mean(all_no):+.3f}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: EXIT MODE comparison
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*105)
    print("EXIT MODE COMPARISON (avg across all symbols, best param+filter only: V1+V3 / E_strict)")
    print("="*105)
    print(f"{'exit_mode':<20} {'TF':<5} {'avg_shrp':>9} {'avg_net%':>9} {'avg_wr%':>8} {'avg_DD%':>8} {'avg_tr':>7} {'pos/n':>7}")
    print("-"*80)

    for tf in TIMEFRAMES:
        for exit_mode, bb_n, bb_std, exit_label in EXIT_MODES:
            for use_mtf in [False, True]:
                mtf_tag = "+MTF" if use_mtf else ""
                er = [r for r in results if r["tf"]==tf and r["exit"]==exit_label
                      and r["flt"]=="V1+V3" and r["par"]=="E_strict" and r["mtf"]==use_mtf
                      and r["trades"]>=5]
                if len(er) < 3: continue
                avg_s = np.mean([r["sharpe"] for r in er])
                avg_n = np.mean([r["net"] for r in er])
                avg_w = np.mean([r["wr"] for r in er])
                avg_d = np.mean([r["dd"] for r in er])
                avg_t = np.mean([r["trades"] for r in er])
                npos = sum(1 for r in er if r["net"]>0)
                label = f"{exit_label}{mtf_tag}"
                print(f"{label:<20} {tf:<5} {avg_s:+8.2f} {avg_n:+8.1f}% {avg_w:7.1f}% {avg_d:7.1f}% {avg_t:6.0f} {npos:>3}/{len(er)}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: BB PARAM sweep for BB exit
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*105)
    print("BB EXIT PARAM SWEEP: which (period, std) works best?")
    print("="*105)
    print(f"{'bb_config':<18} {'avg_shrp':>9} {'avg_net%':>9} {'avg_wr%':>8} {'avg_tr':>7} {'pos/n':>7}")
    print("-"*60)

    for exit_mode, bb_n, bb_std, exit_label in EXIT_MODES:
        if "bb" not in exit_label.lower() and "BB" not in exit_label: continue
        er = [r for r in results if r["exit"]==exit_label and r["flt"]=="V1+V3" and r["trades"]>=5]
        if len(er) < 5: continue
        avg_s = np.mean([r["sharpe"] for r in er])
        avg_n = np.mean([r["net"] for r in er])
        avg_w = np.mean([r["wr"] for r in er])
        avg_t = np.mean([r["trades"] for r in er])
        npos = sum(1 for r in er if r["net"]>0)
        print(f"{exit_label:<18} {avg_s:+8.2f} {avg_n:+8.1f}% {avg_w:7.1f}% {avg_t:6.0f} {npos:>3}/{len(er)}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: WALK-FORWARD on top 10 combos
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*105)
    print("WALK-FORWARD VALIDATION (3-fold OOS) on TOP 10 in-sample combos")
    print("="*105)

    # Pick top 10 unique (sym, tf, par, flt, exit, mtf) by sharpe
    top10 = good[:10]
    print(f"{'#':>3} {'sym':<8} {'TF':<4} {'flt':<6} {'MTF':<4} {'exit':<17} {'par':<10} {'IS_shrp':>8} {'OOS_shrp':>9} {'OOS_net%':>9} {'OOS_DD%':>8}")
    print("-"*95)

    for i, r in enumerate(top10, 1):
        d = load_ohlcv(r["sym"].lower(), r["tf"])
        if d is None: continue
        p = PARAMS[r["par"]]
        filters = FILTER_SETS[r["flt"]]
        # Find exit params
        exit_mode = bb_n = bb_std = None
        for em, bn, bs, el in EXIT_MODES:
            if el == r["exit"]:
                exit_mode, bb_n, bb_std = em, bn, bs; break
        if exit_mode is None: continue

        wf = walk_forward(d, r["tf"], p, filters, exit_mode, bb_n, bb_std, r["mtf"], n_folds=3)
        mtf = "Y" if r["mtf"] else ""
        print(f"{i:3d} {r['sym']:<8} {r['tf']:<4} {r['flt']:<6} {mtf:<4} {r['exit']:<17} {r['par']:<10} "
              f"{r['sharpe']:+7.2f} {wf['sharpe']:+8.2f} {wf['net']:+8.1f}% {wf['dd']:7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # SECTION 6: BEST OVERALL RECOMMENDATION
    # ═══════════════════════════════════════════════════════════
    print("\n"+"="*105)
    print("MOST CONSISTENT COMBOS (profitable on most symbols, min 10 trades)")
    print("="*105)

    consistency = []
    seen = set()
    for r in results:
        key = (r["flt"], r["mtf"], r["exit"], r["par"])
        if key in seen: continue
        seen.add(key)
        # Count profitable symbols for this combo (best TF per symbol)
        sym_pos = 0; sym_tot = 0
        for sym in SYMBOLS:
            sr = [x for x in results if x["sym"]==sym.upper() and x["flt"]==r["flt"]
                  and x["mtf"]==r["mtf"] and x["exit"]==r["exit"] and x["par"]==r["par"]
                  and x["trades"]>=10]
            if sr:
                sym_tot += 1
                best = max(sr, key=lambda x: x["sharpe"])
                if best["net"] > 0: sym_pos += 1
        if sym_tot >= 5:
            consistency.append({**r, "sym_pos":sym_pos, "sym_tot":sym_tot, "pct":sym_pos/sym_tot*100})

    consistency.sort(key=lambda x: x["pct"], reverse=True)
    print(f"{'flt':<8} {'MTF':<4} {'exit':<17} {'par':<10} {'profitable':>12} {'pct':>6}")
    print("-"*65)
    for c in consistency[:20]:
        mtf = "Y" if c["mtf"] else ""
        print(f"{c['flt']:<8} {mtf:<4} {c['exit']:<17} {c['par']:<10} {c['sym_pos']:>5}/{c['sym_tot']:<5} {c['pct']:5.1f}%")

    print(f"\nCompleted: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*105)

if __name__ == "__main__":
    main()
