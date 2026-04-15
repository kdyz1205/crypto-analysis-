"""Trendline: TP mode (BB vs fixed RR) + reversal logic backtest."""
import sys, os, gzip, csv, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.trendline_strategy import build_trendlines, DEFAULT_CONFIG

def load(sym, tf):
    fn = f"data/{sym}_{tf}.csv.gz"
    if not os.path.exists(fn): return None
    with gzip.open(fn, "rt") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 500: return None
    return {k: np.array([float(r[k]) for r in rows]) for k in ["open","high","low","close","volume"]}

def sma(x, n):
    out = np.full(len(x), np.nan)
    if not np.any(np.isnan(x)):
        cs = np.cumsum(x); out[n-1:] = (cs[n-1:] - np.concatenate([[0], cs[:-n]])) / n
    return out

def bb_calc(c, n=20, std=1.5):
    mid = sma(c, n); cs = np.cumsum(c); cs2 = np.cumsum(c**2); s = np.full(len(c), np.nan)
    for i in range(n-1, len(c)):
        sm = cs[i] - (cs[i-n] if i >= n else 0)
        sm2 = cs2[i] - (cs2[i-n] if i >= n else 0)
        s[i] = np.sqrt(max(sm2/n - (sm/n)**2, 0))
    return mid - std*s, mid, mid + std*s

def backtest_tl(o, h, l, c, v, tp_mode="fixed_rr", rr=2.0, bb_n=20, bb_std=1.5, do_reversal=False):
    cfg = {**DEFAULT_CONFIG, "swing_lookback": 20, "buffer_pct": 0.25, "sl_pct": 0.4,
           "rr": rr, "max_hold_bars": 100, "approach_pct": 2.0,
           "min_bars_between": 40, "max_bars_between": 1000, "max_penetrations": 2}

    n = len(c)
    lines = build_trendlines(h, l, c, cfg)
    bb_lo = bb_up = None
    if "bb" in tp_mode:
        bb_lo, _, bb_up = bb_calc(c, bb_n, bb_std)

    approach = cfg["approach_pct"] / 100
    buf = cfg["buffer_pct"] / 100
    sl_p = cfg["sl_pct"] / 100
    max_proj = 200; fee = 0.0005

    lines.sort(key=lambda x: x["i2"])
    active = []; lptr = 0
    pos = 0; entry = 0.0; sl_ = 0.0; tp_ = 0.0; ebar = 0
    equity = 1.0; peak = 1.0; mdd = 0.0
    rets = []; wins = 0; trades = 0; revs = 0

    for i in range(1, n):
        if pos != 0:
            # Dynamic BB TP update
            if "bb" in tp_mode and bb_up is not None and not np.isnan(bb_up[i]):
                tp_ = bb_up[i] if pos == 1 else bb_lo[i]

            sh = (pos == 1 and l[i] <= sl_) or (pos == -1 and h[i] >= sl_)
            th = (pos == 1 and h[i] >= tp_) or (pos == -1 and l[i] <= tp_)
            to = (i - ebar) >= cfg["max_hold_bars"]

            if th and sh:
                if (pos == 1 and c[i] < entry) or (pos == -1 and c[i] > entry): th = False
                else: sh = False

            ep = None
            if th: ep = tp_
            elif sh: ep = sl_
            elif to: ep = c[i]

            if ep is not None:
                pnl = ((ep - entry) / entry if pos == 1 else (entry - ep) / entry) - fee * 2
                equity *= (1 + pnl); rets.append(pnl)
                if pnl > 0: wins += 1
                trades += 1

                # Reversal on SL hit
                if do_reversal and sh and not th:
                    nd = -pos
                    re = ep
                    rd = abs(entry - ep)
                    if rd > 0:
                        rs = re + rd if nd == -1 else re - rd
                        rt = re - rr * rd if nd == -1 else re + rr * rd
                        pos = nd; entry = re; sl_ = rs; tp_ = rt; ebar = i; revs += 1
                    else:
                        pos = 0
                else:
                    pos = 0

                peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak)

        if pos == 0:
            while lptr < len(lines) and lines[lptr]["i2"] < i:
                active.append(lines[lptr]); lptr += 1
            active = [ln for ln in active if i - ln["i2"] <= max_proj]

            bl = None; bd = float("inf")
            for ln in active:
                pr = ln["slope"] * i + ln["intercept"]
                if pr <= 0: continue
                d = abs(c[i] - pr) / pr
                if d > approach: continue
                if ln["type"] == "support" and c[i] > pr and d < bd:
                    bd = d; bl = ln
                elif ln["type"] == "resistance" and c[i] < pr and d < bd:
                    bd = d; bl = ln

            if bl:
                pr = bl["slope"] * i + bl["intercept"]
                if bl["type"] == "support":
                    e = pr * (1 + buf); s = pr * (1 - sl_p); risk = e - s
                    if tp_mode == "fixed_rr": t = e + rr * risk
                    elif bb_up is not None and not np.isnan(bb_up[i]): t = bb_up[i]
                    else: t = e + rr * risk
                    if l[i] <= e <= h[i] or c[i] <= e:
                        pos = 1; entry = e; sl_ = s; tp_ = t; ebar = i
                else:
                    e = pr * (1 - buf); s = pr * (1 + sl_p); risk = s - e
                    if tp_mode == "fixed_rr": t = e - rr * risk
                    elif bb_lo is not None and not np.isnan(bb_lo[i]): t = bb_lo[i]
                    else: t = e - rr * risk
                    if l[i] <= e <= h[i] or c[i] >= e:
                        pos = -1; entry = e; sl_ = s; tp_ = t; ebar = i

    if trades < 2: return {"sharpe": 0, "wr": 0, "trades": trades, "net": 0, "dd": 0, "revs": 0}
    r = np.array(rets)
    return {"sharpe": np.mean(r)/(np.std(r)+1e-12)*np.sqrt(len(r)),
            "wr": wins/trades*100, "trades": trades,
            "net": (equity-1)*100, "dd": mdd*100, "revs": revs}

SYMS = ["btcusdt","ethusdt","solusdt","dogeusdt","nearusdt","adausdt",
        "linkusdt","pepeusdt","suiusdt","xrpusdt","shibusdt"]

configs = [
    ("RR=2 (current)",  "fixed_rr", 2.0, 20, 1.5, False),
    ("RR=3",            "fixed_rr", 3.0, 20, 1.5, False),
    ("RR=1.5",          "fixed_rr", 1.5, 20, 1.5, False),
    ("BB(20,1.5)",      "bb",       2.0, 20, 1.5, False),
    ("BB(20,2.0)",      "bb",       2.0, 20, 2.0, False),
    ("BB(14,1.5)",      "bb",       2.0, 14, 1.5, False),
    ("BB(30,1.5)",      "bb",       2.0, 30, 1.5, False),
    ("RR=2 +reversal",  "fixed_rr", 2.0, 20, 1.5, True),
    ("RR=3 +reversal",  "fixed_rr", 3.0, 20, 1.5, True),
    ("BB(20,1.5)+rev",  "bb",       2.0, 20, 1.5, True),
]

print("=" * 95, flush=True)
print("TRENDLINE: TP MODE + REVERSAL BACKTEST", flush=True)
print("=" * 95, flush=True)

for tf in ["1h", "4h"]:
    print(f"\n--- {tf} ---", flush=True)
    hdr = f"{'config':<20} {'avg_shrp':>9} {'avg_wr%':>8} {'avg_net%':>10} {'avg_DD%':>8} {'avg_tr':>8} {'revs':>6} {'pos/n':>6}"
    print(hdr, flush=True)
    print("-" * 85, flush=True)

    for label, tp_mode, rr, bb_n, bb_std, do_rev in configs:
        results = []
        for sym in SYMS:
            d = load(sym, tf)
            if d is None: continue
            r = backtest_tl(d["open"], d["high"], d["low"], d["close"], d["volume"],
                            tp_mode=tp_mode, rr=rr, bb_n=bb_n, bb_std=bb_std, do_reversal=do_rev)
            if r["trades"] >= 5:
                results.append(r)
        if not results:
            print(f"{label:<20}  (no data)", flush=True)
            continue
        npos = sum(1 for r in results if r["net"] > 0)
        rev = sum(r["revs"] for r in results)
        print(f"{label:<20} {np.mean([r['sharpe'] for r in results]):+8.2f} "
              f"{np.mean([r['wr'] for r in results]):7.1f}% "
              f"{np.mean([r['net'] for r in results]):+9.1f}% "
              f"{np.mean([r['dd'] for r in results]):7.1f}% "
              f"{np.mean([r['trades'] for r in results]):7.0f} "
              f"{rev:>6} "
              f"{npos:>3}/{len(results)}", flush=True)

print("\nDone.", flush=True)
