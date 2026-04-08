"""
ma_ribbon_backtest.py
OKX Top-30 × MA Ribbon (Price>MA5>MA8>EMA21>MA55) Multi-Timeframe Backtest
Run: python ma_ribbon_backtest.py
"""

import httpx, time, numpy as np
from datetime import datetime, timezone
from itertools import product

# ──────────────────────────── OKX DATA ────────────────────────────

def fetch_top30():
    r = httpx.get("https://www.okx.com/api/v5/market/tickers?instType=SWAP", timeout=15)
    data = r.json()["data"]
    usdt = [d for d in data if d["instId"].endswith("-USDT-SWAP")]
    usdt.sort(key=lambda x: float(x.get("volCcy24h", 0) or 0), reverse=True)
    return [d["instId"] for d in usdt[:30]]

def fetch_ohlcv(inst_id, bar="1H", n_pages=4):
    rows = []
    after = ""
    for _ in range(n_pages):
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after:
            params["after"] = after
        try:
            r = httpx.get("https://www.okx.com/api/v5/market/history-candles",
                          params=params, timeout=15)
            chunk = r.json().get("data", [])
        except Exception:
            break
        if not chunk:
            break
        rows.extend(chunk)
        after = chunk[-1][0]
        time.sleep(0.05)
    if not rows:
        try:
            r = httpx.get("https://www.okx.com/api/v5/market/candles",
                          params={"instId": inst_id, "bar": bar, "limit": "300"}, timeout=15)
            rows = r.json().get("data", [])
        except Exception:
            return None
    if not rows:
        return None
    rows.sort(key=lambda x: int(x[0]))
    arr = np.array([[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in rows])
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

# ──────────────────────────── INDICATORS ──────────────────────────

def sma(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.mean(x[i - n + 1:i + 1])
    return out

def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.full(len(x), np.nan)
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def atr(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return sma(tr, n)

def adx_calc(h, l, c, n=14):
    tr  = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c,1)), np.abs(l - np.roll(c,1)))); tr[0] = h[0]-l[0]
    dmp = np.where((h-np.roll(h,1)) > (np.roll(l,1)-l), np.maximum(h-np.roll(h,1),0), 0.0); dmp[0]=0
    dmn = np.where((np.roll(l,1)-l) > (h-np.roll(h,1)), np.maximum(np.roll(l,1)-l,0), 0.0); dmn[0]=0
    atr14 = sma(tr, n)
    dip = 100 * sma(dmp, n) / (atr14 + 1e-12)
    din = 100 * sma(dmn, n) / (atr14 + 1e-12)
    dx  = 100 * np.abs(dip - din) / (dip + din + 1e-12)
    return sma(dx, n)

# ──────────────────────────── RIBBON SIGNALS ──────────────────────

def ribbon_signals(c, ma5_n=5, ma8_n=8, ema21_n=21, ma55_n=55,
                   use_adx=True, adx_min=20, atr_mult=2.0):
    """
    Returns (signals, sl_arr, tp_arr)
    signal +1 = long entry, -1 = short entry, 0 = no action
    """
    n = len(c)
    ma5  = sma(c, ma5_n)
    ma8  = sma(c, ma8_n)
    e21  = ema(c, ema21_n)
    ma55 = sma(c, ma55_n)
    atr14= atr(np.full(n, c.max()), np.full(n, c.min()), c, 14)  # price-only ATR approx

    # Use real ATR: we only have close here; caller passes h,l,c
    adx14= adx_calc(np.roll(c, -1), np.roll(c, 1), c, 14) if use_adx else np.full(n, 99)

    bull = (c > ma5) & (ma5 > ma8) & (ma8 > e21) & (e21 > ma55)
    bear = (c < ma5) & (ma5 < ma8) & (ma8 < e21) & (e21 < ma55)

    adx_ok = adx14 > adx_min

    sig = np.zeros(n)
    for i in range(1, n):
        if bull[i] and not bull[i-1] and adx_ok[i]:
            sig[i] = 1
        elif bear[i] and not bear[i-1] and adx_ok[i]:
            sig[i] = -1

    return sig, bull, bear, ma5, ma8, e21, ma55

def ribbon_signals_full(o, h, l, c, v,
                        ma5_n=5, ma8_n=8, ema21_n=21, ma55_n=55,
                        use_adx=True, adx_min=20,
                        atr_mult=2.0, rr=2.0,
                        vol_filter=True, vol_mult=1.2):
    n    = len(c)
    ma5  = sma(c, ma5_n)
    ma8  = sma(c, ma8_n)
    e21  = ema(c, ema21_n)
    ma55 = sma(c, ma55_n)
    atr14= atr(h, l, c, 14)
    adx14= adx_calc(h, l, c, 14)
    vol_avg = sma(v, 20)

    bull = (c > ma5) & (ma5 > ma8) & (ma8 > e21) & (e21 > ma55)
    bear = (c < ma5) & (ma5 < ma8) & (ma8 < e21) & (e21 < ma55)

    adx_ok = (adx14 > adx_min) if use_adx else np.ones(n, bool)
    vol_ok = (v > vol_mult * vol_avg) if vol_filter else np.ones(n, bool)

    sig = np.zeros(n)
    for i in range(1, n):
        if bull[i] and not bull[i-1] and adx_ok[i] and vol_ok[i]:
            sig[i] = 1
        elif bear[i] and not bear[i-1] and adx_ok[i] and vol_ok[i]:
            sig[i] = -1
    return sig, atr14, rr

# ──────────────────────────── BACKTEST ENGINE ─────────────────────

def backtest(signals, c, atr14=None, atr_mult=2.0, rr=2.0, fee=0.0005):
    pos = 0; entry = 0.0; sl = 0.0; tp = 0.0
    equity = 1.0; peak = 1.0; max_dd = 0.0
    returns = []; wins = 0; trades = 0
    n = len(c)
    _atr = atr14 if atr14 is not None else np.full(n, c.mean() * 0.02)

    for i in range(1, n):
        if pos != 0:
            # Check SL/TP
            hit_sl = (pos == 1 and c[i] <= sl) or (pos == -1 and c[i] >= sl)
            hit_tp = (pos == 1 and c[i] >= tp) or (pos == -1 and c[i] <= tp)

            if hit_tp:
                net = abs(tp - entry) / entry - fee * 2
                equity *= (1 + net); returns.append(net); wins += 1; trades += 1; pos = 0
            elif hit_sl:
                net = -abs(sl - entry) / entry - fee * 2
                equity *= (1 + net); returns.append(net); trades += 1; pos = 0

            # Trail SL for long
            if pos == 1:
                new_sl = c[i] - atr_mult * _atr[i]
                sl = max(sl, new_sl)
            elif pos == -1:
                new_sl = c[i] + atr_mult * _atr[i]
                sl = min(sl, new_sl)

            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)

        if pos == 0 and signals[i] != 0 and not np.isnan(_atr[i]):
            pos   = int(signals[i])
            entry = c[i]
            sl_dist = atr_mult * _atr[i]
            sl = entry - sl_dist if pos == 1 else entry + sl_dist
            tp = entry + rr * sl_dist if pos == 1 else entry - rr * sl_dist

    if trades < 3:
        return 0.0, 0.0, 0.0, trades, max_dd

    r = np.array(returns)
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252 * 24)
    return equity - 1, sharpe, wins / trades, trades, max_dd

# ──────────────────────────── MULTI-TF CONFIG ─────────────────────

TF_CONFIG = {
    # bar_str: (pages, label)
    "5m":  (5, "5m  "),
    "15m": (5, "15m "),
    "1H":  (4, "1H  "),
    "4H":  (4, "4H  "),
    "1D":  (3, "1D  "),
}

# ──────────────────────────── PARAMETER GRID ──────────────────────

PARAM_GRID = [
    # (ma5, ma8, ema21, ma55, atr_mult, rr,   adx_min, vol_filter, label)
    (5,  8,  21, 55,  2.0, 2.0, 20,  True,  "Default (5/8/21/55 ATR2 RR2 ADX20 Vol)"),
    (5,  8,  21, 55,  1.5, 2.0, 20,  True,  "Tight SL (ATR1.5)"),
    (5,  8,  21, 55,  2.5, 3.0, 20,  True,  "Wide SL + RR3"),
    (5,  8,  21, 55,  2.0, 2.0, 25,  True,  "Strict ADX25"),
    (5,  8,  21, 55,  2.0, 2.0, 20,  False, "No Vol Filter"),
    (5,  8,  21, 55,  2.0, 1.5, 20,  True,  "RR 1.5:1"),
    (5,  8,  21, 55,  2.0, 3.0, 20,  True,  "RR 3:1"),
    (3,  8,  21, 55,  2.0, 2.0, 20,  True,  "MA3 fast"),
    (5,  10, 21, 55,  2.0, 2.0, 20,  True,  "MA10 mid"),
    (5,  8,  21, 89,  2.0, 2.0, 20,  True,  "MA89 slow"),
    (5,  8,  21, 55,  2.0, 2.0, 15,  True,  "Loose ADX15"),
    (5,  8,  13, 34,  2.0, 2.0, 20,  True,  "Fib MAs 5/8/13/34"),
]

# ──────────────────────────── MAIN ────────────────────────────────

def main():
    print("=" * 70)
    print("MA Ribbon Backtest: Price>MA5>MA8>EMA21>MA55")
    print("OKX Top-30 × 5 Timeframes × 12 Parameter Sets")
    print("=" * 70)

    print("\n[1] Fetching OKX Top-30 by volume...")
    symbols = fetch_top30()
    print(f"  {len(symbols)} symbols: {', '.join(s.replace('-USDT-SWAP','') for s in symbols)}")

    # results[param_idx][tf] = list of (ret, sharpe, wr, trades, mdd)
    results = {pi: {tf: [] for tf in TF_CONFIG} for pi in range(len(PARAM_GRID))}

    total = len(symbols) * len(TF_CONFIG)
    done  = 0

    print(f"\n[2] Downloading & backtesting ({total} symbol×TF combos)...\n")

    for sym in symbols:
        sym_label = sym.replace("-USDT-SWAP", "")
        for tf, (pages, tf_label) in TF_CONFIG.items():
            done += 1
            print(f"  [{done:3d}/{total}] {sym_label:10s} {tf_label} ...", end=" ", flush=True)

            data = fetch_ohlcv(sym, bar=tf, n_pages=pages)
            if data is None or len(data[3]) < 80:
                print("skip")
                continue

            o, h, l, c, v = data
            print(f"{len(c)}bars", end=" ", flush=True)

            for pi, (m5, m8, e21, m55, atr_m, rr, adx_m, vol_f, _) in enumerate(PARAM_GRID):
                try:
                    sig, atr14, _ = ribbon_signals_full(
                        o, h, l, c, v,
                        ma5_n=m5, ma8_n=m8, ema21_n=e21, ma55_n=m55,
                        use_adx=True, adx_min=adx_m,
                        atr_mult=atr_m, rr=rr,
                        vol_filter=vol_f
                    )
                    ret, sharpe, wr, trades, mdd = backtest(sig, c, atr14, atr_m, rr)
                    results[pi][tf].append((ret, sharpe, wr, trades, mdd))
                except Exception as e:
                    pass
            print("ok")

    # ── Aggregate ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS: Avg Sharpe across Top-30 symbols per TF")
    print("=" * 70)

    # Header
    tf_list = list(TF_CONFIG.keys())
    header = f"{'Params':<45}" + "".join(f"{tf:>10}" for tf in tf_list) + f"{'OVERALL':>10}"
    print(header)
    print("-" * len(header))

    summary = []
    for pi, (m5, m8, e21, m55, atr_m, rr, adx_m, vol_f, label) in enumerate(PARAM_GRID):
        row_sharpes = []
        cells = []
        for tf in tf_list:
            data_list = results[pi][tf]
            if len(data_list) < 3:
                cells.append("     -")
                continue
            sharpes = [x[1] for x in data_list]
            avg_s = np.mean(sharpes)
            row_sharpes.append(avg_s)
            cells.append(f"{avg_s:+8.2f}")

        overall = np.mean(row_sharpes) if row_sharpes else 0
        row = f"{label:<45}" + "".join(f"{c:>10}" for c in cells) + f"{overall:>+10.2f}"
        print(row)
        summary.append((pi, overall, label))

    # ── Top 5 by overall sharpe ────────────────────────────────────
    summary.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 70)
    print("TOP 5 PARAMETER SETS (avg Sharpe across all TFs × symbols)")
    print("=" * 70)

    for rank, (pi, overall_s, label) in enumerate(summary[:5], 1):
        print(f"\n#{rank}  {label}")
        print(f"    MA: {PARAM_GRID[pi][0]}/{PARAM_GRID[pi][1]}/EMA{PARAM_GRID[pi][2]}/{PARAM_GRID[pi][3]} | "
              f"ATR×{PARAM_GRID[pi][4]} | RR {PARAM_GRID[pi][5]}:1 | ADX>{PARAM_GRID[pi][6]} | "
              f"Vol={'✓' if PARAM_GRID[pi][7] else '✗'}")
        print(f"    Overall Sharpe: {overall_s:+.3f}")
        for tf in tf_list:
            data_list = results[pi][tf]
            if len(data_list) < 3:
                continue
            rets    = [x[0]*100 for x in data_list]
            sharpes = [x[1] for x in data_list]
            wrs     = [x[2]*100 for x in data_list]
            trades  = [x[3] for x in data_list]
            mdds    = [x[4]*100 for x in data_list]
            pct_pos = 100 * np.mean(np.array(sharpes) > 0)
            print(f"    {tf:>4}: Sharpe={np.mean(sharpes):+.2f}  Ret={np.mean(rets):+.1f}%  "
                  f"Win={np.mean(wrs):.1f}%  Trades={np.mean(trades):.0f}  "
                  f"MaxDD={np.mean(mdds):.1f}%  Pos%={pct_pos:.0f}%")

    # ── Best TF per param set ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("BEST TIMEFRAME per Top-3 Param Set")
    print("=" * 70)
    for rank, (pi, overall_s, label) in enumerate(summary[:3], 1):
        best_tf = None; best_s = -999
        for tf in tf_list:
            data_list = results[pi][tf]
            if len(data_list) < 3:
                continue
            avg_s = np.mean([x[1] for x in data_list])
            if avg_s > best_s:
                best_s = avg_s; best_tf = tf
        print(f"  #{rank} {label[:40]:<40} → Best TF: {best_tf}  (Sharpe {best_s:+.2f})")

    print(f"\nCompleted at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
