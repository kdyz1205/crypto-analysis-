"""
ma_ribbon_screener.py
全市场扫描: 找出 MA Ribbon 策略 Sharpe > 0 的币种
最优参数: 1H / RR1.5 / ATR2.0 / ADX>20 / Vol过滤
同时跑: 1H + 4H 双周期
"""

import httpx, time, numpy as np
from datetime import datetime, timezone

# ─────────────────────────── OKX DATA ────────────────────────────

def fetch_all_usdt_swap():
    r = httpx.get("https://www.okx.com/api/v5/market/tickers?instType=SWAP", timeout=15)
    data = r.json()["data"]
    usdt = [d for d in data if d["instId"].endswith("-USDT-SWAP")]
    # 按成交量排序，取前100（太多会太慢）
    usdt.sort(key=lambda x: float(x.get("volCcy24h", 0) or 0), reverse=True)
    return [(d["instId"], float(d.get("volCcy24h", 0) or 0)) for d in usdt[:120]]

def fetch_ohlcv(inst_id, bar="1H", n_pages=4):
    rows = []
    after = ""
    for _ in range(n_pages):
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after:
            params["after"] = after
        try:
            r = httpx.get("https://www.okx.com/api/v5/market/history-candles",
                          params=params, timeout=12)
            chunk = r.json().get("data", [])
        except Exception:
            break
        if not chunk:
            break
        rows.extend(chunk)
        after = chunk[-1][0]
        time.sleep(0.04)
    if not rows:
        try:
            r = httpx.get("https://www.okx.com/api/v5/market/candles",
                          params={"instId": inst_id, "bar": bar, "limit": "300"}, timeout=12)
            rows = r.json().get("data", [])
        except Exception:
            return None
    if not rows:
        return None
    rows.sort(key=lambda x: int(x[0]))
    arr = np.array([[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in rows])
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

# ─────────────────────────── INDICATORS ──────────────────────────

def sma(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.mean(x[i - n + 1:i + 1])
    return out

def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def atr_calc(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return sma(tr, n)

def adx_calc(h, l, c, n=14):
    tr  = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1)))); tr[0]=h[0]-l[0]
    dmp = np.where((h-np.roll(h,1))>(np.roll(l,1)-l), np.maximum(h-np.roll(h,1),0), 0.0); dmp[0]=0
    dmn = np.where((np.roll(l,1)-l)>(h-np.roll(h,1)), np.maximum(np.roll(l,1)-l,0), 0.0); dmn[0]=0
    atr14 = sma(tr, n)
    dip = 100 * sma(dmp, n) / (atr14 + 1e-12)
    din = 100 * sma(dmn, n) / (atr14 + 1e-12)
    dx  = 100 * np.abs(dip-din) / (dip+din+1e-12)
    return sma(dx, n)

# ─────────────────────────── BACKTEST ────────────────────────────

def run_backtest(o, h, l, c, v, atr_mult=2.0, rr=1.5, adx_min=20, vol_mult=1.2, fee=0.0005):
    n = len(c)
    if n < 80:
        return None

    ma5  = sma(c, 5)
    ma8  = sma(c, 8)
    e21  = ema(c, 21)
    ma55 = sma(c, 55)
    atr14= atr_calc(h, l, c, 14)
    adx14= adx_calc(h, l, c, 14)
    vol_avg = sma(v, 20)

    bull = (c > ma5) & (ma5 > ma8) & (ma8 > e21) & (e21 > ma55)
    bear = (c < ma5) & (ma5 < ma8) & (ma8 < e21) & (e21 < ma55)

    adx_ok = adx14 > adx_min
    vol_ok = v > vol_mult * vol_avg

    pos = 0; entry = 0.0; sl = 0.0; tp = 0.0
    equity = 1.0; peak = 1.0; max_dd = 0.0
    returns = []; wins = 0; trades = 0

    for i in range(1, n):
        if np.isnan(ma55[i]) or np.isnan(atr14[i]):
            continue

        if pos != 0:
            hit_sl = (pos == 1 and c[i] <= sl) or (pos == -1 and c[i] >= sl)
            hit_tp = (pos == 1 and c[i] >= tp) or (pos == -1 and c[i] <= tp)
            reverse = (pos == 1 and bear[i]) or (pos == -1 and bull[i])

            if hit_tp:
                net = abs(tp - entry) / entry - fee * 2
                equity *= (1 + net); returns.append(net); wins += 1; trades += 1; pos = 0
            elif hit_sl:
                net = -(abs(sl - entry) / entry) - fee * 2
                equity *= (1 + net); returns.append(net); trades += 1; pos = 0
            elif reverse:
                net = pos * (c[i] - entry) / entry - fee * 2
                equity *= (1 + net); returns.append(net)
                if net > 0: wins += 1
                trades += 1; pos = 0
            else:
                # trail SL
                if pos == 1:
                    sl = max(sl, c[i] - atr_mult * atr14[i])
                else:
                    sl = min(sl, c[i] + atr_mult * atr14[i])

            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)

        if pos == 0:
            entry_long  = bull[i] and not bull[i-1] and adx_ok[i] and vol_ok[i]
            entry_short = bear[i] and not bear[i-1] and adx_ok[i] and vol_ok[i]
            if entry_long:
                pos = 1; entry = c[i]
                sl = entry - atr_mult * atr14[i]
                tp = entry + rr * atr_mult * atr14[i]
            elif entry_short:
                pos = -1; entry = c[i]
                sl = entry + atr_mult * atr14[i]
                tp = entry - rr * atr_mult * atr14[i]

    if trades < 4:
        return None

    r = np.array(returns)
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252 * 24)
    total_ret = (equity - 1) * 100
    win_rate  = wins / trades * 100
    return {
        "sharpe": sharpe,
        "return": total_ret,
        "win_rate": win_rate,
        "trades": trades,
        "max_dd": max_dd * 100,
        "profit_factor": sum(x for x in r if x > 0) / (abs(sum(x for x in r if x < 0)) + 1e-12),
    }

# ─────────────────────────── MAIN ────────────────────────────────

def main():
    print("=" * 65)
    print("MA Ribbon Screener: 全市场扫描 Sharpe > 0 的币种")
    print("策略: Price>MA5>MA8>EMA21>MA55 | 1H + 4H | RR1.5 ATR2.0 ADX>20")
    print("=" * 65)

    print("\n[1] 获取 OKX 全市场 USDT-SWAP...")
    all_syms = fetch_all_usdt_swap()
    print(f"  共 {len(all_syms)} 个交易对")

    TFS = [("1H", 4), ("4H", 4)]
    winners_1h = []
    winners_4h = []
    both_positive = []

    total = len(all_syms)
    print(f"\n[2] 扫描中... ({total} 币 × 2 TF)\n")

    for idx, (sym, vol24h) in enumerate(all_syms):
        label = sym.replace("-USDT-SWAP", "")
        print(f"  [{idx+1:3d}/{total}] {label:12s}", end=" ", flush=True)

        r1h = None; r4h = None

        for tf, pages in TFS:
            data = fetch_ohlcv(sym, bar=tf, n_pages=pages)
            if data is None:
                print(f"{tf}:skip ", end="", flush=True)
                continue
            o, h, l, c, v = data
            res = run_backtest(o, h, l, c, v)
            if res is None:
                print(f"{tf}:few  ", end="", flush=True)
                continue
            s = res["sharpe"]
            print(f"{tf}:{s:+.1f} ", end="", flush=True)
            if tf == "1H":
                r1h = res; r1h["sym"] = label; r1h["vol24h"] = vol24h
            else:
                r4h = res; r4h["sym"] = label; r4h["vol24h"] = vol24h

        if r1h and r1h["sharpe"] > 0:
            winners_1h.append(r1h)
        if r4h and r4h["sharpe"] > 0:
            winners_4h.append(r4h)
        if r1h and r4h and r1h["sharpe"] > 0 and r4h["sharpe"] > 0:
            both_positive.append({
                "sym": label,
                "sharpe_1h": r1h["sharpe"],
                "sharpe_4h": r4h["sharpe"],
                "combined": r1h["sharpe"] + r4h["sharpe"],
                "ret_1h": r1h["return"],
                "ret_4h": r4h["return"],
                "wr_1h": r1h["win_rate"],
                "wr_4h": r4h["win_rate"],
                "mdd_1h": r1h["max_dd"],
                "mdd_4h": r4h["max_dd"],
                "trades_1h": r1h["trades"],
                "trades_4h": r4h["trades"],
                "pf_1h": r1h["profit_factor"],
                "vol24h": vol24h,
            })
        print()

    # ── RESULTS ──────────────────────────────────────────────────
    winners_1h.sort(key=lambda x: x["sharpe"], reverse=True)
    winners_4h.sort(key=lambda x: x["sharpe"], reverse=True)
    both_positive.sort(key=lambda x: x["combined"], reverse=True)

    print("\n" + "=" * 65)
    print(f"1H 正 Sharpe: {len(winners_1h)} 个币  |  4H 正 Sharpe: {len(winners_4h)} 个币")
    print(f"双周期都正:   {len(both_positive)} 个币  ← 最可靠")
    print("=" * 65)

    if both_positive:
        print("\n★ 双周期 (1H + 4H) 均正 Sharpe — 最强币种:")
        print(f"  {'币种':<12} {'Sharpe1H':>9} {'Sharpe4H':>9} {'Ret1H':>8} {'Win1H':>7} {'MDD1H':>7} {'Trades1H':>9} {'PF1H':>6}")
        print("  " + "-" * 72)
        for r in both_positive[:20]:
            print(f"  {r['sym']:<12} {r['sharpe_1h']:>+9.2f} {r['sharpe_4h']:>+9.2f} "
                  f"{r['ret_1h']:>+7.1f}% {r['wr_1h']:>6.1f}% {r['mdd_1h']:>6.1f}% "
                  f"{r['trades_1h']:>8.0f}  {r['pf_1h']:>5.2f}x")

    print(f"\n★ 1H Top-20 (按 Sharpe 排):")
    print(f"  {'币种':<12} {'Sharpe':>8} {'Return':>8} {'Win%':>7} {'MaxDD':>7} {'Trades':>8} {'PF':>6}")
    print("  " + "-" * 62)
    for r in winners_1h[:20]:
        print(f"  {r['sym']:<12} {r['sharpe']:>+8.2f} {r['return']:>+7.1f}% "
              f"{r['win_rate']:>6.1f}% {r['max_dd']:>6.1f}% {r['trades']:>7}  {r['profit_factor']:>5.2f}x")

    print(f"\n★ 4H Top-20 (按 Sharpe 排):")
    print(f"  {'币种':<12} {'Sharpe':>8} {'Return':>8} {'Win%':>7} {'MaxDD':>7} {'Trades':>8} {'PF':>6}")
    print("  " + "-" * 62)
    for r in winners_4h[:20]:
        print(f"  {r['sym']:<12} {r['sharpe']:>+8.2f} {r['return']:>+7.1f}% "
              f"{r['win_rate']:>6.1f}% {r['max_dd']:>6.1f}% {r['trades']:>7}  {r['profit_factor']:>5.2f}x")

    print(f"\n完成于 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

if __name__ == "__main__":
    main()
