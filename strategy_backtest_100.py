"""
strategy_backtest_100.py
OKX Top-30 Volume Pairs × 100 Strategies backtest
Run: python strategy_backtest_100.py
"""

import httpx, time, numpy as np, sys
from datetime import datetime, timezone

# ─────────────────────────── OKX DATA ────────────────────────────

def fetch_top30():
    url = "https://www.okx.com/api/v5/market/tickers?instType=SWAP"
    r = httpx.get(url, timeout=15)
    data = r.json()["data"]
    usdt = [d for d in data if d["instId"].endswith("-USDT-SWAP")]
    usdt.sort(key=lambda x: float(x.get("volCcy24h", 0) or 0), reverse=True)
    return [d["instId"] for d in usdt[:30]]

def fetch_ohlcv(inst_id, bar="1H", limit=500):
    """Returns numpy arrays: open, high, low, close, volume (oldest→newest)"""
    url = "https://www.okx.com/api/v5/market/history-candles"
    rows = []
    after = ""
    for _ in range(4):  # up to 4 pages × 100 = 400 bars (history endpoint max 100)
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after:
            params["after"] = after
        try:
            r = httpx.get(url, params=params, timeout=15)
            chunk = r.json().get("data", [])
        except Exception:
            break
        if not chunk:
            break
        rows.extend(chunk)
        after = chunk[-1][0]
        time.sleep(0.05)
    if not rows:
        # fallback to recent candles endpoint
        url2 = "https://www.okx.com/api/v5/market/candles"
        try:
            r = httpx.get(url2, params={"instId": inst_id, "bar": bar, "limit": "300"}, timeout=15)
            rows = r.json().get("data", [])
        except Exception:
            return None
    if not rows:
        return None
    rows.sort(key=lambda x: int(x[0]))  # ascending
    arr = np.array([[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in rows])
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]  # O,H,L,C,V

# ─────────────────────────── INDICATORS ──────────────────────────

def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.full_like(x, np.nan)
    start = n - 1
    out[start] = np.mean(x[:n])
    for i in range(start + 1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def sma(x, n):
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.mean(x[i - n + 1:i + 1])
    return out

def atr(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return sma(tr, n)

def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    avg_g = sma(g, n)
    avg_l = sma(lo, n)
    rs = np.where(avg_l > 1e-12, avg_g / avg_l, 100.0)
    return 100 - 100 / (1 + rs)

def macd(c, fast=12, slow=26, sig=9):
    e_fast = ema(c, fast)
    e_slow = ema(c, slow)
    m = e_fast - e_slow
    s = ema(np.where(np.isnan(m), 0, m), sig)
    hist = m - s
    return m, s, hist

def bb(c, n=20, k=2.0):
    mid = sma(c, n)
    std = np.array([np.std(c[max(0, i - n + 1):i + 1]) if i >= n - 1 else np.nan for i in range(len(c))])
    return mid - k * std, mid, mid + k * std

def stoch(h, l, c, k=14, d=3):
    out = np.full_like(c, np.nan)
    for i in range(k - 1, len(c)):
        lo = np.min(l[i - k + 1:i + 1])
        hi = np.max(h[i - k + 1:i + 1])
        out[i] = 100 * (c[i] - lo) / (hi - lo + 1e-12)
    return out, sma(out, d)

def obv(c, v):
    d = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(d * v)

def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    out = np.full_like(c, np.nan)
    for i in range(n - 1, len(c)):
        sl = tp[i - n + 1:i + 1]
        mean_dev = np.mean(np.abs(sl - np.mean(sl)))
        out[i] = (tp[i] - np.mean(sl)) / (0.015 * mean_dev + 1e-12)
    return out

def mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3
    rmf = tp * v
    pos = np.where(tp > np.roll(tp, 1), rmf, 0.0); pos[0] = 0
    neg = np.where(tp < np.roll(tp, 1), rmf, 0.0); neg[0] = 0
    out = np.full_like(c, np.nan)
    for i in range(n - 1, len(c)):
        ps = np.sum(pos[i - n + 1:i + 1])
        ns = np.sum(neg[i - n + 1:i + 1])
        out[i] = 100 - 100 / (1 + ps / (ns + 1e-12))
    return out

def donchian(h, l, n=20):
    dh = np.array([np.max(h[max(0, i - n + 1):i + 1]) if i >= n - 1 else np.nan for i in range(len(h))])
    dl = np.array([np.min(l[max(0, i - n + 1):i + 1]) if i >= n - 1 else np.nan for i in range(len(l))])
    return dh, dl

def adx(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    dmp = np.where((h - np.roll(h, 1)) > (np.roll(l, 1) - l), np.maximum(h - np.roll(h, 1), 0), 0.0)
    dmn = np.where((np.roll(l, 1) - l) > (h - np.roll(h, 1)), np.maximum(np.roll(l, 1) - l, 0), 0.0)
    dmp[0] = dmn[0] = 0
    atr14 = sma(tr, n)
    dip = 100 * sma(dmp, n) / (atr14 + 1e-12)
    din = 100 * sma(dmn, n) / (atr14 + 1e-12)
    dx = 100 * abs(dip - din) / (dip + din + 1e-12)
    adx_ = sma(dx, n)
    return adx_, dip, din

def supertrend(h, l, c, n=10, mult=3.0):
    a = atr(h, l, c, n)
    hl2 = (h + l) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    st = np.full_like(c, np.nan)
    trend = np.zeros(len(c), dtype=int)
    for i in range(1, len(c)):
        if np.isnan(a[i]):
            continue
        upper[i] = min(upper[i], upper[i-1]) if c[i-1] > upper[i-1] else upper[i]  # type: ignore
        lower[i] = max(lower[i], lower[i-1]) if c[i-1] < lower[i-1] else lower[i]  # type: ignore
        if trend[i-1] == 1:
            trend[i] = -1 if c[i] < lower[i] else 1
        else:
            trend[i] = 1 if c[i] > upper[i] else -1
        st[i] = lower[i] if trend[i] == 1 else upper[i]
    return st, trend

def keltner(c, h, l, n=20, mult=2.0):
    mid = ema(c, n)
    a = atr(h, l, c, n)
    return mid - mult * a, mid, mid + mult * a

def hma(x, n):
    wma1 = sma(x, n // 2)  # approx WMA with SMA
    wma2 = sma(x, n)
    return sma(2 * wma1 - wma2, int(np.sqrt(n)))

def vwap(h, l, c, v):
    tp = (h + l + c) / 3
    return np.cumsum(tp * v) / np.cumsum(v)

def roc(c, n=10):
    out = np.full_like(c, np.nan)
    out[n:] = (c[n:] - c[:-n]) / (c[:-n] + 1e-12) * 100
    return out

def williams_r(h, l, c, n=14):
    out = np.full_like(c, np.nan)
    for i in range(n - 1, len(c)):
        hi = np.max(h[i - n + 1:i + 1])
        lo = np.min(l[i - n + 1:i + 1])
        out[i] = -100 * (hi - c[i]) / (hi - lo + 1e-12)
    return out

def trix(c, n=15):
    e1 = ema(c, n)
    e2 = ema(np.where(np.isnan(e1), 0, e1), n)
    e3 = ema(np.where(np.isnan(e2), 0, e2), n)
    return np.diff(e3, prepend=e3[0]) / (e3 + 1e-12) * 100

def aroon(h, l, n=25):
    up = np.full_like(h, np.nan)
    dn = np.full_like(l, np.nan)
    for i in range(n, len(h)):
        sl_h = h[i - n:i + 1]
        sl_l = l[i - n:i + 1]
        up[i] = 100 * (n - (n - np.argmax(sl_h))) / n
        dn[i] = 100 * (n - (n - np.argmin(sl_l))) / n
    return up, dn

def zscore(c, n=50):
    out = np.full_like(c, np.nan)
    for i in range(n - 1, len(c)):
        sl = c[i - n + 1:i + 1]
        s = np.std(sl)
        out[i] = (c[i] - np.mean(sl)) / (s + 1e-12)
    return out

# ─────────────────────────── BACKTEST ENGINE ─────────────────────

def backtest(signals, c, sl_pct=0.02, tp_pct=0.04, fee=0.0005):
    """
    signals: +1 = buy, -1 = sell/short, 0 = flat
    Returns: total_return, sharpe, win_rate, num_trades, max_dd
    """
    pos = 0
    entry = 0.0
    equity = 1.0
    returns = []
    wins = 0
    trades = 0
    peak = 1.0
    max_dd = 0.0

    for i in range(1, len(c)):
        if pos != 0:
            pnl = pos * (c[i] - entry) / entry
            if (pos == 1 and c[i] < entry * (1 - sl_pct)) or \
               (pos == -1 and c[i] > entry * (1 + sl_pct)):
                pnl = pos * (entry * (1 - sl_pct) - entry) / entry if pos == 1 else \
                      pos * (entry * (1 + sl_pct) - entry) / entry
                pos = 0
            elif (pos == 1 and c[i] > entry * (1 + tp_pct)) or \
                 (pos == -1 and c[i] < entry * (1 - tp_pct)):
                pnl = abs(tp_pct)
                pos = 0
            else:
                pnl = 0

            if pnl != 0:
                net = pnl - fee * 2
                equity *= (1 + net)
                returns.append(net)
                trades += 1
                if net > 0:
                    wins += 1
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
                pos = 0

        if pos == 0 and not np.isnan(signals[i]) and signals[i] != 0:
            pos = int(signals[i])
            entry = c[i]

    if len(returns) < 5:
        return 0, 0, 0, trades, max_dd

    r = np.array(returns)
    total_ret = equity - 1
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252 * 24)
    win_rate = wins / trades if trades > 0 else 0
    return total_ret, sharpe, win_rate, trades, max_dd


# ─────────────────────────── 100 STRATEGIES ──────────────────────

def get_all_strategies(o, h, l, c, v):
    """Returns list of (strategy_id, name, signals)"""
    strats = []

    def add(sid, name, sig):
        strats.append((sid, name, sig))

    n = len(c)
    nan = np.full(n, np.nan)

    # Pre-compute common indicators
    e9   = ema(c, 9);   e21  = ema(c, 21);  e50  = ema(c, 50)
    e200 = ema(c, 200); e8   = ema(c, 8);   e55  = ema(c, 55)
    s20  = sma(c, 20);  s50  = sma(c, 50)
    atr14 = atr(h, l, c, 14)
    rsi14 = rsi(c, 14); rsi5 = rsi(c, 5)
    ml, ms, mh = macd(c)
    bbl, bbm, bbu = bb(c, 20, 2)
    stk, std_ = stoch(h, l, c)
    adx14, dip, din = adx(h, l, c, 14)
    obv_ = obv(c, v)
    cci20 = cci(h, l, c, 20)
    mfi14 = mfi(h, l, c, v, 14)
    wr14  = williams_r(h, l, c, 14)
    roc10 = roc(c, 10)
    trix15= trix(c, 15)
    aroon_up, aroon_dn = aroon(h, l, 25)
    zs50  = zscore(c, 50)
    st_, trend_ = supertrend(h, l, c, 10, 3.0)
    dch, dcl  = donchian(h, l, 20)
    kl, km, ku = keltner(c, h, l, 20, 2.0)
    vwap_ = vwap(h, l, c, v)
    hma9  = hma(c, 9);  hma21 = hma(c, 21)
    e_trix= ema(trix15, 9)
    bbl3, bbm3, bbu3 = bb(c, 20, 1.5)

    # 1. EMA crossover (9/21) + above EMA200
    sig = np.zeros(n)
    sig[1:] = np.where((e9[1:] > e21[1:]) & (e9[:-1] <= e21[:-1]) & (c[1:] > e200[1:]), 1,
              np.where((e9[1:] < e21[1:]) & (e9[:-1] >= e21[:-1]), -1, 0))
    add(1, "EMA Cross 9/21 + EMA200 filter", sig)

    # 2. Triple EMA (8/21/55 aligned)
    sig = np.zeros(n)
    sig[1:] = np.where((e8[1:] > e21[1:]) & (e21[1:] > e55[1:]) & (e8[:-1] <= e21[:-1]), 1,
              np.where((e8[1:] < e21[1:]) & (e21[1:] < e55[1:]) & (e8[:-1] >= e21[:-1]), -1, 0))
    add(2, "Triple EMA 8/21/55 aligned", sig)

    # 3. Supertrend
    sig = np.zeros(n)
    sig[1:] = np.where((trend_[1:] == 1) & (trend_[:-1] == -1), 1,
              np.where((trend_[1:] == -1) & (trend_[:-1] == 1), -1, 0))
    add(3, "Supertrend ATR10 mult3.0", sig)

    # 4. Supertrend + RSI filter
    sig = np.zeros(n)
    sig[1:] = np.where((trend_[1:] == 1) & (trend_[:-1] == -1) & (rsi14[1:] > 50), 1,
              np.where((trend_[1:] == -1) & (trend_[:-1] == 1) & (rsi14[1:] < 50), -1, 0))
    add(4, "Supertrend + RSI50 filter", sig)

    # 5. Donchian 20-period breakout
    sig = np.zeros(n)
    sig[1:] = np.where(c[1:] > dch[:-1], 1,
              np.where(c[1:] < dcl[:-1], -1, 0))
    add(5, "Donchian Channel 20 breakout", sig)

    # 6. Keltner Channel trend
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] > ku[1:]) & (c[:-1] <= ku[:-1]), 1,
              np.where((c[1:] < kl[1:]) & (c[:-1] >= kl[:-1]), -1, 0))
    add(6, "Keltner Channel breakout", sig)

    # 7. ADX + DI cross
    sig = np.zeros(n)
    sig[1:] = np.where((dip[1:] > din[1:]) & (dip[:-1] <= din[:-1]) & (adx14[1:] > 20), 1,
              np.where((dip[1:] < din[1:]) & (dip[:-1] >= din[:-1]) & (adx14[1:] > 20), -1, 0))
    add(7, "ADX+DI cross ADX>20", sig)

    # 8. ADX + DI strict ADX>25
    sig = np.zeros(n)
    sig[1:] = np.where((dip[1:] > din[1:]) & (dip[:-1] <= din[:-1]) & (adx14[1:] > 25), 1,
              np.where((dip[1:] < din[1:]) & (dip[:-1] >= din[:-1]) & (adx14[1:] > 25), -1, 0))
    add(8, "ADX+DI cross ADX>25", sig)

    # 9. MACD histogram crossover
    sig = np.zeros(n)
    sig[1:] = np.where((mh[1:] > 0) & (mh[:-1] <= 0), 1,
              np.where((mh[1:] < 0) & (mh[:-1] >= 0), -1, 0))
    add(9, "MACD histogram crossover", sig)

    # 10. MACD line crossover signal
    sig = np.zeros(n)
    sig[1:] = np.where((ml[1:] > ms[1:]) & (ml[:-1] <= ms[:-1]), 1,
              np.where((ml[1:] < ms[1:]) & (ml[:-1] >= ms[:-1]), -1, 0))
    add(10, "MACD line cross signal", sig)

    # 11. MACD crossover + EMA200
    sig = np.zeros(n)
    sig[1:] = np.where((ml[1:] > ms[1:]) & (ml[:-1] <= ms[:-1]) & (c[1:] > e200[1:]), 1,
              np.where((ml[1:] < ms[1:]) & (ml[:-1] >= ms[:-1]) & (c[1:] < e200[1:]), -1, 0))
    add(11, "MACD cross + EMA200 filter", sig)

    # 12. Bollinger Band reversion
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] < bbl[1:]) & (rsi14[1:] < 35), 1,
              np.where((c[1:] > bbu[1:]) & (rsi14[1:] > 65), -1, 0))
    add(12, "BB reversion + RSI filter", sig)

    # 13. BB squeeze breakout (bandwidth)
    bw = (bbu - bbl) / (bbm + 1e-12)
    bw_min = np.array([np.min(bw[max(0, i-100):i+1]) if i >= 20 else np.nan for i in range(n)])
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] > bbu[1:]) & (np.abs(bw[1:] - bw_min[1:]) < 0.001), 1,
              np.where((c[1:] < bbl[1:]) & (np.abs(bw[1:] - bw_min[1:]) < 0.001), -1, 0))
    add(13, "BB squeeze breakout", sig)

    # 14. RSI mean reversion (ADX<20)
    sig = np.zeros(n)
    sig[1:] = np.where((rsi14[1:] < 30) & (adx14[1:] < 20), 1,
              np.where((rsi14[1:] > 70) & (adx14[1:] < 20), -1, 0))
    add(14, "RSI reversion ADX<20", sig)

    # 15. RSI trend (ADX>25)
    sig = np.zeros(n)
    sig[1:] = np.where((rsi14[1:] > 55) & (rsi14[:-1] <= 55) & (adx14[1:] > 25), 1,
              np.where((rsi14[1:] < 45) & (rsi14[:-1] >= 45) & (adx14[1:] > 25), -1, 0))
    add(15, "RSI midline cross + ADX>25", sig)

    # 16. Stochastic crossover
    sig = np.zeros(n)
    sig[1:] = np.where((stk[1:] > std_[1:]) & (stk[:-1] <= std_[:-1]) & (stk[1:] < 50), 1,
              np.where((stk[1:] < std_[1:]) & (stk[:-1] >= std_[:-1]) & (stk[1:] > 50), -1, 0))
    add(16, "Stochastic K/D cross <50", sig)

    # 17. Stochastic oversold bounce
    sig = np.zeros(n)
    sig[1:] = np.where((stk[1:] > std_[1:]) & (stk[:-1] <= std_[:-1]) & (stk[1:] < 25), 1,
              np.where((stk[1:] < std_[1:]) & (stk[:-1] >= std_[:-1]) & (stk[1:] > 75), -1, 0))
    add(17, "Stochastic oversold <25 / overbought >75", sig)

    # 18. CCI breakout
    sig = np.zeros(n)
    sig[1:] = np.where((cci20[1:] > 100) & (cci20[:-1] <= 100), 1,
              np.where((cci20[1:] < -100) & (cci20[:-1] >= -100), -1, 0))
    add(18, "CCI ±100 breakout", sig)

    # 19. CCI reversion
    sig = np.zeros(n)
    sig[1:] = np.where((cci20[1:] > -100) & (cci20[:-1] <= -100), 1,
              np.where((cci20[1:] < 100) & (cci20[:-1] >= 100), -1, 0))
    add(19, "CCI ±100 reversion", sig)

    # 20. MFI oversold
    sig = np.zeros(n)
    sig[1:] = np.where((mfi14[1:] > 20) & (mfi14[:-1] <= 20), 1,
              np.where((mfi14[1:] < 80) & (mfi14[:-1] >= 80), -1, 0))
    add(20, "MFI oversold <20 / overbought >80", sig)

    # 21. Williams %R reversion
    sig = np.zeros(n)
    sig[1:] = np.where((wr14[1:] > -80) & (wr14[:-1] <= -80), 1,
              np.where((wr14[1:] < -20) & (wr14[:-1] >= -20), -1, 0))
    add(21, "Williams %R -80/-20 reversion", sig)

    # 22. OBV breakout (OBV > EMA of OBV)
    obv_e = ema(obv_, 20)
    sig = np.zeros(n)
    sig[1:] = np.where((obv_[1:] > obv_e[1:]) & (obv_[:-1] <= obv_e[:-1]), 1,
              np.where((obv_[1:] < obv_e[1:]) & (obv_[:-1] >= obv_e[:-1]), -1, 0))
    add(22, "OBV cross EMA20", sig)

    # 23. OBV divergence (price new low, OBV higher low)
    sig = np.zeros(n)
    for i in range(20, n):
        prev_low = np.argmin(c[i-20:i]) + (i-20)
        if c[i] < c[prev_low] and obv_[i] > obv_[prev_low]:
            sig[i] = 1
        elif c[i] > np.max(c[i-20:i]) and obv_[i] < np.max(obv_[i-20:i]):
            sig[i] = -1
    add(23, "OBV divergence 20-bar", sig)

    # 24. VWAP bounce
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] > vwap_[1:]) & (c[:-1] <= vwap_[:-1]) & (c[1:] > e50[1:]), 1,
              np.where((c[1:] < vwap_[1:]) & (c[:-1] >= vwap_[:-1]) & (c[1:] < e50[1:]), -1, 0))
    add(24, "VWAP cross + EMA50 trend", sig)

    # 25. ROC momentum
    sig = np.zeros(n)
    sig[1:] = np.where((roc10[1:] > 3) & (roc10[:-1] <= 3) & (c[1:] > e50[1:]), 1,
              np.where((roc10[1:] < -3) & (roc10[:-1] >= -3) & (c[1:] < e50[1:]), -1, 0))
    add(25, "ROC 10-period ±3% + trend", sig)

    # 26. TRIX crossover
    sig = np.zeros(n)
    sig[1:] = np.where((trix15[1:] > e_trix[1:]) & (trix15[:-1] <= e_trix[:-1]), 1,
              np.where((trix15[1:] < e_trix[1:]) & (trix15[:-1] >= e_trix[:-1]), -1, 0))
    add(26, "TRIX cross signal line", sig)

    # 27. TRIX zero cross
    sig = np.zeros(n)
    sig[1:] = np.where((trix15[1:] > 0) & (trix15[:-1] <= 0), 1,
              np.where((trix15[1:] < 0) & (trix15[:-1] >= 0), -1, 0))
    add(27, "TRIX zero cross", sig)

    # 28. Aroon crossover
    sig = np.zeros(n)
    sig[1:] = np.where((aroon_up[1:] > aroon_dn[1:]) & (aroon_up[:-1] <= aroon_dn[:-1]), 1,
              np.where((aroon_up[1:] < aroon_dn[1:]) & (aroon_up[:-1] >= aroon_dn[:-1]), -1, 0))
    add(28, "Aroon Up/Down cross", sig)

    # 29. Z-score mean reversion
    sig = np.zeros(n)
    sig[1:] = np.where((zs50[1:] > -2) & (zs50[:-1] <= -2), 1,
              np.where((zs50[1:] < 2) & (zs50[:-1] >= 2), -1, 0))
    add(29, "Z-score ±2 reversion 50-bar", sig)

    # 30. Z-score trend (cross 0)
    sig = np.zeros(n)
    sig[1:] = np.where((zs50[1:] > 0) & (zs50[:-1] <= 0), 1,
              np.where((zs50[1:] < 0) & (zs50[:-1] >= 0), -1, 0))
    add(30, "Z-score 50-bar zero cross", sig)

    # 31. HMA crossover
    sig = np.zeros(n)
    sig[1:] = np.where((hma9[1:] > hma21[1:]) & (hma9[:-1] <= hma21[:-1]), 1,
              np.where((hma9[1:] < hma21[1:]) & (hma9[:-1] >= hma21[:-1]), -1, 0))
    add(31, "HMA 9/21 cross", sig)

    # 32. EMA 50/200 golden/death cross
    sig = np.zeros(n)
    sig[1:] = np.where((e50[1:] > e200[1:]) & (e50[:-1] <= e200[:-1]), 1,
              np.where((e50[1:] < e200[1:]) & (e50[:-1] >= e200[:-1]), -1, 0))
    add(32, "EMA 50/200 golden/death cross", sig)

    # 33. Price vs EMA200 + RSI confirmation
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] > e200[1:]) & (c[:-1] <= e200[:-1]) & (rsi14[1:] > 50), 1,
              np.where((c[1:] < e200[1:]) & (c[:-1] >= e200[:-1]) & (rsi14[1:] < 50), -1, 0))
    add(33, "Price cross EMA200 + RSI", sig)

    # 34. BB + RSI combo (tight)
    sig = np.zeros(n)
    sig[1:] = np.where((rsi5[1:] < 20) & (c[1:] < bbl[1:]), 1,
              np.where((rsi5[1:] > 80) & (c[1:] > bbu[1:]), -1, 0))
    add(34, "BB+RSI5 extreme (double confirm)", sig)

    # 35. Ichimoku-like (EMA9/26/52 cloud approx)
    e26 = ema(c, 26); e52 = ema(c, 52)
    cloud_top = np.maximum(e9, e26)
    cloud_bot = np.minimum(e9, e26)
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] > cloud_top[1:]) & (c[:-1] <= cloud_top[:-1]) & (e52[1:] < c[1:]), 1,
              np.where((c[1:] < cloud_bot[1:]) & (c[:-1] >= cloud_bot[:-1]) & (e52[1:] > c[1:]), -1, 0))
    add(35, "Ichimoku-approx EMA cloud", sig)

    # 36. Volume spike + direction
    v_ma20 = sma(v, 20)
    sig = np.zeros(n)
    sig[1:] = np.where((v[1:] > 2 * v_ma20[1:]) & (c[1:] > o[1:]), 1,
              np.where((v[1:] > 2 * v_ma20[1:]) & (c[1:] < o[1:]), -1, 0))
    add(36, "Volume spike 2x + direction", sig)

    # 37. Volume spike 1.5x + trend
    sig = np.zeros(n)
    sig[1:] = np.where((v[1:] > 1.5 * v_ma20[1:]) & (c[1:] > e50[1:]) & (c[1:] > c[:-1]), 1,
              np.where((v[1:] > 1.5 * v_ma20[1:]) & (c[1:] < e50[1:]) & (c[1:] < c[:-1]), -1, 0))
    add(37, "Volume 1.5x + EMA50 trend", sig)

    # 38. MACD + RSI combo
    sig = np.zeros(n)
    sig[1:] = np.where((ml[1:] > ms[1:]) & (ml[:-1] <= ms[:-1]) & (rsi14[1:] > 50), 1,
              np.where((ml[1:] < ms[1:]) & (ml[:-1] >= ms[:-1]) & (rsi14[1:] < 50), -1, 0))
    add(38, "MACD cross + RSI50", sig)

    # 39. Triple indicator: MACD + RSI + EMA trend
    sig = np.zeros(n)
    sig[1:] = np.where((mh[1:] > 0) & (mh[:-1] <= 0) & (rsi14[1:] > 50) & (c[1:] > e200[1:]), 1,
              np.where((mh[1:] < 0) & (mh[:-1] >= 0) & (rsi14[1:] < 50) & (c[1:] < e200[1:]), -1, 0))
    add(39, "MACD hist + RSI50 + EMA200", sig)

    # 40. Stoch + RSI combo
    sig = np.zeros(n)
    sig[1:] = np.where((stk[1:] < 20) & (rsi14[1:] < 35), 1,
              np.where((stk[1:] > 80) & (rsi14[1:] > 65), -1, 0))
    add(40, "Stoch<20 + RSI<35 double-OS", sig)

    # 41. ATR breakout
    sig = np.zeros(n)
    sig[1:] = np.where(c[1:] > c[:-1] + 1.5 * atr14[:-1], 1,
              np.where(c[1:] < c[:-1] - 1.5 * atr14[:-1], -1, 0))
    add(41, "ATR breakout 1.5×ATR bar", sig)

    # 42. Narrow range breakout (NR4)
    sig = np.zeros(n)
    for i in range(4, n):
        ranges = [h[j] - l[j] for j in range(i-3, i+1)]
        if (h[i] - l[i]) == min(ranges):
            sig[i] = 1 if c[i] > c[i-1] else -1
    add(42, "NR4 narrow range breakout", sig)

    # 43. Inside bar breakout
    sig = np.zeros(n)
    for i in range(1, n-1):
        if h[i] < h[i-1] and l[i] > l[i-1]:
            sig[i+1] = 1 if c[i+1] > h[i] else (-1 if c[i+1] < l[i] else 0)
    add(43, "Inside bar breakout", sig)

    # 44. Pin bar (hammer)
    sig = np.zeros(n)
    for i in range(1, n):
        body = abs(c[i] - o[i])
        lower_wick = min(c[i], o[i]) - l[i]
        upper_wick = h[i] - max(c[i], o[i])
        if lower_wick > 2 * body and lower_wick > upper_wick and body > 0:
            sig[i] = 1
        elif upper_wick > 2 * body and upper_wick > lower_wick and body > 0:
            sig[i] = -1
    add(44, "Pin bar hammer/shooting star", sig)

    # 45. Engulfing candle
    sig = np.zeros(n)
    for i in range(1, n):
        if c[i] > o[i] and c[i-1] < o[i-1] and c[i] > o[i-1] and o[i] < c[i-1]:
            sig[i] = 1
        elif c[i] < o[i] and c[i-1] > o[i-1] and c[i] < o[i-1] and o[i] > c[i-1]:
            sig[i] = -1
    add(45, "Engulfing candle pattern", sig)

    # 46. Morning/Evening star (3-candle)
    sig = np.zeros(n)
    for i in range(2, n):
        body1 = abs(c[i-2] - o[i-2]); body2 = abs(c[i-1] - o[i-1]); body3 = abs(c[i] - o[i])
        if (c[i-2] < o[i-2] and body2 < 0.3 * body1 and c[i] > o[i] and c[i] > (o[i-2] + c[i-2]) / 2):
            sig[i] = 1
        elif (c[i-2] > o[i-2] and body2 < 0.3 * body1 and c[i] < o[i] and c[i] < (o[i-2] + c[i-2]) / 2):
            sig[i] = -1
    add(46, "Morning/Evening star 3-candle", sig)

    # 47. Double bottom detection (simplified)
    sig = np.zeros(n)
    for i in range(10, n):
        window = c[i-10:i]
        lows_idx = [j for j in range(1, 9) if window[j] < window[j-1] and window[j] < window[j+1]]
        if len(lows_idx) >= 2:
            l1, l2 = lows_idx[-2], lows_idx[-1]
            if abs(window[l1] - window[l2]) / window[l1] < 0.02 and c[i] > max(window[l1:l2+1]):
                sig[i] = 1
    add(47, "Double bottom 10-bar pattern", sig)

    # 48. 3-bar momentum continuation
    sig = np.zeros(n)
    for i in range(3, n):
        if c[i-3] < c[i-2] < c[i-1] and c[i] > c[i-1]:
            pullback = (c[i-1] - c[i]) / (c[i-1] - c[i-3] + 1e-12)
            if pullback < 0.5:
                sig[i] = 1
        elif c[i-3] > c[i-2] > c[i-1] and c[i] < c[i-1]:
            pullback = (c[i] - c[i-1]) / (c[i-3] - c[i-1] + 1e-12)
            if pullback < 0.5:
                sig[i] = -1
    add(48, "3-bar trend + pullback <50%", sig)

    # 49. Gap fill
    sig = np.zeros(n)
    for i in range(1, n):
        gap_up = o[i] > h[i-1] * 1.005
        gap_dn = o[i] < l[i-1] * 0.995
        if gap_up and c[i] < o[i]:
            sig[i] = -1  # fade gap up
        elif gap_dn and c[i] > o[i]:
            sig[i] = 1   # fade gap down
    add(49, "Gap fade strategy", sig)

    # 50. RSI5 extreme reversion
    sig = np.zeros(n)
    sig[1:] = np.where((rsi5[1:] > 15) & (rsi5[:-1] <= 15), 1,
              np.where((rsi5[1:] < 85) & (rsi5[:-1] >= 85), -1, 0))
    add(50, "RSI5 extreme <15/>85 reversion", sig)

    # 51. EMA ribbon aligned (5/10/20/50)
    e5 = ema(c, 5); e10 = ema(c, 10)
    sig = np.zeros(n)
    sig[1:] = np.where((e5[1:] > e10[1:]) & (e10[1:] > e21[1:]) & (e21[1:] > e50[1:]) &
                       (e5[:-1] <= e10[:-1]), 1,
              np.where((e5[1:] < e10[1:]) & (e10[1:] < e21[1:]) & (e21[1:] < e50[1:]) &
                       (e5[:-1] >= e10[:-1]), -1, 0))
    add(51, "EMA ribbon 5/10/21/50 aligned", sig)

    # 52. BB + OBV
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] < bbl[1:]) & (obv_[1:] > obv_e[1:]), 1,
              np.where((c[1:] > bbu[1:]) & (obv_[1:] < obv_e[1:]), -1, 0))
    add(52, "BB extreme + OBV confirmation", sig)

    # 53. Funding rate proxy (open interest momentum)
    # Use volume as OI proxy
    v_delta = v - np.roll(v, 5)
    sig = np.zeros(n)
    sig[5:] = np.where((v_delta[5:] > 0) & (c[5:] > c[:-5]) & (rsi14[5:] > 55), 1,
              np.where((v_delta[5:] < 0) & (c[5:] < c[:-5]) & (rsi14[5:] < 45), -1, 0))
    add(53, "Volume delta + price momentum", sig)

    # 54. ATR trailing stop entry
    sig = np.zeros(n)
    trail = np.full(n, np.nan)
    for i in range(14, n):
        if np.isnan(trail[i-1]):
            trail[i] = c[i] - 2*atr14[i]
        else:
            trail[i] = max(trail[i-1], c[i] - 2*atr14[i])
        sig[i] = 1 if c[i] > trail[i] and c[i-1] <= trail[i-1] else 0
    add(54, "ATR 2x trailing stop entry", sig)

    # 55. ROC cross zero
    sig = np.zeros(n)
    sig[1:] = np.where((roc10[1:] > 0) & (roc10[:-1] <= 0) & (adx14[1:] > 20), 1,
              np.where((roc10[1:] < 0) & (roc10[:-1] >= 0) & (adx14[1:] > 20), -1, 0))
    add(55, "ROC zero cross + ADX>20", sig)

    # 56. Stoch + MACD
    sig = np.zeros(n)
    sig[1:] = np.where((stk[1:] > std_[1:]) & (stk[:-1] <= std_[:-1]) & (mh[1:] > 0), 1,
              np.where((stk[1:] < std_[1:]) & (stk[:-1] >= std_[:-1]) & (mh[1:] < 0), -1, 0))
    add(56, "Stoch cross + MACD direction", sig)

    # 57. CCI + ADX
    sig = np.zeros(n)
    sig[1:] = np.where((cci20[1:] > 100) & (cci20[:-1] <= 100) & (adx14[1:] > 20), 1,
              np.where((cci20[1:] < -100) & (cci20[:-1] >= -100) & (adx14[1:] > 20), -1, 0))
    add(57, "CCI breakout + ADX>20", sig)

    # 58. MFI + RSI
    sig = np.zeros(n)
    sig[1:] = np.where((mfi14[1:] < 25) & (rsi14[1:] < 35), 1,
              np.where((mfi14[1:] > 75) & (rsi14[1:] > 65), -1, 0))
    add(58, "MFI+RSI double oversold/bought", sig)

    # 59. EMA cross + volume
    sig = np.zeros(n)
    sig[1:] = np.where((e9[1:] > e21[1:]) & (e9[:-1] <= e21[:-1]) & (v[1:] > v_ma20[1:]), 1,
              np.where((e9[1:] < e21[1:]) & (e9[:-1] >= e21[:-1]) & (v[1:] > v_ma20[1:]), -1, 0))
    add(59, "EMA 9/21 cross + volume confirm", sig)

    # 60. OBV + EMA trend + RSI
    sig = np.zeros(n)
    sig[1:] = np.where((obv_[1:] > obv_e[1:]) & (c[1:] > e50[1:]) & (rsi14[1:] > 50), 1,
              np.where((obv_[1:] < obv_e[1:]) & (c[1:] < e50[1:]) & (rsi14[1:] < 50), -1, 0))
    add(60, "OBV+EMA50+RSI triple confirm", sig)

    # 61. Keltner + RSI
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] < kl[1:]) & (rsi14[1:] < 35), 1,
              np.where((c[1:] > ku[1:]) & (rsi14[1:] > 65), -1, 0))
    add(61, "Keltner below + RSI<35", sig)

    # 62. Supertrend + MACD
    sig = np.zeros(n)
    sig[1:] = np.where((trend_[1:] == 1) & (trend_[:-1] == -1) & (mh[1:] > 0), 1,
              np.where((trend_[1:] == -1) & (trend_[:-1] == 1) & (mh[1:] < 0), -1, 0))
    add(62, "Supertrend + MACD confirm", sig)

    # 63. Aroon + ADX
    sig = np.zeros(n)
    sig[1:] = np.where((aroon_up[1:] > 70) & (aroon_dn[1:] < 30) & (adx14[1:] > 20), 1,
              np.where((aroon_up[1:] < 30) & (aroon_dn[1:] > 70) & (adx14[1:] > 20), -1, 0))
    add(63, "Aroon 70/30 + ADX>20", sig)

    # 64. Z-score + trend filter
    sig = np.zeros(n)
    sig[1:] = np.where((zs50[1:] > -1.5) & (zs50[:-1] <= -1.5) & (c[1:] > e200[1:]), 1,
              np.where((zs50[1:] < 1.5) & (zs50[:-1] >= 1.5) & (c[1:] < e200[1:]), -1, 0))
    add(64, "Z-score ±1.5 + EMA200 filter", sig)

    # 65. Donchian + ADX
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] > dch[:-1]) & (adx14[1:] > 20), 1,
              np.where((c[1:] < dcl[:-1]) & (adx14[1:] > 20), -1, 0))
    add(65, "Donchian breakout + ADX>20", sig)

    # 66. VWAP + RSI
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] > vwap_[1:]) & (rsi14[1:] > 55) & (c[:-1] <= vwap_[:-1]), 1,
              np.where((c[1:] < vwap_[1:]) & (rsi14[1:] < 45) & (c[:-1] >= vwap_[:-1]), -1, 0))
    add(66, "VWAP cross + RSI 55/45", sig)

    # 67. HMA + ADX
    sig = np.zeros(n)
    sig[1:] = np.where((hma9[1:] > hma21[1:]) & (hma9[:-1] <= hma21[:-1]) & (adx14[1:] > 20), 1,
              np.where((hma9[1:] < hma21[1:]) & (hma9[:-1] >= hma21[:-1]) & (adx14[1:] > 20), -1, 0))
    add(67, "HMA 9/21 + ADX>20", sig)

    # 68. RSI divergence (simplified)
    sig = np.zeros(n)
    for i in range(10, n):
        c_min_i = np.argmin(c[i-10:i]) + (i-10)
        r_at_min = rsi14[c_min_i]
        if c[i] < c[c_min_i] and rsi14[i] > r_at_min + 5:
            sig[i] = 1
        c_max_i = np.argmax(c[i-10:i]) + (i-10)
        r_at_max = rsi14[c_max_i]
        if c[i] > c[c_max_i] and rsi14[i] < r_at_max - 5:
            sig[i] = -1
    add(68, "RSI divergence 10-bar", sig)

    # 69. MACD + Stoch
    sig = np.zeros(n)
    sig[1:] = np.where((ml[1:] > ms[1:]) & (ml[:-1] <= ms[:-1]) & (stk[1:] < 50), 1,
              np.where((ml[1:] < ms[1:]) & (ml[:-1] >= ms[:-1]) & (stk[1:] > 50), -1, 0))
    add(69, "MACD cross + Stoch direction", sig)

    # 70. Multi-timeframe EMA (approx: fast vs slow EMA hierarchy)
    e100 = ema(c, 100)
    sig = np.zeros(n)
    sig[1:] = np.where((e9[1:] > e21[1:]) & (e21[1:] > e50[1:]) & (e50[1:] > e100[1:]) &
                       (e9[:-1] <= e21[:-1]), 1,
              np.where((e9[1:] < e21[1:]) & (e21[1:] < e50[1:]) & (e50[1:] < e100[1:]) &
                       (e9[:-1] >= e21[:-1]), -1, 0))
    add(70, "4-EMA full alignment 9/21/50/100", sig)

    # 71. Volume Price Trend (VPT)
    vpt = np.cumsum(v * np.diff(c, prepend=c[0]) / (c + 1e-12))
    vpt_e = ema(vpt, 20)
    sig = np.zeros(n)
    sig[1:] = np.where((vpt[1:] > vpt_e[1:]) & (vpt[:-1] <= vpt_e[:-1]) & (c[1:] > e50[1:]), 1,
              np.where((vpt[1:] < vpt_e[1:]) & (vpt[:-1] >= vpt_e[:-1]) & (c[1:] < e50[1:]), -1, 0))
    add(71, "VPT cross EMA20 + trend", sig)

    # 72. Elder Triple Screen (approx)
    sig = np.zeros(n)
    weekly_macd = mh  # use MACD hist as "weekly" direction
    sig[1:] = np.where((weekly_macd[1:] > 0) & (stk[1:] < 30) & (stk[1:] > stk[:-1]), 1,
              np.where((weekly_macd[1:] < 0) & (stk[1:] > 70) & (stk[1:] < stk[:-1]), -1, 0))
    add(72, "Elder triple screen (MACD+Stoch)", sig)

    # 73. Volatility regime + BB
    volatility = np.array([np.std(c[max(0,i-20):i+1]) for i in range(n)])
    vol_ma = sma(volatility, 50)
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] < bbl[1:]) & (volatility[1:] < vol_ma[1:]), 1,
              np.where((c[1:] > bbu[1:]) & (volatility[1:] < vol_ma[1:]), -1, 0))
    add(73, "Low-volatility BB reversion", sig)

    # 74. High-volatility BB breakout
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] > bbu[1:]) & (volatility[1:] > vol_ma[1:]) & (adx14[1:] > 25), 1,
              np.where((c[1:] < bbl[1:]) & (volatility[1:] > vol_ma[1:]) & (adx14[1:] > 25), -1, 0))
    add(74, "High-volatility BB breakout + ADX", sig)

    # 75. CMF (Chaikin Money Flow)
    mf_mult = ((c - l) - (h - c)) / (h - l + 1e-12)
    mf_vol = mf_mult * v
    cmf = np.array([np.sum(mf_vol[max(0,i-21):i+1]) / (np.sum(v[max(0,i-21):i+1]) + 1e-12) for i in range(n)])
    sig = np.zeros(n)
    sig[1:] = np.where((cmf[1:] > 0.1) & (cmf[:-1] <= 0.1) & (c[1:] > e50[1:]), 1,
              np.where((cmf[1:] < -0.1) & (cmf[:-1] >= -0.1) & (c[1:] < e50[1:]), -1, 0))
    add(75, "CMF ±0.1 cross + EMA50", sig)

    # 76. Parabolic SAR approx (ATR-based)
    sig = np.zeros(n)
    sar = l.copy()
    for i in range(1, n):
        sar[i] = max(sar[i-1], c[i] - 2*atr14[i]) if c[i] > sar[i-1] else min(c[i] + 2*atr14[i], c[i])
        if sar[i] > sar[i-1] and sar[i-1] <= sar[i-2] if i >= 2 else False:
            sig[i] = 1
        elif sar[i] < sar[i-1] and sar[i-1] >= sar[i-2] if i >= 2 else False:
            sig[i] = -1
    add(76, "Parabolic SAR (ATR2x approx)", sig)

    # 77. Open interest proxy trend (volume acceleration)
    v_acc = v - sma(v, 10)
    sig = np.zeros(n)
    sig[1:] = np.where((v_acc[1:] > 0) & (v_acc[:-1] <= 0) & (c[1:] > e21[1:]) & (rsi14[1:] > 50), 1,
              np.where((v_acc[1:] < 0) & (v_acc[:-1] >= 0) & (c[1:] < e21[1:]) & (rsi14[1:] < 50), -1, 0))
    add(77, "Volume acceleration + EMA21 + RSI", sig)

    # 78. Breakout + retest
    sig = np.zeros(n)
    for i in range(21, n):
        prev_high = np.max(h[i-20:i])
        if c[i-1] > prev_high and abs(c[i] - prev_high) / prev_high < 0.005:
            sig[i] = 1
        prev_low = np.min(l[i-20:i])
        if c[i-1] < prev_low and abs(c[i] - prev_low) / prev_low < 0.005:
            sig[i] = -1
    add(78, "20-bar breakout retest entry", sig)

    # 79. Fibonacci 50% retracement entry
    sig = np.zeros(n)
    for i in range(20, n):
        sl = c[i-20:i]
        hi = np.max(sl); lo = np.min(sl)
        fib50 = lo + 0.5 * (hi - lo)
        fib618 = lo + 0.382 * (hi - lo)
        if c[i-1] > c[i] and abs(c[i] - fib50) / fib50 < 0.005 and c[i] > lo:
            sig[i] = 1
    add(79, "Fibonacci 50% retracement long", sig)

    # 80. Liquidity hunt (false breakout)
    sig = np.zeros(n)
    for i in range(5, n):
        prev_high = np.max(h[i-5:i])
        prev_low  = np.min(l[i-5:i])
        if h[i] > prev_high and c[i] < prev_high:
            sig[i] = -1  # fake breakout up
        elif l[i] < prev_low and c[i] > prev_low:
            sig[i] = 1   # fake breakout down
    add(80, "Fake breakout / liquidity hunt", sig)

    # 81. Kelly-based: only enter when expected value > threshold
    hist_ret = np.diff(c, prepend=c[0]) / (c + 1e-12)
    roll_mean = np.array([np.mean(hist_ret[max(0,i-20):i]) for i in range(n)])
    roll_std  = np.array([np.std(hist_ret[max(0,i-20):i]) for i in range(n)])
    kelly_f   = roll_mean / (roll_std**2 + 1e-12)
    sig = np.zeros(n)
    sig[1:] = np.where(kelly_f[1:] > 2, 1, np.where(kelly_f[1:] < -2, -1, 0))
    add(81, "Kelly fraction positive EV>2σ", sig)

    # 82. Z-score + EMA cross combo
    sig = np.zeros(n)
    sig[1:] = np.where((zs50[1:] < -1) & (e9[1:] > e21[1:]) & (e9[:-1] <= e21[:-1]), 1,
              np.where((zs50[1:] > 1) & (e9[1:] < e21[1:]) & (e9[:-1] >= e21[:-1]), -1, 0))
    add(82, "Z-score pullback + EMA cross", sig)

    # 83. BB width expansion breakout
    sig = np.zeros(n)
    for i in range(21, n):
        bw_now = bbu[i] - bbl[i]
        bw_prev = bbu[i-1] - bbl[i-1]
        if bw_now > bw_prev * 1.1 and c[i] > bbu[i]:
            sig[i] = 1
        elif bw_now > bw_prev * 1.1 and c[i] < bbl[i]:
            sig[i] = -1
    add(83, "BB expanding + price breakout", sig)

    # 84. EMA9 cross + CCI confirm
    sig = np.zeros(n)
    sig[1:] = np.where((e9[1:] > e21[1:]) & (e9[:-1] <= e21[:-1]) & (cci20[1:] > 0), 1,
              np.where((e9[1:] < e21[1:]) & (e9[:-1] >= e21[:-1]) & (cci20[1:] < 0), -1, 0))
    add(84, "EMA 9/21 cross + CCI direction", sig)

    # 85. Trend + pullback + volume
    sig = np.zeros(n)
    for i in range(5, n):
        if c[i] > e50[i] and c[i-1] < c[i-2] and c[i-2] < c[i-3] and v[i] > v_ma20[i] and c[i] > c[i-1]:
            sig[i] = 1
        elif c[i] < e50[i] and c[i-1] > c[i-2] and c[i-2] > c[i-3] and v[i] > v_ma20[i] and c[i] < c[i-1]:
            sig[i] = -1
    add(85, "Trend pullback + volume re-entry", sig)

    # 86. MFI + Stoch
    sig = np.zeros(n)
    sig[1:] = np.where((mfi14[1:] < 30) & (stk[1:] < 30) & (rsi14[1:] < 40), 1,
              np.where((mfi14[1:] > 70) & (stk[1:] > 70) & (rsi14[1:] > 60), -1, 0))
    add(86, "MFI+Stoch+RSI triple OS/OB", sig)

    # 87. Momentum score: RSI + ROC + MACD
    momentum_score = (rsi14 - 50) / 50 + roc10 / 10 + mh / (atr14 + 1e-12)
    ms_e = ema(momentum_score, 10)
    sig = np.zeros(n)
    sig[1:] = np.where((momentum_score[1:] > ms_e[1:]) & (momentum_score[:-1] <= ms_e[:-1]) & (momentum_score[1:] > 0.5), 1,
              np.where((momentum_score[1:] < ms_e[1:]) & (momentum_score[:-1] >= ms_e[:-1]) & (momentum_score[1:] < -0.5), -1, 0))
    add(87, "Composite momentum score RSI+ROC+MACD", sig)

    # 88. Regime: trend when ADX>25, revert when ADX<20
    sig = np.zeros(n)
    sig[1:] = np.where(
        (adx14[1:] > 25) & (e9[1:] > e21[1:]) & (e9[:-1] <= e21[:-1]), 1,
        np.where(
            (adx14[1:] > 25) & (e9[1:] < e21[1:]) & (e9[:-1] >= e21[:-1]), -1,
            np.where(
                (adx14[1:] < 20) & (rsi14[1:] < 30), 1,
                np.where((adx14[1:] < 20) & (rsi14[1:] > 70), -1, 0)
            )
        )
    )
    add(88, "Regime-adaptive: trend+revert", sig)

    # 89. Volume-weighted momentum
    vwm = np.cumsum(v * (c - np.roll(c, 1))) / (np.cumsum(v) + 1e-12)
    vwm_e = ema(vwm, 20)
    sig = np.zeros(n)
    sig[1:] = np.where((vwm[1:] > vwm_e[1:]) & (vwm[:-1] <= vwm_e[:-1]) & (c[1:] > e50[1:]), 1,
              np.where((vwm[1:] < vwm_e[1:]) & (vwm[:-1] >= vwm_e[:-1]) & (c[1:] < e50[1:]), -1, 0))
    add(89, "Volume-weighted momentum cross", sig)

    # 90. Price action + volume + trend triple filter
    sig = np.zeros(n)
    sig[1:] = np.where(
        (c[1:] > c[:-1]) & (v[1:] > v_ma20[1:]) & (c[1:] > e50[1:]) & (adx14[1:] > 20) & (rsi14[1:] > 50), 1,
        np.where(
            (c[1:] < c[:-1]) & (v[1:] > v_ma20[1:]) & (c[1:] < e50[1:]) & (adx14[1:] > 20) & (rsi14[1:] < 50), -1, 0
        )
    )
    add(90, "5-factor: price+vol+EMA50+ADX+RSI", sig)

    # 91. Stoch + ADX + EMA
    sig = np.zeros(n)
    sig[1:] = np.where((stk[1:] > std_[1:]) & (stk[:-1] <= std_[:-1]) & (adx14[1:] > 20) & (c[1:] > e50[1:]), 1,
              np.where((stk[1:] < std_[1:]) & (stk[:-1] >= std_[:-1]) & (adx14[1:] > 20) & (c[1:] < e50[1:]), -1, 0))
    add(91, "Stoch cross + ADX>20 + EMA50", sig)

    # 92. Williams %R + trend
    sig = np.zeros(n)
    sig[1:] = np.where((wr14[1:] > -80) & (wr14[:-1] <= -80) & (c[1:] > e50[1:]), 1,
              np.where((wr14[1:] < -20) & (wr14[:-1] >= -20) & (c[1:] < e50[1:]), -1, 0))
    add(92, "Williams %R bounce + EMA50 trend", sig)

    # 93. CMF + MACD
    sig = np.zeros(n)
    sig[1:] = np.where((cmf[1:] > 0.05) & (mh[1:] > 0) & (cmf[:-1] <= 0.05), 1,
              np.where((cmf[1:] < -0.05) & (mh[1:] < 0) & (cmf[:-1] >= -0.05), -1, 0))
    add(93, "CMF cross + MACD direction", sig)

    # 94. TRIX + RSI + ADX
    sig = np.zeros(n)
    sig[1:] = np.where((trix15[1:] > 0) & (trix15[:-1] <= 0) & (rsi14[1:] > 50) & (adx14[1:] > 20), 1,
              np.where((trix15[1:] < 0) & (trix15[:-1] >= 0) & (rsi14[1:] < 50) & (adx14[1:] > 20), -1, 0))
    add(94, "TRIX zero cross + RSI + ADX", sig)

    # 95. Aroon + RSI + MACD
    sig = np.zeros(n)
    sig[1:] = np.where((aroon_up[1:] > aroon_dn[1:]) & (rsi14[1:] > 50) & (mh[1:] > 0) &
                       (aroon_up[:-1] <= aroon_dn[:-1]), 1,
              np.where((aroon_up[1:] < aroon_dn[1:]) & (rsi14[1:] < 50) & (mh[1:] < 0) &
                       (aroon_up[:-1] >= aroon_dn[:-1]), -1, 0))
    add(95, "Aroon cross + RSI + MACD triple", sig)

    # 96. BB + Stoch + trend
    sig = np.zeros(n)
    sig[1:] = np.where((c[1:] < bbl3[1:]) & (stk[1:] < 30) & (c[1:] > e200[1:]), 1,
              np.where((c[1:] > bbu3[1:]) & (stk[1:] > 70) & (c[1:] < e200[1:]), -1, 0))
    add(96, "BB 1.5σ + Stoch extreme + EMA200", sig)

    # 97. Volume divergence + RSI
    sig = np.zeros(n)
    for i in range(10, n):
        if c[i] < c[i-5] and v[i] < v[i-5] and rsi14[i] > rsi14[i-5]:
            sig[i] = 1  # price down on low vol, RSI improving
        elif c[i] > c[i-5] and v[i] < v[i-5] and rsi14[i] < rsi14[i-5]:
            sig[i] = -1
    add(97, "Volume divergence + RSI improvement", sig)

    # 98. ADX peak reversal
    sig = np.zeros(n)
    for i in range(5, n):
        adx_peak = np.max(adx14[i-5:i])
        if adx14[i] < adx_peak * 0.85 and adx_peak > 40:
            sig[i] = 1 if c[i] > e50[i] else -1
    add(98, "ADX peak reversal (ADX>40 then fade)", sig)

    # 99. RSI5 + RSI14 combo cross
    sig = np.zeros(n)
    sig[1:] = np.where((rsi5[1:] > rsi14[1:]) & (rsi5[:-1] <= rsi14[:-1]) & (rsi14[1:] > 40), 1,
              np.where((rsi5[1:] < rsi14[1:]) & (rsi5[:-1] >= rsi14[:-1]) & (rsi14[1:] < 60), -1, 0))
    add(99, "RSI5 cross RSI14", sig)

    # 100. Full stack: EMA+MACD+RSI+ADX+Vol 5-factor
    sig = np.zeros(n)
    score = np.zeros(n)
    score += np.where(c > e50, 1, -1)
    score += np.where(mh > 0, 1, -1)
    score += np.where(rsi14 > 50, 1, -1)
    score += np.where(adx14 > 20, 0.5, -0.5)
    score += np.where(v > v_ma20, 0.5, -0.5)
    sig[1:] = np.where((score[1:] >= 3.5) & (score[:-1] < 3.5), 1,
              np.where((score[1:] <= -3.5) & (score[:-1] > -3.5), -1, 0))
    add(100, "Master score: EMA+MACD+RSI+ADX+Vol", sig)

    return strats


# ─────────────────────────── MAIN ────────────────────────────────

def main():
    print("=" * 60)
    print("OKX Top-30 × 100 Strategies Backtest")
    print("=" * 60)

    print("\n[1/3] Fetching Top 30 USDT-SWAP by volume...")
    try:
        symbols = fetch_top30()
        print(f"  Got {len(symbols)} symbols")
        for i, s in enumerate(symbols[:10]):
            print(f"    {i+1}. {s}")
        if len(symbols) > 10:
            print(f"    ... and {len(symbols)-10} more")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print("\n[2/3] Downloading 1H OHLCV and running 100 strategies...")

    # strategy_id -> [sharpe list across all symbols]
    strat_sharpes = {i: [] for i in range(1, 101)}
    strat_names = {}
    strat_returns = {i: [] for i in range(1, 101)}
    strat_winrates = {i: [] for i in range(1, 101)}
    strat_trades  = {i: [] for i in range(1, 101)}
    strat_maxdd   = {i: [] for i in range(1, 101)}

    ok_symbols = 0
    for idx, sym in enumerate(symbols):
        print(f"  [{idx+1}/30] {sym}...", end=" ", flush=True)
        data = fetch_ohlcv(sym, bar="1H", limit=400)
        if data is None or len(data[0]) < 100:
            print("skip (insufficient data)")
            continue
        o, h, l, c, v = data
        print(f"{len(c)} bars", end=" | ", flush=True)
        ok_symbols += 1

        try:
            strats = get_all_strategies(o, h, l, c, v)
        except Exception as e:
            print(f"indicators error: {e}")
            continue

        for sid, name, signals in strats:
            strat_names[sid] = name
            try:
                ret, sharpe, wr, trades, mdd = backtest(signals, c)
                strat_sharpes[sid].append(sharpe)
                strat_returns[sid].append(ret)
                strat_winrates[sid].append(wr)
                strat_trades[sid].append(trades)
                strat_maxdd[sid].append(mdd)
            except Exception:
                pass
        print("done")

    print(f"\n[3/3] Aggregating results across {ok_symbols} symbols...\n")

    # Score = avg sharpe (minimum 3 symbols to qualify)
    results = []
    for sid in range(1, 101):
        sharpes = strat_sharpes[sid]
        rets    = strat_returns[sid]
        wrs     = strat_winrates[sid]
        trades  = strat_trades[sid]
        mdds    = strat_maxdd[sid]
        if len(sharpes) < 3:
            continue
        results.append({
            "id": sid,
            "name": strat_names.get(sid, "?"),
            "avg_sharpe": np.mean(sharpes),
            "avg_return": np.mean(rets) * 100,
            "avg_winrate": np.mean(wrs) * 100,
            "avg_trades": np.mean(trades),
            "avg_maxdd": np.mean(mdds) * 100,
            "symbols_tested": len(sharpes),
            "positive_sharpe_pct": 100 * np.mean(np.array(sharpes) > 0),
        })

    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    print("=" * 70)
    print("TOP 10 STRATEGIES (ranked by avg Sharpe across Top-30 OKX symbols)")
    print("=" * 70)

    for rank, r in enumerate(results[:10], 1):
        print(f"\n#{rank}  [{r['id']:3d}] {r['name']}")
        print(f"       Sharpe: {r['avg_sharpe']:+.3f}  |  Return: {r['avg_return']:+.1f}%  |  Win%: {r['avg_winrate']:.1f}%")
        print(f"       MaxDD:  {r['avg_maxdd']:.1f}%   |  Trades: {r['avg_trades']:.0f}    |  Syms>0: {r['positive_sharpe_pct']:.0f}%")

    print("\n" + "=" * 70)
    print(f"Tested {len(results)} strategies × {ok_symbols} symbols")
    print(f"Completed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
