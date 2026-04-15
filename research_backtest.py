"""
MA Ribbon Strategy V0-V3 Ablation Backtest
Uses existing cached data from data/*.csv.gz
Run: python research_backtest.py
"""

import gzip, csv, sys, numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_ohlcv(symbol, tf):
    """Load OHLCV from data/{symbol}_{tf}.csv.gz"""
    fn = f"data/{symbol}_{tf}.csv.gz"
    with gzip.open(fn, "rt") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    n = len(rows)
    o = np.array([float(r["open"]) for r in rows])
    h = np.array([float(r["high"]) for r in rows])
    l = np.array([float(r["low"]) for r in rows])
    c = np.array([float(r["close"]) for r in rows])
    v = np.array([float(r["volume"]) for r in rows])
    ts = [r["open_time"] for r in rows]
    return ts, o, h, l, c, v

# ═══════════════════════════════════════════════════════════════
# INDICATORS (pure functions)
# ═══════════════════════════════════════════════════════════════

def sma(x, n):
    out = np.full(len(x), np.nan)
    # Use rolling window approach that handles NaN from prior computations
    for i in range(n - 1, len(x)):
        window = x[i - n + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = np.mean(window)
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
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    prev_c = np.roll(c, 1)
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
    upper = mid + std * s
    lower = mid - std * s
    return lower, mid, upper

def ma_slope(ma, lookback=5):
    """Slope = (ma[i] - ma[i-lookback]) / ma[i-lookback] * 100 (percentage)"""
    slope = np.full(len(ma), np.nan)
    for i in range(lookback, len(ma)):
        if not np.isnan(ma[i]) and not np.isnan(ma[i-lookback]) and ma[i-lookback] != 0:
            slope[i] = (ma[i] - ma[i-lookback]) / ma[i-lookback] * 100
    return slope

def fanning_distance(ma5, ma8, ema21, ma55, c):
    """Average percentage distance between adjacent MAs, normalized by close price"""
    dist = np.full(len(c), np.nan)
    for i in range(len(c)):
        if any(np.isnan([ma5[i], ma8[i], ema21[i], ma55[i]])):
            continue
        d1 = abs(ma5[i] - ma8[i])
        d2 = abs(ma8[i] - ema21[i])
        d3 = abs(ema21[i] - ma55[i])
        dist[i] = (d1 + d2 + d3) / (3 * c[i]) * 100
    return dist

# ═══════════════════════════════════════════════════════════════
# V0 CORE: Ribbon alignment + crossover + ADX + Volume
# ═══════════════════════════════════════════════════════════════

def compute_ribbon_state(c, ma5, ma8, ema21, ma55):
    bull = (c > ma5) & (ma5 > ma8) & (ma8 > ema21) & (ema21 > ma55)
    bear = (c < ma5) & (ma5 < ma8) & (ma8 < ema21) & (ema21 < ma55)
    return bull, bear

def generate_signals_v0(o, h, l, c, v, params):
    """V0: Core MA ribbon strategy - alignment crossover + ADX + Volume"""
    p = params
    ma5 = sma(c, p["ma5_n"])
    ma8 = sma(c, p["ma8_n"])
    e21 = ema(c, p["ema21_n"])
    ma55_ = sma(c, p["ma55_n"])
    atr14 = atr(h, l, c, 14)
    adx14 = adx_calc(h, l, c, 14)
    vol_avg = sma(v, 20)

    bull, bear = compute_ribbon_state(c, ma5, ma8, e21, ma55_)
    adx_ok = adx14 > p["adx_min"]
    vol_ok = v > p["vol_mult"] * vol_avg

    sig = np.zeros(len(c))
    for i in range(1, len(c)):
        if bull[i] and not bull[i-1] and adx_ok[i] and vol_ok[i]:
            sig[i] = 1
        elif bear[i] and not bear[i-1] and adx_ok[i] and vol_ok[i]:
            sig[i] = -1

    return sig, atr14, ma5, ma8, e21, ma55_

def generate_signals_v1(o, h, l, c, v, params):
    """V1: V0 + slope filter (all MAs must slope in trade direction)"""
    sig_v0, atr14, ma5, ma8, e21, ma55_ = generate_signals_v0(o, h, l, c, v, params)

    slope_ma5 = ma_slope(ma5, lookback=5)
    slope_ma8 = ma_slope(ma8, lookback=5)
    slope_e21 = ma_slope(e21, lookback=5)
    slope_threshold = params.get("slope_threshold", 0.0)

    sig = np.zeros(len(c))
    for i in range(len(c)):
        if sig_v0[i] == 1:
            # All slopes must be positive for long
            if (not np.isnan(slope_ma5[i]) and slope_ma5[i] > slope_threshold and
                not np.isnan(slope_ma8[i]) and slope_ma8[i] > slope_threshold and
                not np.isnan(slope_e21[i]) and slope_e21[i] > slope_threshold):
                sig[i] = 1
        elif sig_v0[i] == -1:
            # All slopes must be negative for short
            if (not np.isnan(slope_ma5[i]) and slope_ma5[i] < -slope_threshold and
                not np.isnan(slope_ma8[i]) and slope_ma8[i] < -slope_threshold and
                not np.isnan(slope_e21[i]) and slope_e21[i] < -slope_threshold):
                sig[i] = -1

    return sig, atr14

def generate_signals_v2(o, h, l, c, v, params):
    """V2: V0 + Bollinger Band overextension filter (reject entries beyond BB bands)"""
    sig_v0, atr14, ma5, ma8, e21, ma55_ = generate_signals_v0(o, h, l, c, v, params)

    bb_lower, bb_mid, bb_upper = bollinger_bands(c, n=20, std=2.0)
    bb_max_ext = params.get("bb_max_extension", 1.0)  # max bb_pos for long (0=lower, 1=upper)

    sig = np.zeros(len(c))
    for i in range(len(c)):
        if sig_v0[i] == 0:
            continue
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range <= 0:
            continue
        position_in_bb = (c[i] - bb_lower[i]) / bb_range

        if sig_v0[i] == 1:
            # Long: reject if price is too far above BB upper (overextended)
            if position_in_bb <= bb_max_ext:
                sig[i] = 1
        elif sig_v0[i] == -1:
            # Short: reject if price is too far below BB lower (overextended)
            if position_in_bb >= (1 - bb_max_ext):
                sig[i] = -1

    return sig, atr14

def generate_signals_v3(o, h, l, c, v, params):
    """V3: V0 + fanning distance filter (MAs must be sufficiently spread)"""
    sig_v0, atr14, ma5, ma8, e21, ma55_ = generate_signals_v0(o, h, l, c, v, params)

    fan_dist = fanning_distance(ma5, ma8, e21, ma55_, c)
    fan_threshold = params.get("fanning_min_pct", 0.5)  # minimum 0.5% average distance

    sig = np.zeros(len(c))
    for i in range(len(c)):
        if sig_v0[i] != 0 and not np.isnan(fan_dist[i]) and fan_dist[i] >= fan_threshold:
            sig[i] = sig_v0[i]

    return sig, atr14

# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def backtest(signals, h, l, c, atr14, atr_mult=2.0, rr=2.0, fee=0.0005):
    """Bar-close backtest with ATR trailing SL and RR-based TP"""
    pos = 0; entry = 0.0; sl = 0.0; tp = 0.0
    equity = 1.0; peak = 1.0; max_dd = 0.0
    returns = []; wins = 0; trades = 0
    n = len(c)
    _atr = atr14

    for i in range(1, n):
        if pos != 0:
            # Check SL/TP using high/low for intrabar
            if pos == 1:
                hit_sl = l[i] <= sl
                hit_tp = h[i] >= tp
            else:
                hit_sl = h[i] >= sl
                hit_tp = l[i] <= tp

            if hit_tp and hit_sl:
                # Both hit in same bar - assume SL hit first if close is losing
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

            # Trail SL
            if pos == 1 and not np.isnan(_atr[i]):
                new_sl = c[i] - atr_mult * _atr[i]
                sl = max(sl, new_sl)
            elif pos == -1 and not np.isnan(_atr[i]):
                new_sl = c[i] + atr_mult * _atr[i]
                sl = min(sl, new_sl)

            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)

        # New entry
        if pos == 0 and signals[i] != 0 and not np.isnan(_atr[i]):
            pos = int(signals[i])
            entry = c[i]
            sl_dist = atr_mult * _atr[i]
            sl = entry - sl_dist if pos == 1 else entry + sl_dist
            tp = entry + rr * sl_dist if pos == 1 else entry - rr * sl_dist

    if trades < 2:
        return {"net_pct": 0, "sharpe": 0, "winrate": 0, "trades": trades, "max_dd": 0}

    r = np.array(returns)
    # TF-aware annualization handled by caller
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(len(r))
    return {
        "net_pct": (equity - 1) * 100,
        "sharpe": sharpe,
        "winrate": wins / trades * 100,
        "trades": trades,
        "max_dd": max_dd * 100,
    }

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]
TIMEFRAMES = ["15m", "1h", "4h"]

DEFAULT_PARAMS = {
    "ma5_n": 5, "ma8_n": 8, "ema21_n": 21, "ma55_n": 55,
    "adx_min": 20, "vol_mult": 1.2,
    "atr_mult": 2.0, "rr": 2.0,
    # V1 params
    "slope_threshold": 0.0,
    # V2 params
    "bb_entry_zone": 0.3,
    # V3 params
    "fanning_min_pct": 0.5,
}

VARIANTS = {
    "V0_core": generate_signals_v0,
    "V1_slope": generate_signals_v1,
    "V2_bb": generate_signals_v2,
    "V3_fanning": generate_signals_v3,
}

def main():
    print("=" * 90)
    print("MA RIBBON STRATEGY - V0/V1/V2/V3 ABLATION BACKTEST")
    print(f"Symbols: {', '.join(s.upper() for s in SYMBOLS)}")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Params: MA {DEFAULT_PARAMS['ma5_n']}/{DEFAULT_PARAMS['ma8_n']}/EMA{DEFAULT_PARAMS['ema21_n']}/{DEFAULT_PARAMS['ma55_n']}")
    print(f"        ATR x{DEFAULT_PARAMS['atr_mult']}  RR {DEFAULT_PARAMS['rr']}:1  ADX>{DEFAULT_PARAMS['adx_min']}  VolMult>{DEFAULT_PARAMS['vol_mult']}")
    print(f"        V1 slope_threshold={DEFAULT_PARAMS['slope_threshold']}  V2 bb_zone={DEFAULT_PARAMS['bb_entry_zone']}  V3 fan_min={DEFAULT_PARAMS['fanning_min_pct']}%")
    print("=" * 90)

    all_results = []

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\n--- {sym.upper()} {tf} ---")
            try:
                ts, o, h, l, c, v = load_ohlcv(sym, tf)
            except Exception as e:
                print(f"  SKIP: {e}")
                continue

            print(f"  {len(c)} bars: {ts[0]} to {ts[-1]}")

            for variant_name, signal_fn in VARIANTS.items():
                result = signal_fn(o, h, l, c, v, DEFAULT_PARAMS)
                if variant_name == "V0_core":
                    sig, atr14, _, _, _, _ = result
                else:
                    sig, atr14 = result

                bt = backtest(sig, h, l, c, atr14,
                              atr_mult=DEFAULT_PARAMS["atr_mult"],
                              rr=DEFAULT_PARAMS["rr"])

                all_results.append({
                    "symbol": sym.upper(),
                    "tf": tf,
                    "variant": variant_name,
                    **bt
                })

                print(f"  {variant_name:12s}  trades={bt['trades']:3d}  "
                      f"win={bt['winrate']:5.1f}%  sharpe={bt['sharpe']:+6.2f}  "
                      f"net={bt['net_pct']:+7.2f}%  maxDD={bt['max_dd']:5.1f}%")

    # ── Summary Table ──
    print("\n" + "=" * 90)
    print("FULL RESULTS TABLE")
    print("=" * 90)
    print(f"{'symbol':<10} {'TF':<5} {'variant':<14} {'trades':>6} {'winrate':>8} {'sharpe':>8} {'net%':>8} {'maxDD%':>8}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['symbol']:<10} {r['tf']:<5} {r['variant']:<14} {r['trades']:6d} "
              f"{r['winrate']:7.1f}% {r['sharpe']:+7.2f} {r['net_pct']:+7.2f}% {r['max_dd']:7.1f}%")

    # ── Ablation: V1/V2/V3 vs V0 ──
    print("\n" + "=" * 90)
    print("ABLATION: Filter impact vs V0 baseline")
    print("=" * 90)
    print(f"{'symbol':<10} {'TF':<5} {'variant':<14} {'trades_delta':>13} {'sharpe_delta':>13} {'net_delta':>11}")
    print("-" * 72)

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            v0 = [r for r in all_results if r["symbol"] == sym.upper() and r["tf"] == tf and r["variant"] == "V0_core"]
            if not v0:
                continue
            v0 = v0[0]
            for vname in ["V1_slope", "V2_bb", "V3_fanning"]:
                vx = [r for r in all_results if r["symbol"] == sym.upper() and r["tf"] == tf and r["variant"] == vname]
                if not vx:
                    continue
                vx = vx[0]
                dt = vx["trades"] - v0["trades"]
                ds = vx["sharpe"] - v0["sharpe"]
                dn = vx["net_pct"] - v0["net_pct"]
                print(f"{sym.upper():<10} {tf:<5} {vname:<14} {dt:+13d} {ds:+13.2f} {dn:+10.2f}%")

    # ── Per-variant aggregate ──
    print("\n" + "=" * 90)
    print("AGGREGATE BY VARIANT (across all symbols x TFs)")
    print("=" * 90)

    for vname in VARIANTS:
        vr = [r for r in all_results if r["variant"] == vname and r["trades"] >= 2]
        if not vr:
            print(f"  {vname}: no results with >= 2 trades")
            continue
        avg_trades = np.mean([r["trades"] for r in vr])
        avg_wr = np.mean([r["winrate"] for r in vr])
        avg_sharpe = np.mean([r["sharpe"] for r in vr])
        avg_net = np.mean([r["net_pct"] for r in vr])
        avg_dd = np.mean([r["max_dd"] for r in vr])
        n_positive = sum(1 for r in vr if r["net_pct"] > 0)
        print(f"  {vname:12s}  avg_trades={avg_trades:5.1f}  win={avg_wr:5.1f}%  "
              f"sharpe={avg_sharpe:+6.2f}  net={avg_net:+7.2f}%  maxDD={avg_dd:5.1f}%  "
              f"profitable={n_positive}/{len(vr)}")

    print(f"\nCompleted: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 90)


if __name__ == "__main__":
    main()
