"""
Microstructure Strategies
=========================
Three sub-strategies based on volume/price microstructure:

1. Wyckoff Accumulation: vol squeeze → breakout (smart money loading)
2. OBV Divergence: volume direction disagrees with price → reversal
3. Volume Dry-Up Breakout: ultra-low volume → sudden spike = controlled move

These work best on high-control / low-float coins where volume patterns
are driven by a few large participants (market makers, whales).

Future extensions (need additional data sources):
- Orderbook depth imbalance (Bitget websocket L2)
- Funding rate extremes (perp-specific)
- Open interest spikes
- On-chain whale wallet movement (blockchain API)
- News sentiment (NLP on Twitter/Telegram)

Run: python -m research.microstructure_backtest
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════

def sma(x, n):
    out = np.full(len(x), np.nan)
    if not np.any(np.isnan(x)):
        cs = np.cumsum(x); out[n-1:] = (cs[n-1:] - np.concatenate([[0], cs[:-n]])) / n
    return out


def ema(x, n):
    a = 2.0 / (n + 1); out = np.full(len(x), np.nan); out[n-1] = np.mean(x[:n])
    for i in range(n, len(x)): out[i] = a * x[i] + (1 - a) * out[i-1]
    return out


def atr(h, l, c, n=14):
    pc = np.roll(c, 1); tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]; return sma(tr, n)


def obv(c, v):
    """On-Balance Volume: cumulative volume in price direction."""
    out = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i-1]:
            out[i] = out[i-1] + v[i]
        elif c[i] < c[i-1]:
            out[i] = out[i-1] - v[i]
        else:
            out[i] = out[i-1]
    return out


def volatility(c, n=20):
    """Rolling std dev of returns as % of price."""
    ret = np.zeros(len(c))
    ret[1:] = (c[1:] - c[:-1]) / c[:-1]
    out = np.full(len(c), np.nan)
    for i in range(n, len(c)):
        out[i] = np.std(ret[i-n+1:i+1]) * 100
    return out


def volume_ratio(v, short_n=5, long_n=50):
    """Ratio of short-term avg volume to long-term avg volume."""
    short = sma(v, short_n)
    long_ = sma(v, long_n)
    ratio = np.full(len(v), np.nan)
    valid = (~np.isnan(short)) & (~np.isnan(long_)) & (long_ > 0)
    ratio[valid] = short[valid] / long_[valid]
    return ratio


# ═══════════════════════════════════════════════════════════════
# STRATEGY 1: WYCKOFF ACCUMULATION / DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
# Pattern: volume dries up + volatility compresses → sudden breakout
# = smart money finished accumulating, now marking up
#
# Detection:
#   1. Volume ratio (5-bar / 50-bar) drops below 0.3 (dry phase)
#   2. Volatility drops below 20th percentile of its own history
#   3. Both conditions hold for at least N bars (consolidation)
#   4. Then volume spikes above 3x AND price breaks out of range
#   = Entry

def wyckoff_signals(o, h, l, c, v, cfg=None):
    """Detect Wyckoff accumulation/distribution breakouts."""
    if cfg is None:
        cfg = {}
    n = len(c)

    vol_ratio = volume_ratio(v, cfg.get("vol_short", 5), cfg.get("vol_long", 50))
    vol_20 = volatility(c, cfg.get("vol_period", 20))
    atr_val = atr(h, l, c, 14)

    dry_threshold = cfg.get("dry_vol_ratio", 0.4)  # volume must be below 40% of normal
    min_dry_bars = cfg.get("min_dry_bars", 10)      # must stay dry for 10+ bars
    spike_mult = cfg.get("spike_mult", 3.0)         # breakout volume must be 3x
    breakout_atr_mult = cfg.get("breakout_atr", 1.5)  # price must move 1.5x ATR

    signals = np.zeros(n)
    entry_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)
    tp_prices = np.full(n, np.nan)

    # Track dry spell
    dry_count = 0
    range_high = 0.0
    range_low = float('inf')

    for i in range(60, n):
        if np.isnan(vol_ratio[i]) or np.isnan(atr_val[i]):
            continue

        # Volatility percentile (rolling 200-bar)
        if i >= 200:
            vol_hist = vol_20[max(0, i-200):i]
            vol_hist = vol_hist[~np.isnan(vol_hist)]
            if len(vol_hist) > 0:
                vol_pct = np.sum(vol_hist < vol_20[i]) / len(vol_hist)
            else:
                vol_pct = 0.5
        else:
            vol_pct = 0.5

        is_dry = vol_ratio[i] < dry_threshold and vol_pct < 0.3

        if is_dry:
            dry_count += 1
            range_high = max(range_high, h[i])
            range_low = min(range_low, l[i])
        else:
            # Check for breakout after dry spell
            if dry_count >= min_dry_bars and vol_ratio[i] > spike_mult:
                price_move = abs(c[i] - c[i-1])
                if not np.isnan(atr_val[i]) and price_move > breakout_atr_mult * atr_val[i]:
                    # Breakout direction
                    if c[i] > range_high:
                        # Upward breakout → long (accumulation complete)
                        signals[i] = 1
                        entry_prices[i] = c[i]
                        sl_prices[i] = range_low  # SL at bottom of range
                        risk = c[i] - range_low
                        tp_prices[i] = c[i] + 2 * risk  # RR 2:1
                    elif c[i] < range_low:
                        # Downward breakout → short (distribution complete)
                        signals[i] = -1
                        entry_prices[i] = c[i]
                        sl_prices[i] = range_high
                        risk = range_high - c[i]
                        tp_prices[i] = c[i] - 2 * risk

            dry_count = 0
            range_high = 0.0
            range_low = float('inf')

    return signals, entry_prices, sl_prices, tp_prices


# ═══════════════════════════════════════════════════════════════
# STRATEGY 2: OBV DIVERGENCE
# ═══════════════════════════════════════════════════════════════
# When price makes new low but OBV doesn't → bullish divergence
# When price makes new high but OBV doesn't → bearish divergence
# = Smart money is accumulating/distributing against the visible trend

def obv_divergence_signals(o, h, l, c, v, cfg=None):
    """Detect OBV divergence signals."""
    if cfg is None:
        cfg = {}
    n = len(c)
    lookback = cfg.get("divergence_lookback", 20)
    min_price_move = cfg.get("min_price_move_pct", 1.0) / 100

    obv_line = obv(c, v)
    obv_smooth = sma(obv_line, cfg.get("obv_smooth", 5))
    atr_val = atr(h, l, c, 14)

    signals = np.zeros(n)
    entry_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)
    tp_prices = np.full(n, np.nan)

    for i in range(lookback + 10, n):
        if np.isnan(obv_smooth[i]) or np.isnan(atr_val[i]):
            continue

        # Price and OBV over lookback window
        price_window = c[i-lookback:i+1]
        obv_window = obv_smooth[i-lookback:i+1]
        if np.any(np.isnan(obv_window)):
            continue

        # Bullish divergence: price at/near low, OBV trending up
        price_near_low = c[i] <= np.percentile(price_window, 10)
        obv_rising = obv_window[-1] > obv_window[0] and obv_window[-1] > np.mean(obv_window)
        price_dropped = (np.max(price_window) - c[i]) / np.max(price_window) > min_price_move

        if price_near_low and obv_rising and price_dropped:
            signals[i] = 1
            entry_prices[i] = c[i]
            sl_prices[i] = c[i] - 2 * atr_val[i]
            tp_prices[i] = c[i] + 4 * atr_val[i]  # RR 2:1

        # Bearish divergence: price at/near high, OBV trending down
        price_near_high = c[i] >= np.percentile(price_window, 90)
        obv_falling = obv_window[-1] < obv_window[0] and obv_window[-1] < np.mean(obv_window)
        price_rose = (c[i] - np.min(price_window)) / np.min(price_window) > min_price_move

        if price_near_high and obv_falling and price_rose:
            signals[i] = -1
            entry_prices[i] = c[i]
            sl_prices[i] = c[i] + 2 * atr_val[i]
            tp_prices[i] = c[i] - 4 * atr_val[i]

    return signals, entry_prices, sl_prices, tp_prices


# ═══════════════════════════════════════════════════════════════
# STRATEGY 3: VOLUME DRY-UP BREAKOUT (high-control coin mode)
# ═══════════════════════════════════════════════════════════════
# For high-control coins: volume nearly disappears for extended period
# → then a single massive candle (10x+ volume) = controlled breakout
# Much stricter than generic volume spike: requires the "quiet before storm"

def dryup_breakout_signals(o, h, l, c, v, cfg=None):
    """Detect volume dry-up → controlled breakout on high-control coins."""
    if cfg is None:
        cfg = {}
    n = len(c)

    lookback = cfg.get("dryup_lookback", 30)
    dry_pct = cfg.get("dry_percentile", 10)  # volume must be below 10th percentile
    min_dry_bars = cfg.get("min_dry_bars", 5)
    spike_mult = cfg.get("spike_mult", 10)   # breakout candle must be 10x dry volume
    min_candle_pct = cfg.get("min_candle_pct", 0.5) / 100  # candle body must be > 0.5%

    vol_ma = sma(v, lookback)
    atr_val = atr(h, l, c, 14)

    signals = np.zeros(n)
    entry_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)
    tp_prices = np.full(n, np.nan)

    for i in range(lookback + 10, n):
        if np.isnan(vol_ma[i]) or np.isnan(atr_val[i]):
            continue

        # Check for dry period before this bar
        dry_bars = 0
        dry_vol_avg = 0
        for j in range(i - min_dry_bars, i):
            if j < 0: continue
            # Is this bar's volume in the bottom percentile?
            vol_hist = v[max(0, j-100):j]
            if len(vol_hist) == 0: continue
            threshold = np.percentile(vol_hist, dry_pct)
            if v[j] <= threshold:
                dry_bars += 1
                dry_vol_avg += v[j]

        if dry_bars < min_dry_bars:
            continue
        dry_vol_avg /= max(dry_bars, 1)

        # Current bar must be a spike relative to the dry period
        if dry_vol_avg <= 0 or v[i] < spike_mult * dry_vol_avg:
            continue

        # Candle must have meaningful body (not a doji)
        candle_body = abs(c[i] - o[i]) / o[i]
        if candle_body < min_candle_pct:
            continue

        # Direction from candle
        if c[i] > o[i]:  # bullish candle
            signals[i] = 1
            entry_prices[i] = c[i]
            sl_prices[i] = l[i]  # SL at candle low
            risk = c[i] - l[i]
            tp_prices[i] = c[i] + 3 * risk if risk > 0 else c[i] * 1.02
        else:  # bearish candle
            signals[i] = -1
            entry_prices[i] = c[i]
            sl_prices[i] = h[i]  # SL at candle high
            risk = h[i] - c[i]
            tp_prices[i] = c[i] - 3 * risk if risk > 0 else c[i] * 0.98

    return signals, entry_prices, sl_prices, tp_prices


# ═══════════════════════════════════════════════════════════════
# UNIFIED BACKTEST
# ═══════════════════════════════════════════════════════════════

def backtest(signals, entry_prices, sl_prices, tp_prices, h, l, c,
             max_hold=50, fee=0.0005):
    """Generic backtest for any signal array."""
    n = len(c)
    pos = 0; entry = 0.0; sl_ = 0.0; tp_ = 0.0; ebar = 0
    equity = 1.0; peak = 1.0; mdd = 0.0
    returns = []; wins = 0; trades = 0; trade_log = []

    for i in range(1, n):
        if pos != 0:
            sh = (pos == 1 and l[i] <= sl_) or (pos == -1 and h[i] >= sl_)
            th = (pos == 1 and h[i] >= tp_) or (pos == -1 and l[i] <= tp_)
            to = (i - ebar) >= max_hold
            if th and sh:
                if (pos == 1 and c[i] < entry) or (pos == -1 and c[i] > entry): th = False
                else: sh = False
            ep = None; et = None
            if th: ep = tp_; et = "TP"
            elif sh: ep = sl_; et = "SL"
            elif to: ep = c[i]; et = "TIMEOUT"
            if ep is not None:
                pnl = ((ep - entry) / entry if pos == 1 else (entry - ep) / entry) - fee * 2
                equity *= (1 + pnl); returns.append(pnl)
                if pnl > 0: wins += 1
                trades += 1
                trade_log.append({"bar": ebar, "exit_bar": i, "side": pos,
                                  "entry": entry, "exit": ep, "pnl_pct": pnl * 100, "type": et})
                pos = 0
            peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak)

        if pos == 0 and signals[i] != 0 and not np.isnan(entry_prices[i]):
            pos = int(signals[i]); entry = entry_prices[i]
            sl_ = sl_prices[i]; tp_ = tp_prices[i]
            ebar = i

    if trades < 2:
        return {"trades": trades, "net_pct": 0, "sharpe": 0, "winrate": 0, "max_dd": 0, "trade_log": trade_log}
    r = np.array(returns)
    return {
        "trades": trades, "net_pct": (equity - 1) * 100,
        "sharpe": np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(len(r)),
        "winrate": wins / trades * 100, "max_dd": mdd * 100,
        "avg_win": np.mean([x * 100 for x in returns if x > 0]) if wins > 0 else 0,
        "avg_loss": np.mean([x * 100 for x in returns if x <= 0]) if trades > wins else 0,
        "trade_log": trade_log,
    }
