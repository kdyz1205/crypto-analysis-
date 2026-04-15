"""
Volume Spike Strategy
=====================
Entry: 2-bar volume > N × historical 2-bar average → buy/short based on price direction
Exit:  Fixed SL% / TP% or trailing, learned from data

Targets: high-control coins (low float, high vol spike = someone is loading/dumping)

Run backtest: python -m research.volume_spike_backtest
"""

import numpy as np


DEFAULT_CONFIG = {
    # Spike detection
    "spike_window": 2,          # sum volume over N bars (2 = 2-minute window on 1m data)
    "lookback": 100,            # historical average lookback (100 bars)
    "spike_mult": 15,           # volume must be >= 15x average

    # Direction detection
    "direction_bars": 2,        # look at price change over last N bars to determine direction
    # If price went UP during spike → long (someone is buying aggressively)
    # If price went DOWN during spike → short (someone is dumping)

    # Exit
    "sl_pct": 0.5,              # stop loss 0.5%
    "tp_pct": 2.0,              # take profit 2.0% (RR = 4:1)
    "max_hold_bars": 30,        # timeout after 30 bars (30 min on 1m)
    "trailing_activate_pct": 0.5,  # activate trailing after 0.5% profit
    "trailing_pct": 0.3,        # trail at 0.3% from peak

    # Filters
    "min_price_move_pct": 0.1,  # spike must have at least 0.1% price move (not just wash trading)
    "cooldown_bars": 5,         # no new entry within 5 bars of last exit
    "fee": 0.0005,
}


def detect_spikes(c, v, cfg=None):
    """
    Detect volume spikes and their direction.
    Returns: (spike_bars, directions)
      spike_bars: array of bar indices where spike occurred
      directions: +1 (long) or -1 (short) for each spike
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    n = len(c)
    sw = cfg["spike_window"]
    lb = cfg["lookback"]
    mult = cfg["spike_mult"]
    dir_bars = cfg["direction_bars"]
    min_move = cfg["min_price_move_pct"] / 100

    if n < lb + sw + 10:
        return np.array([], dtype=int), np.array([])

    # Rolling N-bar volume sum
    vol_sum = np.zeros(n)
    for i in range(sw - 1, n):
        vol_sum[i] = np.sum(v[i - sw + 1:i + 1])

    # Rolling average of vol_sum
    vol_avg = np.full(n, np.nan)
    for i in range(lb + sw - 1, n):
        vol_avg[i] = np.mean(vol_sum[i - lb:i])

    spikes = []
    directions = []

    for i in range(lb + sw, n):
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 0:
            continue
        if vol_sum[i] < mult * vol_avg[i]:
            continue

        # Price move during spike window
        price_change = (c[i] - c[max(0, i - dir_bars)]) / c[max(0, i - dir_bars)]
        if abs(price_change) < min_move:
            continue  # wash trading / no real move

        direction = 1 if price_change > 0 else -1
        spikes.append(i)
        directions.append(direction)

    return np.array(spikes, dtype=int), np.array(directions)


def backtest(c, v, h=None, l=None, cfg=None):
    """
    Backtest volume spike strategy.
    h, l are optional (uses c if not provided for SL/TP check).
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    if h is None:
        h = c
    if l is None:
        l = c

    n = len(c)
    spike_bars, directions = detect_spikes(c, v, cfg)

    if len(spike_bars) == 0:
        return {"trades": 0, "net_pct": 0, "sharpe": 0, "winrate": 0,
                "max_dd": 0, "spikes_found": 0, "trade_log": []}

    sl_pct = cfg["sl_pct"] / 100
    tp_pct = cfg["tp_pct"] / 100
    max_hold = cfg["max_hold_bars"]
    trail_act = cfg["trailing_activate_pct"] / 100
    trail_pct = cfg["trailing_pct"] / 100
    fee = cfg["fee"]
    cooldown = cfg["cooldown_bars"]

    pos = 0; entry = 0.0; sl_ = 0.0; tp_ = 0.0; entry_bar = 0
    peak_profit = 0.0; trailing_sl = 0.0
    equity = 1.0; peak_eq = 1.0; max_dd = 0.0
    returns = []; wins = 0; trades = 0
    trade_log = []
    last_exit_bar = -cooldown - 1

    spike_set = set(spike_bars.tolist())
    spike_dir = dict(zip(spike_bars.tolist(), directions.tolist()))

    for i in range(1, n):
        if pos != 0:
            # Current unrealized profit
            if pos == 1:
                unrealized = (c[i] - entry) / entry
            else:
                unrealized = (entry - c[i]) / entry

            # Update trailing SL
            if unrealized > trail_act:
                peak_profit = max(peak_profit, unrealized)
                new_trail = entry * (1 + (peak_profit - trail_pct)) if pos == 1 else entry * (1 - (peak_profit - trail_pct))
                if pos == 1:
                    trailing_sl = max(trailing_sl, new_trail)
                else:
                    trailing_sl = min(trailing_sl, new_trail) if trailing_sl > 0 else new_trail

            # Check exits
            sl_hit = (pos == 1 and l[i] <= sl_) or (pos == -1 and h[i] >= sl_)
            tp_hit = (pos == 1 and h[i] >= tp_) or (pos == -1 and l[i] <= tp_)

            # Trailing SL check
            trail_hit = False
            if trailing_sl > 0:
                if pos == 1 and l[i] <= trailing_sl:
                    trail_hit = True
                elif pos == -1 and h[i] >= trailing_sl:
                    trail_hit = True

            timeout = (i - entry_bar) >= max_hold

            exit_price = None; exit_type = None
            if tp_hit:
                exit_price = tp_; exit_type = "TP"
            elif sl_hit:
                exit_price = sl_; exit_type = "SL"
            elif trail_hit:
                exit_price = trailing_sl; exit_type = "TRAIL"
            elif timeout:
                exit_price = c[i]; exit_type = "TIMEOUT"

            if exit_price is not None:
                if pos == 1:
                    pnl = (exit_price - entry) / entry - fee * 2
                else:
                    pnl = (entry - exit_price) / entry - fee * 2

                equity *= (1 + pnl); returns.append(pnl)
                if pnl > 0: wins += 1
                trades += 1
                trade_log.append({
                    "bar_entry": entry_bar, "bar_exit": i,
                    "side": "LONG" if pos == 1 else "SHORT",
                    "entry": entry, "exit": exit_price,
                    "pnl_pct": pnl * 100, "exit_type": exit_type,
                    "hold_bars": i - entry_bar,
                    "vol_mult": vol_sum_at_entry,
                })
                pos = 0
                last_exit_bar = i
                peak_profit = 0.0; trailing_sl = 0.0

            peak_eq = max(peak_eq, equity)
            max_dd = max(max_dd, (peak_eq - equity) / peak_eq)

        # New entry on spike
        if pos == 0 and i in spike_set and (i - last_exit_bar) >= cooldown:
            direction = spike_dir[i]
            pos = direction
            entry = c[i]
            if pos == 1:
                sl_ = entry * (1 - sl_pct)
                tp_ = entry * (1 + tp_pct)
            else:
                sl_ = entry * (1 + sl_pct)
                tp_ = entry * (1 - tp_pct)
            entry_bar = i
            trailing_sl = 0.0; peak_profit = 0.0

            # Store volume info for logging
            sw = cfg["spike_window"]
            vol_sum_at_entry = np.sum(v[max(0, i-sw+1):i+1])

    if trades < 2:
        return {"trades": trades, "net_pct": 0, "sharpe": 0, "winrate": 0,
                "max_dd": 0, "spikes_found": len(spike_bars), "trade_log": trade_log}

    r = np.array(returns)
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(len(r))
    return {
        "trades": trades,
        "net_pct": (equity - 1) * 100,
        "sharpe": sharpe,
        "winrate": wins / trades * 100,
        "max_dd": max_dd * 100,
        "spikes_found": len(spike_bars),
        "avg_hold": np.mean([t["hold_bars"] for t in trade_log]),
        "avg_win": np.mean([t["pnl_pct"] for t in trade_log if t["pnl_pct"] > 0]) if wins > 0 else 0,
        "avg_loss": np.mean([t["pnl_pct"] for t in trade_log if t["pnl_pct"] <= 0]) if trades > wins else 0,
        "trade_log": trade_log,
    }
