"""
Trendline Bounce Strategy
=========================
Core idea: Draw support/resistance trendlines from swing highs/lows,
project forward, place limit orders with buffer BEFORE price touches,
tight SL just beyond the line, TP at risk-reward ratio.

Logic:
  1. Detect swing highs and swing lows (local extrema)
  2. Connect pairs of swing lows → support trendlines
  3. Connect pairs of swing highs → resistance trendlines
  4. Validate: line must have "reaction" at both points (bounce)
  5. Project line forward to current bar
  6. If price is approaching the projected line → entry signal
  7. Long at support + buffer, Short at resistance - buffer
  8. SL just beyond the line, TP at RR ratio

Run backtest: python -m research.trendline_backtest
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Swing detection
    "swing_lookback": 10,       # bars each side to confirm swing high/low

    # Trendline validation
    "min_bars_between": 20,     # minimum bars between two anchor points
    "max_bars_between": 500,    # maximum bars between two anchor points
    "max_penetrations": 2,      # max times price can cross the line between anchors
    "max_post_penetrations": 0, # ANY break after anchor2 invalidates the line
    "min_bounces": 2,           # the two anchor points count as 2 bounces

    # Entry
    "buffer_pct": 0.12,         # 0.12% default; order manager uses dynamic 0.10-0.15%
    "approach_pct": 1.0,        # signal when price is within 1.0% of projected line

    # Exit — V3 model: stop = line itself (穿线即止损)
    # total risk = buffer_pct only (no separate sl_pct)
    "sl_pct": 0.0,              # 0% = stop AT the line (V3: 穿线即止损)
    "rr": 15.0,                 # RR=15: limit order backtest optimal (1M+ trades validated)

    # Execution
    "fee": 0.0005,              # 0.05% per side
    "max_hold_bars": 50,        # max bars to hold before timeout exit
}


# ═══════════════════════════════════════════════════════════════
# SWING DETECTION
# ═══════════════════════════════════════════════════════════════

def find_swing_highs(h, lookback=10):
    """Find bars where high[i] is highest within ±lookback bars."""
    n = len(h)
    swings = []
    for i in range(lookback, n - lookback):
        window = h[max(0, i - lookback):i + lookback + 1]
        if h[i] == np.max(window):
            swings.append(i)
    return np.array(swings, dtype=int)


def find_swing_lows(l, lookback=10):
    """Find bars where low[i] is lowest within ±lookback bars."""
    n = len(l)
    swings = []
    for i in range(lookback, n - lookback):
        window = l[max(0, i - lookback):i + lookback + 1]
        if l[i] == np.min(window):
            swings.append(i)
    return np.array(swings, dtype=int)


# ═══════════════════════════════════════════════════════════════
# TRENDLINE DETECTION
# ═══════════════════════════════════════════════════════════════

def build_trendlines(h, l, c, cfg):
    """
    Build all valid support and resistance trendlines.
    Returns list of dicts: {type, i1, i2, p1, p2, slope, intercept}
    Optimized: only connect each swing to its nearest N neighbors.
    """
    lb = cfg["swing_lookback"]
    swing_highs = find_swing_highs(h, lb)
    swing_lows = find_swing_lows(l, lb)
    min_gap = cfg["min_bars_between"]
    max_gap = cfg["max_bars_between"]
    max_pen = cfg["max_penetrations"]
    max_neighbors = 8  # only try nearest 8 swing points to limit O(n^2)

    lines = []

    n = len(c)
    # Max allowed penetrations AFTER anchor2 before line is considered broken
    max_post_pen = cfg.get("max_post_penetrations", 0)  # default: 0 = any break invalidates

    # ────────────────────────────────────────────────────────
    # LINE VALIDATION RULES:
    #
    # Any slope is allowed (ascending, descending, horizontal).
    # Rising channel upper = valid resistance. Falling channel lower = valid support.
    #
    # What matters is:
    # 1. REACTION at anchors: price must bounce away after touching
    #    (not just pass through). Measured as: did price move at least
    #    min_bounce_pct% away from the line within bounce_lookback bars?
    # 2. NOT BROKEN after anchor2: price hasn't decisively crossed
    #    the line since the second anchor point.
    # 3. Not too many penetrations between anchors.
    # ────────────────────────────────────────────────────────
    min_bounce_pct = cfg.get("min_bounce_pct", 0.3)  # 0.3% min reaction at each anchor
    bounce_lookback = cfg.get("bounce_lookback", 5)   # check reaction within 5 bars after anchor

    def _has_bounce(prices_after, anchor_price, direction, pct):
        """Check if price bounced at least pct% in the expected direction after anchor."""
        if len(prices_after) == 0:
            return True  # can't check, give benefit of doubt
        if direction == "support":
            # After touching support, price should go UP
            max_after = np.max(prices_after)
            return (max_after - anchor_price) / anchor_price * 100 >= pct
        else:
            # After touching resistance, price should go DOWN
            min_after = np.min(prices_after)
            return (anchor_price - min_after) / anchor_price * 100 >= pct

    # Support lines: connect pairs of swing lows
    for i in range(len(swing_lows)):
        count = 0
        for j in range(i + 1, len(swing_lows)):
            i1, i2 = swing_lows[i], swing_lows[j]
            gap = i2 - i1
            if gap < min_gap: continue
            if gap > max_gap: break
            count += 1
            if count > max_neighbors: break

            p1, p2 = l[i1], l[i2]
            slope = (p2 - p1) / gap
            intercept = p1 - slope * i1

            # BOUNCE CHECK: did price react at both anchor points?
            # Anchor 1: check if price bounced up after the low
            a1_end = min(i1 + bounce_lookback, n)
            if not _has_bounce(h[i1+1:a1_end], p1, "support", min_bounce_pct):
                continue

            # Anchor 2: check if price bounced up after the low
            a2_end = min(i2 + bounce_lookback, n)
            if not _has_bounce(h[i2+1:a2_end], p2, "support", min_bounce_pct):
                continue

            # Penetration check between anchors
            bars_between = np.arange(i1 + 1, i2)
            if len(bars_between) > 0:
                line_vals = slope * bars_between + intercept
                penetrations = np.sum(l[i1+1:i2] < line_vals * 0.999)
                if penetrations > max_pen: continue

            # Post-anchor check: line not broken since anchor2
            if i2 + 1 < n:
                post_bars = np.arange(i2 + 1, n)
                post_line = slope * post_bars + intercept
                post_breaks = np.sum(l[i2+1:n] < post_line)
                if post_breaks > max_post_pen:
                    continue

            lines.append({
                "type": "support", "i1": i1, "i2": i2,
                "p1": p1, "p2": p2, "slope": slope, "intercept": intercept,
            })

    # Resistance lines: connect pairs of swing highs
    for i in range(len(swing_highs)):
        count = 0
        for j in range(i + 1, len(swing_highs)):
            i1, i2 = swing_highs[i], swing_highs[j]
            gap = i2 - i1
            if gap < min_gap: continue
            if gap > max_gap: break
            count += 1
            if count > max_neighbors: break

            p1, p2 = h[i1], h[i2]
            slope = (p2 - p1) / gap
            intercept = p1 - slope * i1

            # BOUNCE CHECK: did price react at both anchor points?
            # Anchor 1: check if price bounced down after the high
            a1_end = min(i1 + bounce_lookback, n)
            if not _has_bounce(l[i1+1:a1_end], p1, "resistance", min_bounce_pct):
                continue

            # Anchor 2: check if price bounced down after the high
            a2_end = min(i2 + bounce_lookback, n)
            if not _has_bounce(l[i2+1:a2_end], p2, "resistance", min_bounce_pct):
                continue

            # Penetration check between anchors
            bars_between = np.arange(i1 + 1, i2)
            if len(bars_between) > 0:
                line_vals = slope * bars_between + intercept
                penetrations = np.sum(h[i1+1:i2] > line_vals * 1.001)
                if penetrations > max_pen: continue

            # Post-anchor check: line not broken since anchor2
            if i2 + 1 < n:
                post_bars = np.arange(i2 + 1, n)
                post_line = slope * post_bars + intercept
                post_breaks = np.sum(h[i2+1:n] > post_line)
                if post_breaks > max_post_pen:
                    continue

            lines.append({
                "type": "resistance", "i1": i1, "i2": i2,
                "p1": p1, "p2": p2, "slope": slope, "intercept": intercept,
            })

    return lines


def project_line(line, bar_idx):
    """Get the projected price of a trendline at a given bar index."""
    return line["slope"] * bar_idx + line["intercept"]


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_signals(o, h, l, c, v, cfg=None):
    """
    Generate entry signals based on trendline proximity.
    Returns: signals array (+1=long at support, -1=short at resistance),
             entry_prices, sl_prices, tp_prices, line_info
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    n = len(c)
    lines = build_trendlines(h, l, c, cfg)

    signals = np.zeros(n)
    entry_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)
    tp_prices = np.full(n, np.nan)
    line_refs = [None] * n

    approach_pct = cfg["approach_pct"] / 100
    buffer_pct = cfg["buffer_pct"] / 100
    sl_pct = cfg["sl_pct"] / 100
    rr = cfg["rr"]

    max_proj = cfg.get("max_projection_bars", 200)

    # Pre-compute: for each line, compute the range of bars where it's active
    # active range = (i2+1, i2+max_proj)
    # Build a list sorted by activation bar
    lines.sort(key=lambda x: x["i2"])

    # Sliding window of active lines
    active_lines = []
    line_ptr = 0

    for bar in range(1, n):
        # Add newly active lines
        while line_ptr < len(lines) and lines[line_ptr]["i2"] < bar:
            active_lines.append(lines[line_ptr])
            line_ptr += 1

        # Remove expired lines
        active_lines = [ln for ln in active_lines if bar - ln["i2"] <= max_proj]

        if not active_lines:
            continue

        best_line = None
        best_distance = float('inf')

        for line in active_lines:
            proj = line["slope"] * bar + line["intercept"]
            if proj <= 0: continue
            distance_pct = abs(c[bar] - proj) / proj
            if distance_pct > approach_pct: continue
            if line["type"] == "support" and c[bar] > proj and distance_pct < best_distance:
                best_distance = distance_pct; best_line = line
            elif line["type"] == "resistance" and c[bar] < proj and distance_pct < best_distance:
                best_distance = distance_pct; best_line = line

        if best_line is not None:
            proj = project_line(best_line, bar)

            if best_line["type"] == "support":
                # Long: enter slightly above the line
                entry_p = proj * (1 + buffer_pct)
                sl_p = proj * (1 - sl_pct)  # SL below the line
                risk = entry_p - sl_p
                tp_p = entry_p + rr * risk

                # Only signal if current price can fill at entry_p
                if l[bar] <= entry_p <= h[bar] or c[bar] <= entry_p:
                    signals[bar] = 1
                    entry_prices[bar] = entry_p
                    sl_prices[bar] = sl_p
                    tp_prices[bar] = tp_p
                    line_refs[bar] = best_line

            elif best_line["type"] == "resistance":
                # Short: enter slightly below the line
                entry_p = proj * (1 - buffer_pct)
                sl_p = proj * (1 + sl_pct)  # SL above the line
                risk = sl_p - entry_p
                tp_p = entry_p - rr * risk

                if l[bar] <= entry_p <= h[bar] or c[bar] >= entry_p:
                    signals[bar] = -1
                    entry_prices[bar] = entry_p
                    sl_prices[bar] = sl_p
                    tp_prices[bar] = tp_p
                    line_refs[bar] = best_line

    return signals, entry_prices, sl_prices, tp_prices, line_refs


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def backtest(o, h, l, c, v, cfg=None):
    """Run backtest on trendline bounce strategy."""
    if cfg is None:
        cfg = DEFAULT_CONFIG
    n = len(c)
    signals, entry_prices, sl_prices, tp_prices, line_refs = generate_signals(o, h, l, c, v, cfg)

    fee = cfg["fee"]
    max_hold = cfg["max_hold_bars"]

    pos = 0; entry = 0.0; sl_ = 0.0; tp_ = 0.0; entry_bar = 0
    equity = 1.0; peak = 1.0; max_dd = 0.0
    returns = []; wins = 0; trades = 0
    trade_log = []

    for i in range(1, n):
        if pos != 0:
            # Check SL/TP
            if pos == 1:
                sl_hit = l[i] <= sl_
                tp_hit = h[i] >= tp_
            else:
                sl_hit = h[i] >= sl_
                tp_hit = l[i] <= tp_

            # Timeout
            timeout = (i - entry_bar) >= max_hold

            if tp_hit and sl_hit:
                if (pos == 1 and c[i] < entry) or (pos == -1 and c[i] > entry):
                    tp_hit = False
                else:
                    sl_hit = False

            exit_price = None; exit_type = None
            if tp_hit:
                exit_price = tp_; exit_type = "TP"
            elif sl_hit:
                exit_price = sl_; exit_type = "SL"
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
                    "sl": sl_, "tp": tp_,
                    "pnl_pct": pnl * 100, "exit_type": exit_type,
                    "hold_bars": i - entry_bar,
                })
                pos = 0

            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)

        # New entry
        if pos == 0 and signals[i] != 0:
            pos = int(signals[i])
            entry = entry_prices[i]
            sl_ = sl_prices[i]
            tp_ = tp_prices[i]
            entry_bar = i

    if trades < 2:
        return {"net_pct": 0, "sharpe": 0, "winrate": 0, "trades": trades,
                "max_dd": 0, "trade_log": trade_log, "lines_found": len(
                    build_trendlines(h, l, c, cfg))}

    r = np.array(returns)
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(len(r))
    return {
        "net_pct": (equity - 1) * 100,
        "sharpe": sharpe,
        "winrate": wins / trades * 100,
        "trades": trades,
        "max_dd": max_dd * 100,
        "avg_win": np.mean([x for x in returns if x > 0]) * 100 if wins > 0 else 0,
        "avg_loss": np.mean([x for x in returns if x <= 0]) * 100 if trades > wins else 0,
        "avg_hold": np.mean([t["hold_bars"] for t in trade_log]),
        "trade_log": trade_log,
        "lines_found": len(build_trendlines(h, l, c, cfg)),
    }
