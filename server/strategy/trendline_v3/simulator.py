"""V3 simulator — buffer-only model, line IS the stop.

Rules (from TRENDLINE_TRADING_RULES.md):
  entry = line_price + buffer   (long)  or  line_price - buffer  (short)
  stop  = line_price            (穿线即止损)
  total_price_risk = buffer
  leverage = account_risk_pct / buffer

No separate entry_buffer / stop_buffer. One number: buffer.
"""
import numpy as np
from numba import njit

EXIT_WIN = 1
EXIT_STOP = 2
EXIT_LINEBENT = 3
EXIT_TIMEOUT = 4
EXIT_SWEEP = 5


@njit(cache=True)
def simulate_line_v3(
    highs, lows, closes, atr,
    bb_upper, bb_lower,
    slope, intercept, kind_is_support,
    buffer,           # single number: distance from line to entry
    target_rr,        # target = entry ± buffer × rr
    max_hold_bars,
    pivot_k,
    max_track_bars,
    anchor2_idx,
    pre_break_atr,
    # outputs (length 1 arrays)
    out_retest_bar, out_exit_bar, out_entry_price, out_exit_price,
    out_pnl_pct, out_exit_code,
):
    n = len(highs)
    start = anchor2_idx + pivot_k + 1
    if start >= n:
        return 0
    end = min(n, start + max_track_bars)

    last_touch = -5
    b = start
    while b < end:
        cur_atr = atr[b]
        if cur_atr <= 0.0 or not np.isfinite(cur_atr):
            b += 1
            continue

        line_p = slope * b + intercept

        # Pre-touch decisive break kills line
        if kind_is_support and closes[b] < line_p - pre_break_atr * cur_atr:
            return 0
        if (not kind_is_support) and closes[b] > line_p + pre_break_atr * cur_atr:
            return 0

        # Entry + stop
        if kind_is_support:
            entry_price = line_p * (1.0 + buffer)
            stop_price = line_p  # LINE IS THE STOP
            entry_hits = lows[b] <= entry_price
            sweep = entry_hits and (lows[b] <= stop_price)
        else:
            entry_price = line_p * (1.0 - buffer)
            stop_price = line_p
            entry_hits = highs[b] >= entry_price
            sweep = entry_hits and (highs[b] >= stop_price)

        if not entry_hits:
            b += 1
            continue
        if b - last_touch < 3:
            b += 1
            continue
        last_touch = b

        if sweep:
            out_retest_bar[0] = b
            out_exit_bar[0] = b
            out_entry_price[0] = entry_price
            out_exit_price[0] = entry_price
            out_pnl_pct[0] = 0.0
            out_exit_code[0] = EXIT_SWEEP
            return 1

        # Target: fixed RR or BB, whichever hits first (hybrid)
        risk = buffer  # total price risk = buffer
        if kind_is_support:
            pct_target = entry_price * (1.0 + risk * target_rr)
        else:
            pct_target = entry_price * (1.0 - risk * target_rr)

        exit_bar = -1
        exit_price = 0.0
        exit_code = 0
        max_fav = 0.0

        for k in range(1, max_hold_bars + 1):
            j = b + k
            if j >= n:
                break
            bar_h = highs[j]
            bar_l = lows[j]

            # Walking stop: line moves each bar
            line_j = slope * j + intercept
            if kind_is_support:
                cur_stop = line_j  # stop IS the line
            else:
                cur_stop = line_j

            # MFE
            if kind_is_support:
                fav = (bar_h - entry_price) / entry_price
            else:
                fav = (entry_price - bar_l) / entry_price
            if fav > max_fav:
                max_fav = fav

            # Pessimistic: stop first
            if kind_is_support:
                if bar_l <= cur_stop:
                    exit_bar = j
                    exit_price = cur_stop
                    exit_code = EXIT_LINEBENT if max_fav >= 2.0 * risk else EXIT_STOP
                    break
                # Target: pct OR BB upper
                hit_pct = bar_h >= pct_target
                hit_bb = np.isfinite(bb_upper[j]) and bar_h >= bb_upper[j]
                if hit_pct or hit_bb:
                    if hit_pct and hit_bb:
                        exit_price = min(pct_target, bb_upper[j])
                    elif hit_pct:
                        exit_price = pct_target
                    else:
                        exit_price = bb_upper[j]
                    exit_bar = j
                    exit_code = EXIT_WIN
                    break
            else:
                if bar_h >= cur_stop:
                    exit_bar = j
                    exit_price = cur_stop
                    exit_code = EXIT_LINEBENT if max_fav >= 2.0 * risk else EXIT_STOP
                    break
                hit_pct = bar_l <= pct_target
                hit_bb = np.isfinite(bb_lower[j]) and bar_l <= bb_lower[j]
                if hit_pct or hit_bb:
                    if hit_pct and hit_bb:
                        exit_price = max(pct_target, bb_lower[j])
                    elif hit_pct:
                        exit_price = pct_target
                    else:
                        exit_price = bb_lower[j]
                    exit_bar = j
                    exit_code = EXIT_WIN
                    break

        if exit_bar < 0:
            last_j = min(b + max_hold_bars, n - 1)
            exit_bar = last_j
            exit_price = closes[last_j]
            exit_code = EXIT_TIMEOUT

        if kind_is_support:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        out_retest_bar[0] = b
        out_exit_bar[0] = exit_bar
        out_entry_price[0] = entry_price
        out_exit_price[0] = exit_price
        out_pnl_pct[0] = pnl
        out_exit_code[0] = exit_code
        return 1  # first retest only

    return 0


@njit(cache=True)
def simulate_all_v3(
    highs, lows, closes, atr, bb_upper, bb_lower,
    anchor1, anchor2, slopes, intercepts, kinds_is_support,
    buffer, target_rr, max_hold_bars, pivot_k,
    pre_break_atr=0.8, max_track_bars=300,
):
    n_lines = len(slopes)
    retest_bars = np.zeros(n_lines, dtype=np.int64)
    exit_bars = np.zeros(n_lines, dtype=np.int64)
    entry_prices = np.zeros(n_lines, dtype=np.float64)
    exit_prices = np.zeros(n_lines, dtype=np.float64)
    pnl_pct = np.zeros(n_lines, dtype=np.float64)
    exit_codes = np.zeros(n_lines, dtype=np.int8)
    kinds_out = np.zeros(n_lines, dtype=np.int8)
    valid = np.zeros(n_lines, dtype=np.int8)

    tmp_rb = np.zeros(1, dtype=np.int64)
    tmp_eb = np.zeros(1, dtype=np.int64)
    tmp_ep = np.zeros(1, dtype=np.float64)
    tmp_xp = np.zeros(1, dtype=np.float64)
    tmp_pnl = np.zeros(1, dtype=np.float64)
    tmp_code = np.zeros(1, dtype=np.int8)

    total = 0
    for i in range(n_lines):
        n_ev = simulate_line_v3(
            highs, lows, closes, atr, bb_upper, bb_lower,
            slopes[i], intercepts[i], kinds_is_support[i],
            buffer, target_rr, max_hold_bars, pivot_k, max_track_bars,
            anchor2[i], pre_break_atr,
            tmp_rb, tmp_eb, tmp_ep, tmp_xp, tmp_pnl, tmp_code,
        )
        if n_ev > 0:
            retest_bars[total] = tmp_rb[0]
            exit_bars[total] = tmp_eb[0]
            entry_prices[total] = tmp_ep[0]
            exit_prices[total] = tmp_xp[0]
            pnl_pct[total] = tmp_pnl[0]
            exit_codes[total] = tmp_code[0]
            kinds_out[total] = 1 if kinds_is_support[i] else 0
            total += 1

    return (
        retest_bars[:total], exit_bars[:total],
        entry_prices[:total], exit_prices[:total],
        pnl_pct[:total], exit_codes[:total], kinds_out[:total],
    )


def run_v3(df, records, buffer, target_rr, max_hold_bars=100,
           pivot_k=3, bb_period=21, bb_std=2.1, atr_period=14):
    from ts.research.pivots import compute_atr
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    atr = compute_atr(df, atr_period).values.astype(np.float64)
    ma = df["close"].rolling(bb_period).mean()
    sd = df["close"].rolling(bb_period).std()
    bb_u = (ma + bb_std * sd).values.astype(np.float64)
    bb_l = (ma - bb_std * sd).values.astype(np.float64)

    n_lines = len(records)
    a1 = np.zeros(n_lines, dtype=np.int64)
    a2 = np.zeros(n_lines, dtype=np.int64)
    sl = np.zeros(n_lines, dtype=np.float64)
    ic = np.zeros(n_lines, dtype=np.float64)
    ks = np.zeros(n_lines, dtype=np.bool_)
    for i, r in enumerate(records):
        a1[i] = r.anchor1_idx
        a2[i] = r.anchor2_idx
        sl[i] = r.slope
        ic[i] = r.intercept
        ks[i] = (r.kind == "support")

    return simulate_all_v3(
        highs, lows, closes, atr, bb_u, bb_l,
        a1, a2, sl, ic, ks,
        float(buffer), float(target_rr), int(max_hold_bars), int(pivot_k),
    )
