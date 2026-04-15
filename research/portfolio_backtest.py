"""
Portfolio-Level Backtest: Position Sizing + Risk Management
============================================================
Proper time-aligned simulation across multiple symbols × strategies.
Each trade carries real timestamps, equity is tracked on a unified timeline.

Run: python -m research.portfolio_backtest
"""

import gzip, csv, sys, os, numpy as np
from datetime import datetime, timezone
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.strategy import Strategy as RibbonStrategy
from research.trendline_strategy import backtest as trendline_bt, DEFAULT_CONFIG as TL_DEFAULT

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_ohlcv(symbol, tf):
    fn = f"data/{symbol}_{tf}.csv.gz"
    if not os.path.exists(fn): return None
    with gzip.open(fn, "rt") as f:
        reader = csv.DictReader(f); rows = list(reader)
    if len(rows) < 500: return None
    return {
        "o": np.array([float(r["open"]) for r in rows]),
        "h": np.array([float(r["high"]) for r in rows]),
        "l": np.array([float(r["low"]) for r in rows]),
        "c": np.array([float(r["close"]) for r in rows]),
        "v": np.array([float(r["volume"]) for r in rows]),
        "ts": [r["open_time"] for r in rows],
    }


# ═══════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════

def size_fixed_risk(equity, risk_pct, entry, sl, max_position_pct=0.15):
    """Risk exactly risk_pct of equity. Size = risk$ / per-unit-risk."""
    if entry <= 0 or sl == 0:
        return equity * 0.02
    risk_per_unit = abs(entry - sl) / entry
    if risk_per_unit < 1e-8:
        return equity * 0.02
    position = (equity * risk_pct) / risk_per_unit
    return min(position, equity * max_position_pct)


def size_fixed_pct(equity, pct):
    """Fixed percentage of equity per trade."""
    return equity * pct


# ═══════════════════════════════════════════════════════════════
# TRADE COLLECTION (with real timestamps)
# ═══════════════════════════════════════════════════════════════

def collect_ribbon_trades(symbols, tfs):
    """Run ribbon strategy, return trades with real timestamps."""
    s = RibbonStrategy()
    trades = []
    for sym in symbols:
        for tf in tfs:
            d = load_ohlcv(sym, tf)
            if d is None:
                continue
            bt = s.backtest(d["o"], d["h"], d["l"], d["c"], d["v"])
            ts_arr = d["ts"]
            for t in bt["trade_log"]:
                bar_i = t["bar"]
                if bar_i < 0 or bar_i >= len(ts_arr):
                    continue
                trades.append({
                    "ts_entry": ts_arr[bar_i],
                    "ts_exit": ts_arr[min(bar_i + 1, len(ts_arr) - 1)],
                    "entry": float(t["entry"]),
                    "exit": float(t["exit"]),
                    "sl": float(t["entry"] * (0.97 if t["side"] == 1 else 1.03)),
                    "pnl_pct": float(t["pnl_pct"]),
                    "exit_type": t["type"],
                    "symbol": sym.upper(),
                    "tf": tf,
                    "strategy": "ribbon",
                    "side": "LONG" if t["side"] == 1 else "SHORT",
                })
    return trades


def collect_trendline_trades(symbols, tfs):
    """Run trendline strategy, return trades with real timestamps."""
    cfg = {**TL_DEFAULT,
           "swing_lookback": 20, "buffer_pct": 0.25, "sl_pct": 0.4, "rr": 2.0,
           "max_hold_bars": 100, "approach_pct": 2.0, "min_bars_between": 40,
           "max_bars_between": 1000, "max_penetrations": 2}
    trades = []
    for sym in symbols:
        for tf in tfs:
            d = load_ohlcv(sym, tf)
            if d is None:
                continue
            bt = trendline_bt(d["o"], d["h"], d["l"], d["c"], d["v"], cfg)
            ts_arr = d["ts"]
            for t in bt["trade_log"]:
                be, bx = t["bar_entry"], t["bar_exit"]
                if be < 0 or be >= len(ts_arr) or bx < 0 or bx >= len(ts_arr):
                    continue
                trades.append({
                    "ts_entry": ts_arr[be],
                    "ts_exit": ts_arr[bx],
                    "entry": float(t["entry"]),
                    "exit": float(t["exit"]),
                    "sl": float(t["sl"]),
                    "pnl_pct": float(t["pnl_pct"]),
                    "exit_type": t["exit_type"],
                    "symbol": sym.upper(),
                    "tf": tf,
                    "strategy": "trendline",
                    "side": t["side"],
                })
    return trades


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATOR (time-aligned)
# ═══════════════════════════════════════════════════════════════

def simulate_portfolio(trades, sizing="fixed_risk", sizing_params=None,
                       initial_equity=10000, max_concurrent=5,
                       max_per_symbol=1, dd_halt_pct=0.25):
    """
    Time-aligned portfolio simulation.
    - Sorts all trades by real timestamp (ts_entry)
    - Tracks open positions by exit timestamp
    - Enforces concurrent limits, per-symbol limits, DD halt
    - Returns detailed results
    """
    if sizing_params is None:
        sizing_params = {"risk_pct": 0.01}

    # Sort by entry timestamp
    trades = sorted(trades, key=lambda t: t["ts_entry"])

    equity = float(initial_equity)
    peak = equity
    max_dd = 0.0
    open_positions = []  # each: {ts_exit, symbol, size, pnl_pct, ...}
    closed_trades = []
    halted = False
    equity_snapshots = []  # (timestamp, equity)

    for trade in trades:
        if halted:
            break

        ts_now = trade["ts_entry"]

        # Close positions that have exited by now
        still_open = []
        for pos in open_positions:
            if pos["ts_exit"] <= ts_now:
                pnl_dollars = pos["size"] * pos["pnl_pct"] / 100
                equity += pnl_dollars
                closed_trades.append({
                    **pos,
                    "pnl_dollars": pnl_dollars,
                    "equity_after": equity,
                })
                equity_snapshots.append((pos["ts_exit"], equity))
            else:
                still_open.append(pos)
        open_positions = still_open

        # Update drawdown
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        # DD halt
        if dd >= dd_halt_pct:
            halted = True
            continue

        # Concurrent limit
        if len(open_positions) >= max_concurrent:
            continue

        # Per-symbol limit
        sym_open = sum(1 for p in open_positions if p["symbol"] == trade["symbol"])
        if sym_open >= max_per_symbol:
            continue

        # Don't open same symbol+tf+strategy if already open
        dup = any(p["symbol"] == trade["symbol"] and p["tf"] == trade["tf"]
                  and p["strategy"] == trade["strategy"] for p in open_positions)
        if dup:
            continue

        # Position sizing
        if sizing == "fixed_risk":
            size = size_fixed_risk(
                equity, sizing_params.get("risk_pct", 0.01),
                trade["entry"], trade["sl"],
                sizing_params.get("max_pos_pct", 0.15),
            )
        elif sizing == "fixed_pct":
            size = size_fixed_pct(equity, sizing_params.get("pct", 0.02))
        else:
            size = equity * 0.02

        if size <= 0 or equity <= 0:
            continue

        open_positions.append({
            "ts_entry": trade["ts_entry"],
            "ts_exit": trade["ts_exit"],
            "symbol": trade["symbol"],
            "tf": trade["tf"],
            "strategy": trade["strategy"],
            "side": trade["side"],
            "entry": trade["entry"],
            "exit": trade["exit"],
            "sl": trade["sl"],
            "pnl_pct": trade["pnl_pct"],
            "exit_type": trade["exit_type"],
            "size": size,
        })

    # Close remaining open positions
    for pos in open_positions:
        pnl_dollars = pos["size"] * pos["pnl_pct"] / 100
        equity += pnl_dollars
        closed_trades.append({
            **pos,
            "pnl_dollars": pnl_dollars,
            "equity_after": equity,
        })

    peak = max(peak, equity)
    max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)

    # Compute stats
    n = len(closed_trades)
    if n < 2:
        return {"net_pct": 0, "sharpe": 0, "winrate": 0, "trades": n,
                "max_dd": 0, "final_equity": equity, "halted": halted,
                "closed_trades": closed_trades}

    pnls = np.array([t["pnl_pct"] / 100 for t in closed_trades])
    wins = np.sum(pnls > 0)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-12) * np.sqrt(n)

    # Per-strategy breakdown
    strat_stats = {}
    for sname in set(t["strategy"] for t in closed_trades):
        st = [t for t in closed_trades if t["strategy"] == sname]
        st_pnls = np.array([t["pnl_pct"] / 100 for t in st])
        st_wins = np.sum(st_pnls > 0)
        st_dollars = sum(t["pnl_dollars"] for t in st)
        strat_stats[sname] = {
            "trades": len(st),
            "winrate": st_wins / len(st) * 100 if st else 0,
            "net_dollars": st_dollars,
            "avg_pnl_pct": np.mean(st_pnls) * 100,
        }

    # Per-symbol breakdown
    sym_stats = {}
    for sym in set(t["symbol"] for t in closed_trades):
        st = [t for t in closed_trades if t["symbol"] == sym]
        st_dollars = sum(t["pnl_dollars"] for t in st)
        sym_stats[sym] = {
            "trades": len(st),
            "net_dollars": st_dollars,
            "winrate": sum(1 for t in st if t["pnl_pct"] > 0) / len(st) * 100,
        }

    # Max concurrent (scan closed trades timeline)
    events = []
    for t in closed_trades:
        events.append((t["ts_entry"], +1))
        events.append((t["ts_exit"], -1))
    events.sort()
    concurrent = 0; max_conc = 0
    for _, delta in events:
        concurrent += delta
        max_conc = max(max_conc, concurrent)

    return {
        "net_pct": (equity / initial_equity - 1) * 100,
        "sharpe": sharpe,
        "winrate": wins / n * 100,
        "trades": n,
        "max_dd": max_dd * 100,
        "final_equity": equity,
        "halted": halted,
        "avg_pnl_pct": np.mean(pnls) * 100,
        "avg_win_pct": np.mean(pnls[pnls > 0]) * 100 if wins > 0 else 0,
        "avg_loss_pct": np.mean(pnls[pnls <= 0]) * 100 if n > wins else 0,
        "max_concurrent": max_conc,
        "strategy_breakdown": strat_stats,
        "symbol_breakdown": sym_stats,
        "closed_trades": closed_trades,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "dogeusdt", "nearusdt",
           "adausdt", "bnbusdt", "linkusdt", "pepeusdt", "suiusdt", "xrpusdt"]
TFS = ["1h", "4h"]


def fmt(r):
    """Format a result dict into a summary line."""
    h = "HALT" if r["halted"] else ""
    return (f"trades={r['trades']:5d}  win={r['winrate']:5.1f}%  sharpe={r['sharpe']:+7.2f}  "
            f"net={r['net_pct']:+10.1f}%  DD={r['max_dd']:5.1f}%  "
            f"final=${r['final_equity']:12,.0f}  maxConc={r.get('max_concurrent',0):2d} {h}")


def main():
    print("=" * 105, flush=True)
    print("PORTFOLIO-LEVEL BACKTEST (time-aligned)", flush=True)
    print(f"  {len(SYMBOLS)} symbols x {len(TFS)} TFs x 2 strategies", flush=True)
    print("=" * 105, flush=True)

    # ── Collect trades ──
    print("\n[1] Collecting MA Ribbon trades...", flush=True)
    rib = collect_ribbon_trades(SYMBOLS, TFS)
    print(f"  {len(rib)} trades | ts range: {rib[0]['ts_entry'][:10]} to {rib[-1]['ts_entry'][:10]}" if rib else "  0 trades", flush=True)

    print("[2] Collecting Trendline trades (4h only)...", flush=True)
    tl = collect_trendline_trades(SYMBOLS, ["4h"])
    print(f"  {len(tl)} trades | ts range: {tl[0]['ts_entry'][:10]} to {tl[-1]['ts_entry'][:10]}" if tl else "  0 trades", flush=True)

    both = rib + tl
    print(f"  Combined: {len(both)} trades\n", flush=True)

    # ── Test 1: Sizing methods ──
    print("=" * 105, flush=True)
    print("TEST 1: SIZING METHOD (concurrent=5, dd_halt=25%)", flush=True)
    print("=" * 105, flush=True)

    sizing_tests = [
        ("fixed_2%",    "fixed_pct",  {"pct": 0.02}),
        ("fixed_5%",    "fixed_pct",  {"pct": 0.05}),
        ("fixed_10%",   "fixed_pct",  {"pct": 0.10}),
        ("risk_0.5%",   "fixed_risk", {"risk_pct": 0.005}),
        ("risk_1%",     "fixed_risk", {"risk_pct": 0.01}),
        ("risk_2%",     "fixed_risk", {"risk_pct": 0.02}),
        ("risk_3%",     "fixed_risk", {"risk_pct": 0.03}),
    ]

    for label, method, params in sizing_tests:
        print(f"\n  --- {label} ---", flush=True)
        for sname, strads in [("ribbon", rib), ("trendline", tl), ("BOTH", both)]:
            r = simulate_portfolio(strads, method, params,
                                   initial_equity=10000, max_concurrent=5, dd_halt_pct=0.25)
            print(f"    {sname:<12} {fmt(r)}", flush=True)

    # ── Test 2: Concurrent limits ──
    print("\n" + "=" * 105, flush=True)
    print("TEST 2: CONCURRENT LIMIT (risk_1%, dd_halt=25%)", flush=True)
    print("=" * 105, flush=True)

    for mc in [1, 2, 3, 5, 8, 10]:
        r = simulate_portfolio(both, "fixed_risk", {"risk_pct": 0.01},
                               initial_equity=10000, max_concurrent=mc, dd_halt_pct=0.25)
        print(f"  max_concurrent={mc:2d}  {fmt(r)}", flush=True)

    # ── Test 3: DD halt threshold ──
    print("\n" + "=" * 105, flush=True)
    print("TEST 3: DD HALT THRESHOLD (risk_1%, concurrent=5)", flush=True)
    print("=" * 105, flush=True)

    for dd in [0.10, 0.15, 0.20, 0.25, 0.50, 1.0]:
        r = simulate_portfolio(both, "fixed_risk", {"risk_pct": 0.01},
                               initial_equity=10000, max_concurrent=5, dd_halt_pct=dd)
        label = f"{dd*100:.0f}%" if dd < 1.0 else "OFF"
        print(f"  dd_halt={label:>4}  {fmt(r)}", flush=True)

    # ── Test 4: Strategy mix ──
    print("\n" + "=" * 105, flush=True)
    print("TEST 4: STRATEGY ALLOCATION (risk_1%, concurrent=5, dd_halt=25%)", flush=True)
    print("=" * 105, flush=True)

    for label, trades in [("ribbon only", rib), ("trendline only", tl), ("both", both)]:
        r = simulate_portfolio(trades, "fixed_risk", {"risk_pct": 0.01},
                               initial_equity=10000, max_concurrent=5, dd_halt_pct=0.25)
        print(f"\n  {label}:", flush=True)
        print(f"    {fmt(r)}", flush=True)
        if r.get("strategy_breakdown"):
            for sn, ss in r["strategy_breakdown"].items():
                print(f"      {sn:<12} trades={ss['trades']:5d}  win={ss['winrate']:5.1f}%  "
                      f"pnl=${ss['net_dollars']:+12,.0f}  avg={ss['avg_pnl_pct']:+5.2f}%", flush=True)

    # ── Detailed: Best config symbol breakdown ──
    print("\n" + "=" * 105, flush=True)
    print("BEST CONFIG DETAIL: risk_1% + concurrent=5 + both strategies", flush=True)
    print("=" * 105, flush=True)

    r = simulate_portfolio(both, "fixed_risk", {"risk_pct": 0.01},
                           initial_equity=10000, max_concurrent=5, dd_halt_pct=0.25)

    print(f"\n  OVERALL: {fmt(r)}\n", flush=True)

    print(f"  {'SYMBOL':<10} {'trades':>6} {'win%':>6} {'net$':>12} {'strategy_mix'}", flush=True)
    print("  " + "-" * 60, flush=True)
    for sym in sorted(r.get("symbol_breakdown", {}), key=lambda s: r["symbol_breakdown"][s]["net_dollars"], reverse=True):
        ss = r["symbol_breakdown"][sym]
        # Count per strategy
        rib_n = sum(1 for t in r["closed_trades"] if t["symbol"] == sym and t["strategy"] == "ribbon")
        tl_n = sum(1 for t in r["closed_trades"] if t["symbol"] == sym and t["strategy"] == "trendline")
        print(f"  {sym:<10} {ss['trades']:6d} {ss['winrate']:5.1f}% ${ss['net_dollars']:+11,.0f}  "
              f"rib={rib_n} tl={tl_n}", flush=True)

    # ── Sample trades (first 15 + last 15) ──
    ct = r["closed_trades"]
    print(f"\n  SAMPLE TRADES (first 10):", flush=True)
    print(f"  {'ts_entry':<20} {'sym':<8} {'strat':<10} {'side':<6} {'entry':>10} {'exit':>10} {'pnl%':>7} {'pnl$':>10} {'type':<5}", flush=True)
    print("  " + "-" * 95, flush=True)
    for t in ct[:10]:
        print(f"  {t['ts_entry'][:19]:<20} {t['symbol']:<8} {t['strategy']:<10} {t['side']:<6} "
              f"{t['entry']:10.4f} {t['exit']:10.4f} {t['pnl_pct']:+6.2f}% ${t['pnl_dollars']:+9.1f} {t['exit_type']:<5}", flush=True)

    print(f"\n  SAMPLE TRADES (last 10):", flush=True)
    for t in ct[-10:]:
        print(f"  {t['ts_entry'][:19]:<20} {t['symbol']:<8} {t['strategy']:<10} {t['side']:<6} "
              f"{t['entry']:10.4f} {t['exit']:10.4f} {t['pnl_pct']:+6.2f}% ${t['pnl_dollars']:+9.1f} {t['exit_type']:<5}", flush=True)

    print(f"\nCompleted: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
