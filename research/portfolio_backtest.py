"""
Portfolio-Level Backtest: Position Sizing + Risk Management
Tests different allocation strategies across both strategies × all symbols × TFs

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
# POSITION SIZING METHODS
# ═══════════════════════════════════════════════════════════════

class PositionSizer:
    """
    Methods:
    1. fixed_pct:      Fixed % of equity per trade (e.g., 2%)
    2. fixed_risk:     Risk fixed % of equity per trade, size adjusted by SL distance
    3. kelly:          Kelly criterion based on historical winrate + avg win/loss
    4. vol_adjusted:   Size inversely proportional to ATR (volatile = smaller)
    5. equal_weight:   Equal weight across all open positions (rebalance)
    """

    @staticmethod
    def fixed_pct(equity, pct=0.02, **kwargs):
        """Allocate fixed % of equity to each trade."""
        return equity * pct

    @staticmethod
    def fixed_risk(equity, risk_pct=0.01, entry=0, sl=0, **kwargs):
        """Risk exactly risk_pct of equity. Position size = risk$ / (entry-SL distance)."""
        if entry == 0 or sl == 0: return equity * 0.02
        risk_per_unit = abs(entry - sl) / entry  # % risk per unit
        if risk_per_unit == 0: return equity * 0.02
        risk_dollars = equity * risk_pct
        position_size = risk_dollars / risk_per_unit
        # Cap at max_pct of equity
        return min(position_size, equity * 0.15)

    @staticmethod
    def kelly(equity, winrate=0.5, avg_win=0.01, avg_loss=0.01, fraction=0.25, **kwargs):
        """Kelly criterion: f* = (p*b - q) / b, use fraction of full Kelly."""
        if avg_loss == 0: return equity * 0.02
        b = abs(avg_win / avg_loss)  # win/loss ratio
        p = winrate; q = 1 - p
        kelly_f = (p * b - q) / b
        kelly_f = max(0, min(kelly_f, 0.5))  # cap
        return equity * kelly_f * fraction

    @staticmethod
    def vol_adjusted(equity, base_pct=0.03, atr_pct=0.02, target_atr_pct=0.01, **kwargs):
        """Size inversely proportional to volatility. Low vol = bigger position."""
        if atr_pct == 0: return equity * base_pct
        scale = target_atr_pct / atr_pct
        scale = max(0.3, min(scale, 3.0))  # bounds
        return equity * base_pct * scale


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATOR
# ═══════════════════════════════════════════════════════════════

def simulate_portfolio(trade_streams, sizing_method, sizing_params, initial_equity=10000,
                       max_concurrent=10, max_per_symbol=1, max_drawdown_halt=0.25,
                       correlation_limit=0.5):
    """
    Simulate portfolio across multiple trade streams.

    trade_streams: list of dicts, each with:
        - trades: list of {bar, side, entry, sl, tp, exit, pnl_pct, exit_type, symbol, tf, strategy}
        - sorted by bar

    Returns portfolio equity curve and stats.
    """
    # Merge all trades into one timeline sorted by entry bar
    all_trades = []
    for stream in trade_streams:
        for t in stream:
            all_trades.append(t)
    all_trades.sort(key=lambda t: t["bar_entry"])

    equity = initial_equity
    peak = equity
    max_dd = 0
    open_positions = []  # list of active trades
    equity_curve = [equity]
    realized_returns = []
    trade_results = []
    halted = False

    for trade in all_trades:
        if halted:
            break

        # Close any positions that should be closed by now
        still_open = []
        for pos in open_positions:
            if trade["bar_entry"] >= pos["bar_exit"]:
                # Position closed
                pnl_dollars = pos["size"] * pos["pnl_pct"] / 100
                equity += pnl_dollars
                realized_returns.append(pos["pnl_pct"] / 100)
                trade_results.append({**pos, "pnl_dollars": pnl_dollars, "equity_after": equity})
            else:
                still_open.append(pos)
        open_positions = still_open

        # Check drawdown halt
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
        if dd >= max_drawdown_halt:
            halted = True
            continue

        # Check concurrent position limit
        if len(open_positions) >= max_concurrent:
            continue

        # Check per-symbol limit
        sym_count = sum(1 for p in open_positions if p["symbol"] == trade["symbol"])
        if sym_count >= max_per_symbol:
            continue

        # Calculate position size
        atr_pct = abs(trade["entry"] - trade["sl"]) / trade["entry"] if trade["entry"] > 0 else 0.02

        if sizing_method == "fixed_pct":
            size = PositionSizer.fixed_pct(equity, **sizing_params)
        elif sizing_method == "fixed_risk":
            size = PositionSizer.fixed_risk(equity, entry=trade["entry"], sl=trade["sl"], **sizing_params)
        elif sizing_method == "kelly":
            # Use running stats
            if len(realized_returns) >= 10:
                wins = [r for r in realized_returns if r > 0]
                losses = [r for r in realized_returns if r <= 0]
                wr = len(wins) / len(realized_returns) if realized_returns else 0.5
                aw = np.mean(wins) if wins else 0.01
                al = abs(np.mean(losses)) if losses else 0.01
                size = PositionSizer.kelly(equity, winrate=wr, avg_win=aw, avg_loss=al, **sizing_params)
            else:
                size = PositionSizer.fixed_pct(equity, pct=0.02)
        elif sizing_method == "vol_adjusted":
            size = PositionSizer.vol_adjusted(equity, atr_pct=atr_pct, **sizing_params)
        else:
            size = equity * 0.02

        # Don't risk more than we have
        size = min(size, equity * 0.95)
        if size <= 0:
            continue

        # Open position
        open_positions.append({
            **trade,
            "size": size,
        })
        equity_curve.append(equity)

    # Close remaining positions
    for pos in open_positions:
        pnl_dollars = pos["size"] * pos["pnl_pct"] / 100
        equity += pnl_dollars
        realized_returns.append(pos["pnl_pct"] / 100)
        trade_results.append({**pos, "pnl_dollars": pnl_dollars, "equity_after": equity})

    peak = max(peak, equity)
    max_dd = max(max_dd, (peak - equity) / peak)

    if len(realized_returns) < 2:
        return {"net_pct": 0, "sharpe": 0, "trades": 0, "max_dd": 0, "halted": halted,
                "final_equity": equity}

    r = np.array(realized_returns)
    sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(len(r))

    return {
        "net_pct": (equity / initial_equity - 1) * 100,
        "sharpe": sharpe,
        "winrate": np.mean(r > 0) * 100,
        "trades": len(realized_returns),
        "max_dd": max_dd * 100,
        "final_equity": equity,
        "halted": halted,
        "avg_concurrent": np.mean([len([p for p in open_positions]) for _ in range(1)]),  # simplified
        "avg_win_pct": np.mean(r[r > 0]) * 100 if np.any(r > 0) else 0,
        "avg_loss_pct": np.mean(r[r <= 0]) * 100 if np.any(r <= 0) else 0,
    }


# ═══════════════════════════════════════════════════════════════
# COLLECT TRADE STREAMS
# ═══════════════════════════════════════════════════════════════

def collect_ribbon_trades(symbols, tfs):
    """Run ribbon strategy on all symbols/TFs and collect trade logs."""
    s = RibbonStrategy()
    trades = []
    for sym in symbols:
        for tf in tfs:
            d = load_ohlcv(sym, tf)
            if d is None: continue
            bt = s.backtest(d["o"], d["h"], d["l"], d["c"], d["v"])
            for t in bt["trade_log"]:
                trades.append({
                    "bar_entry": t["bar"], "bar_exit": t["bar"] + 1,  # simplified: 1-bar hold for timeline
                    "side": 1 if t["side"] == 1 else -1,
                    "entry": t["entry"], "exit": t["exit"],
                    "sl": t["entry"] * 0.97 if t["side"] == 1 else t["entry"] * 1.03,  # approx
                    "tp": t["exit"] if t["type"] == "TP" else t["entry"],
                    "pnl_pct": t["pnl_pct"], "exit_type": t["type"],
                    "symbol": sym, "tf": tf, "strategy": "ribbon",
                })
    return trades


def collect_trendline_trades(symbols, tfs):
    """Run trendline strategy on all symbols/TFs and collect trade logs."""
    cfg = {**TL_DEFAULT,
           "swing_lookback": 20, "buffer_pct": 0.25, "sl_pct": 0.4, "rr": 2.0,
           "max_hold_bars": 100, "approach_pct": 2.0, "min_bars_between": 40,
           "max_bars_between": 1000, "max_penetrations": 2}
    trades = []
    for sym in symbols:
        for tf in tfs:
            d = load_ohlcv(sym, tf)
            if d is None: continue
            bt = trendline_bt(d["o"], d["h"], d["l"], d["c"], d["v"], cfg)
            for t in bt["trade_log"]:
                trades.append({
                    "bar_entry": t["bar_entry"], "bar_exit": t["bar_exit"],
                    "side": 1 if t["side"] == "LONG" else -1,
                    "entry": t["entry"], "exit": t["exit"],
                    "sl": t["sl"], "tp": t["tp"],
                    "pnl_pct": t["pnl_pct"], "exit_type": t["exit_type"],
                    "symbol": sym, "tf": tf, "strategy": "trendline",
                })
    return trades


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "dogeusdt", "nearusdt",
           "adausdt", "bnbusdt", "linkusdt", "pepeusdt", "suiusdt", "xrpusdt"]
TFS = ["1h", "4h"]

SIZING_CONFIGS = [
    ("fixed_2pct",    "fixed_pct",   {"pct": 0.02}),
    ("fixed_5pct",    "fixed_pct",   {"pct": 0.05}),
    ("fixed_10pct",   "fixed_pct",   {"pct": 0.10}),
    ("risk_1pct",     "fixed_risk",  {"risk_pct": 0.01}),
    ("risk_2pct",     "fixed_risk",  {"risk_pct": 0.02}),
    ("kelly_quarter", "kelly",       {"fraction": 0.25}),
    ("kelly_tenth",   "kelly",       {"fraction": 0.10}),
    ("vol_adj",       "vol_adjusted",{"base_pct": 0.03, "target_atr_pct": 0.01}),
]

CONCURRENT_LIMITS = [3, 5, 10, 20]
DD_HALTS = [0.15, 0.25, 0.50]


def main():
    print("=" * 100)
    print("PORTFOLIO-LEVEL BACKTEST: POSITION SIZING + RISK MANAGEMENT")
    print(f"  {len(SYMBOLS)} symbols x {len(TFS)} TFs x 2 strategies")
    print("=" * 100)

    # Collect trades
    print("\n[1/3] Collecting MA Ribbon trades...")
    ribbon_trades = collect_ribbon_trades(SYMBOLS, TFS)
    print(f"  {len(ribbon_trades)} ribbon trades")

    print("[2/3] Collecting Trendline trades...")
    tl_trades = collect_trendline_trades(SYMBOLS, TFS)
    print(f"  {len(tl_trades)} trendline trades")

    all_trades = ribbon_trades + tl_trades
    ribbon_only = ribbon_trades
    tl_only = tl_trades

    # ═══════════════════════════════════════════════════════════
    # Test 1: Sizing method comparison (fixed concurrent=5, dd_halt=25%)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TEST 1: SIZING METHOD COMPARISON (concurrent=5, dd_halt=25%)")
    print("=" * 100)
    print(f"{'sizing':<16} {'strategy':<12} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>10} {'DD%':>6} {'final$':>10} {'halted':>7}")
    print("-" * 85)

    for label, method, params in SIZING_CONFIGS:
        for strat_name, strat_trades in [("both", all_trades), ("ribbon", ribbon_only), ("trendline", tl_only)]:
            r = simulate_portfolio(
                [strat_trades], method, params,
                initial_equity=10000, max_concurrent=5, max_drawdown_halt=0.25,
            )
            halt_str = "YES" if r["halted"] else ""
            print(f"{label:<16} {strat_name:<12} {r['trades']:6d} {r['winrate']:5.1f}% "
                  f"{r['sharpe']:+6.2f} {r['net_pct']:+9.1f}% {r['max_dd']:5.1f}% "
                  f"{r['final_equity']:10.0f} {halt_str:>7}")

    # ═══════════════════════════════════════════════════════════
    # Test 2: Concurrent position limit sweep
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TEST 2: CONCURRENT POSITION LIMIT (risk_1pct sizing, dd_halt=25%)")
    print("=" * 100)
    print(f"{'max_concurrent':>14} {'strategy':<12} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>10} {'DD%':>6} {'final$':>10}")
    print("-" * 75)

    for mc in CONCURRENT_LIMITS:
        for sn, st in [("both", all_trades), ("ribbon", ribbon_only), ("trendline", tl_only)]:
            r = simulate_portfolio(
                [st], "fixed_risk", {"risk_pct": 0.01},
                initial_equity=10000, max_concurrent=mc, max_drawdown_halt=0.25,
            )
            print(f"{mc:14d} {sn:<12} {r['trades']:6d} {r['winrate']:5.1f}% "
                  f"{r['sharpe']:+6.2f} {r['net_pct']:+9.1f}% {r['max_dd']:5.1f}% {r['final_equity']:10.0f}")

    # ═══════════════════════════════════════════════════════════
    # Test 3: Drawdown halt threshold
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TEST 3: DRAWDOWN HALT THRESHOLD (risk_1pct, concurrent=5)")
    print("=" * 100)
    print(f"{'dd_halt':>8} {'strategy':<12} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>10} {'DD%':>6} {'halted':>7}")
    print("-" * 65)

    for dd in DD_HALTS:
        for sn, st in [("both", all_trades), ("ribbon", ribbon_only)]:
            r = simulate_portfolio(
                [st], "fixed_risk", {"risk_pct": 0.01},
                initial_equity=10000, max_concurrent=5, max_drawdown_halt=dd,
            )
            halt_str = "YES" if r["halted"] else ""
            print(f"{dd*100:7.0f}% {sn:<12} {r['trades']:6d} {r['winrate']:5.1f}% "
                  f"{r['sharpe']:+6.2f} {r['net_pct']:+9.1f}% {r['max_dd']:5.1f}% {halt_str:>7}")

    # ═══════════════════════════════════════════════════════════
    # Test 4: Strategy allocation split
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("TEST 4: STRATEGY ALLOCATION (risk_1pct, concurrent=5, dd_halt=25%)")
    print("=" * 100)
    print(f"{'allocation':<20} {'trades':>6} {'win%':>6} {'sharpe':>7} {'net%':>10} {'DD%':>6} {'final$':>10}")
    print("-" * 70)

    configs = [
        ("100% ribbon",    1.0, 0.0),
        ("100% trendline", 0.0, 1.0),
        ("70/30 rib/tl",   0.7, 0.3),
        ("50/50",          0.5, 0.5),
        ("30/70 rib/tl",   0.3, 0.7),
    ]

    for label, rib_w, tl_w in configs:
        # Simulate by adjusting position sizes
        mixed = []
        for t in ribbon_trades:
            mixed.append({**t, "_weight": rib_w})
        for t in tl_trades:
            mixed.append({**t, "_weight": tl_w})
        mixed.sort(key=lambda x: x["bar_entry"])

        # Custom simulate with weights
        equity = 10000.0; peak = 10000.0; max_dd = 0; returns = []; halted = False
        open_pos = []; wins = 0; trades_done = 0

        for trade in mixed:
            if halted: break
            # Close expired
            still = []
            for p in open_pos:
                if trade["bar_entry"] >= p["bar_exit"]:
                    pnl = p["size"] * p["pnl_pct"] / 100
                    equity += pnl; returns.append(p["pnl_pct"] / 100)
                    if p["pnl_pct"] > 0: wins += 1
                    trades_done += 1
                else:
                    still.append(p)
            open_pos = still

            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
            if dd >= 0.25: halted = True; continue
            if len(open_pos) >= 5: continue

            w = trade.get("_weight", 1.0)
            risk_pct = 0.01 * w
            if trade["entry"] > 0 and trade["sl"] > 0:
                risk_per_unit = abs(trade["entry"] - trade["sl"]) / trade["entry"]
                if risk_per_unit > 0:
                    size = min(equity * risk_pct / risk_per_unit, equity * 0.15)
                else:
                    size = equity * 0.02 * w
            else:
                size = equity * 0.02 * w
            if size <= 0: continue

            open_pos.append({**trade, "size": size})

        for p in open_pos:
            pnl = p["size"] * p["pnl_pct"] / 100
            equity += pnl; returns.append(p["pnl_pct"] / 100)
            if p["pnl_pct"] > 0: wins += 1
            trades_done += 1

        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

        if trades_done >= 2:
            r = np.array(returns)
            sharpe = np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(len(r))
            wr = wins / trades_done * 100
        else:
            sharpe = 0; wr = 0

        print(f"{label:<20} {trades_done:6d} {wr:5.1f}% {sharpe:+6.2f} "
              f"{(equity/10000-1)*100:+9.1f}% {max_dd*100:5.1f}% {equity:10.0f}")

    # ═══════════════════════════════════════════════════════════
    # RECOMMENDATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("RECOMMENDED PRODUCTION CONFIG")
    print("=" * 100)
    print("""
  Position Sizing:  fixed_risk (1% equity risk per trade)
                    Size = (equity × 0.01) / (entry - SL distance as %)
                    Cap: max 15% equity per single position

  Risk Limits:      max 5 concurrent positions
                    max 1 position per symbol
                    drawdown halt at 25% (stop all new trades)

  Strategy Mix:     Start with 100% MA Ribbon (higher winrate, more predictable)
                    Add trendline after 50+ live ribbon trades confirm edge

  Per-Trade Rules:  Never risk > 1% equity per trade
                    SL is mandatory and pre-calculated
                    No manual override of SL
    """)

    print(f"Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
