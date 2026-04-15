"""
Daily Report Generator
======================
Generates structured JSON + human-readable report for AI review/learning.
Covers: trades of the day, strategy performance, risk metrics, anomalies.

Run daily: python -m research.daily_report
Output: data/reports/daily/YYYY-MM-DD.json + .txt
"""

import gzip, csv, json, sys, os, glob, numpy as np
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.strategy import Strategy as RibbonStrategy
from research.trendline_strategy import backtest as trendline_bt, DEFAULT_CONFIG as TL_DEFAULT


def load_ohlcv(symbol, tf):
    fn = f"data/{symbol}_{tf}.csv.gz"
    if not os.path.exists(fn): return None
    with gzip.open(fn, "rt") as f:
        reader = csv.DictReader(f); rows = list(reader)
    if len(rows) < 200: return None
    return {
        "o": np.array([float(r["open"]) for r in rows]),
        "h": np.array([float(r["high"]) for r in rows]),
        "l": np.array([float(r["low"]) for r in rows]),
        "c": np.array([float(r["close"]) for r in rows]),
        "v": np.array([float(r["volume"]) for r in rows]),
        "ts": [r["open_time"] for r in rows],
    }


SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "dogeusdt", "nearusdt",
           "adausdt", "bnbusdt", "linkusdt", "pepeusdt", "suiusdt", "xrpusdt",
           "hypeusdt", "avaxusdt", "dotusdt", "shibusdt", "wifusdt"]
TFS = ["1h", "4h"]


def generate_report(target_date=None):
    """
    Generate daily report for target_date (default: today).
    Returns dict with all metrics + human-readable text.
    """
    if target_date is None:
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    report = {
        "date": target_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategies": {},
        "signals_today": [],
        "market_state": [],
        "cumulative": {},
        "anomalies": [],
        "recommendations": [],
    }

    ribbon = RibbonStrategy()
    tl_cfg = {**TL_DEFAULT,
              "swing_lookback": 20, "buffer_pct": 0.25, "sl_pct": 0.4, "rr": 2.0,
              "max_hold_bars": 100, "approach_pct": 2.0, "min_bars_between": 40,
              "max_bars_between": 1000, "max_penetrations": 2}

    # ── Per-strategy, per-symbol, per-TF stats ──
    ribbon_all_trades = []
    tl_all_trades = []

    for sym in SYMBOLS:
        for tf in TFS:
            d = load_ohlcv(sym, tf)
            if d is None:
                continue

            ts_arr = d["ts"]

            # ── Ribbon ──
            bt_rib = ribbon.backtest(d["o"], d["h"], d["l"], d["c"], d["v"])
            for t in bt_rib["trade_log"]:
                bar_i = t["bar"]
                if bar_i < 0 or bar_i >= len(ts_arr):
                    continue
                trade_date = ts_arr[bar_i][:10]
                trade_entry = {
                    "date": trade_date,
                    "ts": ts_arr[bar_i],
                    "symbol": sym.upper(),
                    "tf": tf,
                    "strategy": "ribbon",
                    "side": "LONG" if t["side"] == 1 else "SHORT",
                    "entry": float(t["entry"]),
                    "exit": float(t["exit"]),
                    "pnl_pct": float(t["pnl_pct"]),
                    "exit_type": t["type"],
                }
                ribbon_all_trades.append(trade_entry)
                if trade_date == target_date:
                    report["signals_today"].append(trade_entry)

            # ── Ribbon current state ──
            state = ribbon.current_state(d["o"], d["h"], d["l"], d["c"], d["v"])
            report["market_state"].append({
                "symbol": sym.upper(), "tf": tf,
                "bull_ribbon": state["bull_ribbon"],
                "bear_ribbon": state["bear_ribbon"],
                "bull_crossover": state.get("bull_crossover", False),
                "bear_crossover": state.get("bear_crossover", False),
                "adx": state.get("adx"),
                "fanning": state.get("fanning"),
                "signal": state.get("signal", 0),
                "close": state.get("close"),
                "bull_conditions": state.get("bull_conditions_met", 0),
                "bear_conditions": state.get("bear_conditions_met", 0),
            })

            # ── Trendline (4h only) ──
            if tf == "4h":
                bt_tl = trendline_bt(d["o"], d["h"], d["l"], d["c"], d["v"], tl_cfg)
                for t in bt_tl["trade_log"]:
                    be = t["bar_entry"]
                    if be < 0 or be >= len(ts_arr):
                        continue
                    trade_date = ts_arr[be][:10]
                    trade_entry = {
                        "date": trade_date,
                        "ts": ts_arr[be],
                        "symbol": sym.upper(),
                        "tf": tf,
                        "strategy": "trendline",
                        "side": t["side"],
                        "entry": float(t["entry"]),
                        "exit": float(t["exit"]),
                        "pnl_pct": float(t["pnl_pct"]),
                        "exit_type": t["exit_type"],
                    }
                    tl_all_trades.append(trade_entry)
                    if trade_date == target_date:
                        report["signals_today"].append(trade_entry)

    # ── Strategy summary (all time) ──
    for sname, trades in [("ribbon", ribbon_all_trades), ("trendline", tl_all_trades)]:
        if not trades:
            continue
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        report["strategies"][sname] = {
            "total_trades": len(trades),
            "winrate": len(wins) / len(trades) * 100,
            "avg_win_pct": np.mean(wins) if wins else 0,
            "avg_loss_pct": np.mean(losses) if losses else 0,
            "expectancy": np.mean(pnls),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "sharpe": np.mean(pnls) / (np.std(pnls) + 1e-12) * np.sqrt(len(pnls)),
        }

        # Last 7 days performance
        recent = [t for t in trades if t["date"] >= (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")]
        if recent:
            r_pnls = [t["pnl_pct"] for t in recent]
            r_wins = [p for p in r_pnls if p > 0]
            report["strategies"][sname]["last_7d"] = {
                "trades": len(recent),
                "winrate": len(r_wins) / len(recent) * 100 if recent else 0,
                "net_pnl_pct": sum(r_pnls),
                "avg_pnl_pct": np.mean(r_pnls),
            }

        # Per-symbol breakdown
        sym_perf = {}
        for t in trades:
            if t["symbol"] not in sym_perf:
                sym_perf[t["symbol"]] = {"trades": 0, "wins": 0, "pnl_sum": 0}
            sym_perf[t["symbol"]]["trades"] += 1
            sym_perf[t["symbol"]]["pnl_sum"] += t["pnl_pct"]
            if t["pnl_pct"] > 0:
                sym_perf[t["symbol"]]["wins"] += 1

        report["strategies"][sname]["by_symbol"] = {
            sym: {
                "trades": d["trades"],
                "winrate": d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0,
                "net_pnl_pct": d["pnl_sum"],
            }
            for sym, d in sym_perf.items()
        }

    # ── Today's summary ──
    today_trades = report["signals_today"]
    if today_trades:
        t_pnls = [t["pnl_pct"] for t in today_trades]
        t_wins = [p for p in t_pnls if p > 0]
        report["today_summary"] = {
            "trades": len(today_trades),
            "wins": len(t_wins),
            "losses": len(today_trades) - len(t_wins),
            "net_pnl_pct": sum(t_pnls),
            "best": max(t_pnls),
            "worst": min(t_pnls),
            "by_strategy": {},
        }
        for sn in set(t["strategy"] for t in today_trades):
            st = [t for t in today_trades if t["strategy"] == sn]
            sp = [t["pnl_pct"] for t in st]
            report["today_summary"]["by_strategy"][sn] = {
                "trades": len(st),
                "net_pnl_pct": sum(sp),
                "winrate": sum(1 for p in sp if p > 0) / len(sp) * 100,
            }
    else:
        report["today_summary"] = {"trades": 0, "note": "no trades today"}

    # ── Anomaly detection ──
    for sname, trades in [("ribbon", ribbon_all_trades), ("trendline", tl_all_trades)]:
        if not trades:
            continue
        pnls = [t["pnl_pct"] for t in trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)

        # Check for streaks
        recent_20 = trades[-20:]
        if len(recent_20) >= 10:
            recent_wr = sum(1 for t in recent_20 if t["pnl_pct"] > 0) / len(recent_20)
            all_wr = sum(1 for p in pnls if p > 0) / len(pnls)

            if recent_wr < all_wr - 0.15:
                report["anomalies"].append({
                    "type": "winrate_drop",
                    "strategy": sname,
                    "message": f"{sname}: recent 20 trades winrate {recent_wr:.1%} vs historical {all_wr:.1%} (>{15}pp drop)",
                    "severity": "warning",
                })

            # Check for consecutive losses
            streak = 0; max_streak = 0
            for t in recent_20:
                if t["pnl_pct"] <= 0:
                    streak += 1; max_streak = max(max_streak, streak)
                else:
                    streak = 0
            if max_streak >= 5:
                report["anomalies"].append({
                    "type": "loss_streak",
                    "strategy": sname,
                    "message": f"{sname}: {max_streak} consecutive losses in last 20 trades",
                    "severity": "warning",
                })

        # Outlier trades today
        for t in today_trades:
            if t["strategy"] == sname and abs(t["pnl_pct"]) > abs(mean_pnl) + 3 * std_pnl:
                report["anomalies"].append({
                    "type": "outlier_trade",
                    "strategy": sname,
                    "message": f"{sname} outlier: {t['symbol']} {t['side']} pnl={t['pnl_pct']:+.2f}% (3σ = {3*std_pnl:.2f}%)",
                    "severity": "info",
                })

    # ── Recommendations ──
    if not report["anomalies"]:
        report["recommendations"].append("All systems nominal. No parameter changes needed.")
    else:
        for a in report["anomalies"]:
            if a["type"] == "winrate_drop":
                report["recommendations"].append(
                    f"Review {a['strategy']} signals. Consider pausing if drop persists >3 days.")
            elif a["type"] == "loss_streak":
                report["recommendations"].append(
                    f"Reduce {a['strategy']} position size by 50% until streak breaks.")

    return report


def render_text(report):
    """Render report as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"DAILY STRATEGY REPORT — {report['date']}")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("=" * 80)

    # Today summary
    ts = report.get("today_summary", {})
    lines.append(f"\nTODAY: {ts.get('trades', 0)} trades | "
                 f"wins={ts.get('wins', 0)} losses={ts.get('losses', 0)} | "
                 f"net={ts.get('net_pnl_pct', 0):+.2f}%")
    if ts.get("by_strategy"):
        for sn, ss in ts["by_strategy"].items():
            lines.append(f"  {sn}: {ss['trades']} trades, net {ss['net_pnl_pct']:+.2f}%, win {ss['winrate']:.0f}%")

    # Strategy overview
    lines.append(f"\nSTRATEGY OVERVIEW (all time):")
    for sn, ss in report.get("strategies", {}).items():
        lines.append(f"\n  {sn.upper()}")
        lines.append(f"    Total: {ss['total_trades']} trades | win {ss['winrate']:.1f}% | "
                     f"expect {ss['expectancy']:+.3f}% | sharpe {ss['sharpe']:+.2f}")
        lines.append(f"    Avg win: {ss['avg_win_pct']:+.2f}% | Avg loss: {ss['avg_loss_pct']:+.2f}% | "
                     f"Best: {ss['best_trade']:+.2f}% | Worst: {ss['worst_trade']:+.2f}%")
        if ss.get("last_7d"):
            l7 = ss["last_7d"]
            lines.append(f"    Last 7d: {l7['trades']} trades, net {l7['net_pnl_pct']:+.2f}%, "
                         f"win {l7['winrate']:.0f}%, avg {l7['avg_pnl_pct']:+.3f}%")

        # Top/bottom symbols
        if ss.get("by_symbol"):
            sorted_syms = sorted(ss["by_symbol"].items(), key=lambda x: x[1]["net_pnl_pct"], reverse=True)
            top3 = ", ".join(f"{s}({d['net_pnl_pct']:+.1f}%)" for s, d in sorted_syms[:3])
            bot3 = ", ".join(f"{s}({d['net_pnl_pct']:+.1f}%)" for s, d in sorted_syms[-3:])
            lines.append(f"    Top 3:    {top3}")
            lines.append(f"    Bottom 3: {bot3}")

    # Anomalies
    if report["anomalies"]:
        lines.append(f"\nANOMALIES ({len(report['anomalies'])}):")
        for a in report["anomalies"]:
            lines.append(f"  [{a['severity'].upper()}] {a['message']}")

    # Recommendations
    lines.append(f"\nRECOMMENDATIONS:")
    for r in report["recommendations"]:
        lines.append(f"  - {r}")

    # Current signals
    active = [s for s in report["market_state"] if s["signal"] != 0]
    approaching = [s for s in report["market_state"]
                   if s.get("bull_conditions", 0) == 3 or s.get("bear_conditions", 0) == 3]

    lines.append(f"\nMARKET STATE:")
    lines.append(f"  Active signals: {len(active)}")
    lines.append(f"  Approaching (3/4 conditions): {len(approaching)}")
    for s in approaching[:10]:
        direction = "BULL" if s.get("bull_conditions", 0) == 3 else "BEAR"
        adx_str = f"{s['adx']:.1f}" if s.get('adx') is not None else "N/A"
        lines.append(f"    {s['symbol']:<10} {s['tf']:<4} {direction:<5} ADX={adx_str}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    target = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Generating report for {target}...", flush=True)

    report = generate_report(target)

    # Save JSON
    out_dir = "data/reports/daily"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{target}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"  JSON: {json_path}", flush=True)

    # Save text
    txt = render_text(report)
    txt_path = os.path.join(out_dir, f"{target}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"  Text: {txt_path}", flush=True)

    # Print to stdout
    print(txt, flush=True)


if __name__ == "__main__":
    main()
