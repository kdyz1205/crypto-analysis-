"""
Autoresearch: Autonomous strategy optimization for crypto-analysis.

Runs the MFI/MA backtest across multiple symbols and timeframes,
computes a composite score, and prints results in a machine-readable format.

Usage:
    python autoresearch/autoresearch.py
    python autoresearch/autoresearch.py --symbol BTCUSDT --interval 4h
    python autoresearch/autoresearch.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""

import sys
import os
import time
import math
import argparse
import asyncio

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np


# ── Scoring (FIXED — do not modify) ──────────────────────────────────────────

def compute_score(result: dict) -> float:
    """
    Composite risk-adjusted score (HIGHER is better).
    score = sharpe × √(trade_factor) − drawdown_penalty
    """
    trades = result.get("trades", [])
    num_trades = len(trades)

    # Hard cutoffs
    if num_trades < 3:
        return -999.0
    max_dd = result.get("max_drawdown_pct", 0)
    if max_dd > 50.0:
        return -999.0

    sharpe = result.get("sharpe", 0)
    trade_factor = min(num_trades / 20.0, 1.0)
    dd_penalty = max(0, max_dd - 10.0) * 0.05

    return sharpe * math.sqrt(trade_factor) - dd_penalty


# ── Data loading (uses OKX API async) ────────────────────────────────────────

async def load_data(symbol: str, interval: str, days: int = 365):
    """Load OHLCV data via the existing data service."""
    from server.data_service import get_ohlcv_with_df
    df, _ = await get_ohlcv_with_df(symbol, interval, days=days)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_experiment(symbol: str, interval: str, days: int = 365):
    """Run a single backtest experiment and return results."""
    from server.backtest_service import run_backtest, BacktestParams

    df = await load_data(symbol, interval, days)
    if df is None or df.is_empty():
        return None

    params = BacktestParams()  # uses current defaults from backtest_service.py
    result = run_backtest(df, params)
    result["symbol"] = symbol
    result["interval"] = interval
    result["score"] = compute_score(result)
    return result


async def run_multi(symbols: list[str], intervals: list[str], days: int = 365):
    """Run backtest across multiple symbols/intervals, return aggregate score."""
    all_results = []
    for symbol in symbols:
        for interval in intervals:
            try:
                result = await run_experiment(symbol, interval, days)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"ERROR: {symbol} {interval}: {e}", file=sys.stderr)

    if not all_results:
        print("score:              -999.000000")
        print("total_trades:       0")
        print("sharpe:             0.000000")
        return {"score": -999.0, "results": []}

    # Aggregate: average score across all symbol/interval combos
    scores = [r["score"] for r in all_results]
    avg_score = np.mean(scores)
    total_trades = sum(r["total_trades"] for r in all_results)
    avg_sharpe = np.mean([r.get("sharpe", 0) for r in all_results])
    avg_pnl = np.mean([r["total_pnl_pct"] for r in all_results])
    avg_wr = np.mean([r["win_rate"] for r in all_results])
    avg_dd = np.mean([r.get("max_drawdown_pct", 0) for r in all_results])
    avg_pf = np.mean([r.get("profit_factor", 0) for r in all_results])

    return {
        "score": float(avg_score),
        "total_trades": total_trades,
        "sharpe": float(avg_sharpe),
        "total_pnl_pct": float(avg_pnl),
        "win_rate": float(avg_wr),
        "max_drawdown_pct": float(avg_dd),
        "profit_factor": float(avg_pf),
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Autoresearch: crypto strategy optimizer")
    parser.add_argument("--symbol", default="BTCUSDT", help="Single symbol (default: BTCUSDT)")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT,SOLUSDT)")
    parser.add_argument("--interval", default="4h", help="Single interval (default: 4h)")
    parser.add_argument("--intervals", default=None, help="Comma-separated intervals (e.g. 1h,4h)")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data (default: 365)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else [args.symbol]
    intervals = args.intervals.split(",") if args.intervals else [args.interval]

    t_start = time.time()

    agg = asyncio.run(run_multi(symbols, intervals, args.days))

    t_end = time.time()

    # Print results in machine-readable format (same as Karpathy's autoresearch)
    print("---")
    print(f"score:              {agg['score']:.6f}")
    print(f"sharpe:             {agg.get('sharpe', 0):.6f}")
    print(f"total_pnl_pct:      {agg.get('total_pnl_pct', 0):.6f}")
    print(f"win_rate:           {agg.get('win_rate', 0):.6f}")
    print(f"max_drawdown_pct:   {agg.get('max_drawdown_pct', 0):.6f}")
    print(f"profit_factor:      {agg.get('profit_factor', 0):.6f}")
    print(f"total_trades:       {agg.get('total_trades', 0)}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"symbols:            {','.join(symbols)}")
    print(f"intervals:          {','.join(intervals)}")

    # Per-symbol breakdown
    for r in agg.get("results", []):
        print(f"  {r['symbol']} {r['interval']}: score={r['score']:.4f} sharpe={r.get('sharpe', 0):.4f} "
              f"pnl={r['total_pnl_pct']:.2f}% trades={r['total_trades']} wr={r['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
