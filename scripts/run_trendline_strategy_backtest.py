from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.data_service import get_ohlcv_with_df
from server.execution import PaperExecutionConfig
from server.strategy import StrategyConfig
from server.strategy.backtest import run_strategy_backtest, summarize_backtest_results


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "XRPUSDT")
DEFAULT_INTERVALS = ("4h", "1h")
DEFAULT_TRIGGER_MODES = ("rejection", "failed_breakout", "pre_limit")


async def _load_inputs(symbol: str, interval: str, *, days: int, history_mode: str, analysis_bars: int) -> tuple[pd.DataFrame, StrategyConfig]:
    polars_df, market_payload = await get_ohlcv_with_df(
        symbol,
        interval,
        None,
        days,
        history_mode=history_mode,
        include_price_precision=True,
        include_render_payload=False,
    )
    if polars_df is None or polars_df.is_empty():
        raise ValueError(f"No data for {symbol} {interval}")

    candles_df = polars_df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    candles_df = candles_df.rename(columns={"open_time": "timestamp"})
    candles_df["timestamp"] = candles_df["timestamp"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    for column in ("open", "high", "low", "close", "volume"):
        candles_df[column] = pd.to_numeric(candles_df[column], errors="raise")
    candles_df = candles_df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    if len(candles_df) > analysis_bars:
        candles_df = candles_df.iloc[-analysis_bars:].reset_index(drop=True)

    price_precision = market_payload.get("pricePrecision") if isinstance(market_payload, dict) else None
    strategy_config = StrategyConfig()
    if price_precision is not None:
        tick_size = 1.0 if int(price_precision) <= 0 else float(10 ** (-int(price_precision)))
        strategy_config = StrategyConfig(tick_size=tick_size)
    return candles_df, strategy_config


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the current trendline strategy through the paper-execution backtest path.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--intervals", default=",".join(DEFAULT_INTERVALS))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--analysis-bars", type=int, default=600)
    parser.add_argument("--history-mode", choices=("fast_window", "full_history"), default="full_history")
    parser.add_argument("--trigger-modes", default=",".join(DEFAULT_TRIGGER_MODES))
    parser.add_argument("--window-bars", type=int, default=None)
    parser.add_argument("--lookback-bars", type=int, default=None)
    parser.add_argument("--min-touches", type=int, default=None)
    parser.add_argument("--confirm-threshold", type=float, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--rr-target", type=float, default=None)
    parser.add_argument("--risk-per-trade", type=float, default=0.003)
    parser.add_argument("--max-concurrent-positions", type=int, default=3)
    parser.add_argument("--max-positions-per-symbol", type=int, default=1)
    parser.add_argument("--max-total-exposure", type=float, default=1.0)
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    intervals = [interval.strip() for interval in args.intervals.split(",") if interval.strip()]
    enabled_trigger_modes = tuple(mode.strip() for mode in args.trigger_modes.split(",") if mode.strip())

    execution_config = PaperExecutionConfig(
        risk_per_trade=float(args.risk_per_trade),
        max_concurrent_positions=int(args.max_concurrent_positions),
        max_positions_per_symbol=int(args.max_positions_per_symbol),
        max_total_exposure=float(args.max_total_exposure),
        starting_equity=float(args.starting_equity),
    )

    results = []
    failures = []
    for symbol in symbols:
        for interval in intervals:
            try:
                candles_df, strategy_config = await _load_inputs(
                    symbol,
                    interval,
                    days=args.days,
                    history_mode=args.history_mode,
                    analysis_bars=args.analysis_bars,
                )
                if args.lookback_bars is not None:
                    strategy_config = replace(strategy_config, lookback_bars=int(args.lookback_bars))
                if args.min_touches is not None:
                    strategy_config = replace(strategy_config, min_touches=int(args.min_touches))
                if args.confirm_threshold is not None:
                    strategy_config = replace(strategy_config, confirm_threshold=float(args.confirm_threshold))
                if args.score_threshold is not None:
                    strategy_config = replace(strategy_config, score_threshold=float(args.score_threshold))
                if args.rr_target is not None:
                    strategy_config = replace(strategy_config, rr_target=float(args.rr_target))

                result = run_strategy_backtest(
                    candles_df,
                    strategy_config,
                    execution_config,
                    symbol=symbol,
                    timeframe=interval,
                    enabled_trigger_modes=enabled_trigger_modes,
                    window_bars=args.window_bars,
                )
                results.append(result)
            except Exception as exc:
                failures.append({"symbol": symbol, "timeframe": interval, "error": str(exc)})

    summary = summarize_backtest_results(results)

    print("=== Trendline Strategy Backtest Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=True))
    if failures:
        print("=== Failed Markets ===")
        print(json.dumps(failures, indent=2, ensure_ascii=True))

    if args.json_out is not None:
        payload = {
            "summary": summary,
            "results": [result.to_dict() for result in results],
            "failures": failures,
        }
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
