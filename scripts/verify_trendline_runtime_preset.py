from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.data_service import get_ohlcv_with_df
from server.execution import PaperExecutionConfig
from server.strategy import StrategyConfig
from server.strategy.backtest import run_strategy_backtest, summarize_backtest_results


VALIDATED_SYMBOLS = ("HYPEUSDT", "RIVERUSDT", "ENSOUSDT")
VALIDATED_INTERVAL = "1h"
VALIDATED_ANALYSIS_BARS = 200
VALIDATED_WINDOW_BARS = 100
VALIDATED_LOOKBACK_BARS = 80
VALIDATED_TRIGGER_MODES = ("pre_limit",)


async def _load_market(symbol: str, interval: str, *, bars: int) -> tuple[pd.DataFrame, StrategyConfig]:
    df, payload = await get_ohlcv_with_df(
        symbol,
        interval,
        None,
        365,
        history_mode="full_history",
        include_price_precision=True,
        include_render_payload=False,
    )
    if df is None or df.is_empty():
        raise ValueError(f"No local data for {symbol} {interval}")

    pdf = df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    pdf["timestamp"] = pdf["timestamp"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    for column in ("open", "high", "low", "close", "volume"):
        pdf[column] = pd.to_numeric(pdf[column], errors="raise")
    pdf = pdf[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True).iloc[-bars:].reset_index(drop=True)

    price_precision = payload.get("pricePrecision") if isinstance(payload, dict) else None
    tick_size = 1.0 if price_precision is None or int(price_precision) <= 0 else float(10 ** (-int(price_precision)))
    return pdf, replace(StrategyConfig(tick_size=tick_size), lookback_bars=VALIDATED_LOOKBACK_BARS)


async def main() -> int:
    execution_config = PaperExecutionConfig(starting_equity=10_000.0)
    results = []

    for symbol in VALIDATED_SYMBOLS:
        candles_df, strategy_config = await _load_market(symbol, VALIDATED_INTERVAL, bars=VALIDATED_ANALYSIS_BARS)
        result = run_strategy_backtest(
            candles_df,
            strategy_config,
            execution_config,
            symbol=symbol,
            timeframe=VALIDATED_INTERVAL,
            enabled_trigger_modes=VALIDATED_TRIGGER_MODES,
            window_bars=VALIDATED_WINDOW_BARS,
        )
        results.append(result)

    summary = summarize_backtest_results(results)
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=True))

    is_valid = (
        summary["trade_count"] >= 6
        and summary["total_pnl"] > 0
        and (summary["profit_factor"] or 0) > 1.5
    )
    return 0 if is_valid else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
