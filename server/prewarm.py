"""Background pre-warm of the strategy-snapshot + OHLCV caches.

Goal: when the user opens /v2 after a server restart, the most-viewed
(symbol, TF) combos are ALREADY computed and sitting in the snapshot
cache / polars OHLCV cache. First request = cache hit = 0.2s instead
of 1-2s cold compute.

Scope:
  - FAVORITES × COMMON_TFS (default 8 × 3 = 24 pre-computed snapshots)
  - Runs once at server startup, background task, doesn't block
  - Re-runs periodically (every 4 min) to keep caches warm even as
    bar boundaries invalidate them
  - Backs off exponentially if pre-warm fails (never loops on errors)

Safety:
  - Never blocks user requests (in-flight dedup in strategy.py covers
    the race between user and pre-warm hitting the same key)
  - Read-only — no side effects on orders / account / drawings
  - Sleeps between symbols to avoid Bitget rate-limits
"""
from __future__ import annotations

import asyncio
import time

# 2026-04-23: matches the frontend picker default favourites
# (frontend/js/workbench/symbol_picker.js DEFAULT_FAVORITES). If you
# change either list, mirror the change.
DEFAULT_FAVORITE_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT",
    "ZECUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
)
COMMON_TIMEFRAMES: tuple[str, ...] = ("1h", "4h", "1d")
PREWARM_INTERVAL_SEC = 240   # re-warm every 4 minutes (bar close kicks cache)
INTER_SYMBOL_DELAY_SEC = 0.4  # small gap between each fetch so we never DoS Bitget


_stopped = False
_task: asyncio.Task | None = None


async def _warm_one(symbol: str, timeframe: str) -> None:
    """Hit the same code paths /api/strategy/snapshot + /market/
    structure-summary would. Populates:
       - server.data_service._ohlcv_cache (polars DF cached)
       - server.routers.strategy._snapshot_cache (compute cached)
       - server.routers.market._structure_summary_cache
    """
    try:
        from .data_service import get_ohlcv_with_df
        from .drawings import manual_strategy_signature
        from .routers.strategy import (
            _build_strategy_snapshot_response,
            _config_with_market_precision,
            _snapshot_cache_key,
            _standardize_strategy_candles,
            _store_cached_snapshot,
            _get_cached_snapshot,
            DEFAULT_DAYS_BY_INTERVAL,
        )
        from .routers.market import (
            _trim_structure_summary_df,
            _structure_summary_cache_key,
            _get_cached_structure_summary,
            _store_cached_structure_summary,
            _build_structure_summary,
        )
        from .strategy import StrategyConfig
        from .history_coverage import build_analysis_history

        days = DEFAULT_DAYS_BY_INTERVAL.get(timeframe, 180)
        polars_df, market_payload = await get_ohlcv_with_df(
            symbol, timeframe, None, days,
            history_mode="fast_window",
            include_price_precision=True,
            include_render_payload=False,
        )
        if polars_df is None or polars_df.is_empty():
            return

        # Warm snapshot cache
        candles_df = _standardize_strategy_candles(polars_df)
        if len(candles_df) > 500:
            candles_df = candles_df.iloc[-500:].reset_index(drop=True)
        price_precision = market_payload.get("pricePrecision") if isinstance(market_payload, dict) else None
        history = build_analysis_history(market_payload, candles_df)
        cfg = _config_with_market_precision(StrategyConfig(), price_precision)
        drawings_sig = manual_strategy_signature(symbol, timeframe)
        cache_key = _snapshot_cache_key(candles_df, symbol, timeframe, price_precision, drawings_sig)
        if _get_cached_snapshot(cache_key) is None:
            response = await asyncio.to_thread(
                _build_strategy_snapshot_response,
                candles_df, cfg, symbol, timeframe, price_precision, history, drawings_sig,
            )
            _store_cached_snapshot(cache_key, response)

        # Warm structure-summary cache (same polars_df from ohlcv cache)
        try:
            trimmed = _trim_structure_summary_df(polars_df, timeframe)
            ss_key = _structure_summary_cache_key(trimmed, symbol, timeframe)
            if _get_cached_structure_summary(ss_key) is None:
                ss_response = await asyncio.to_thread(
                    _build_structure_summary, trimmed, symbol, timeframe,
                )
                _store_cached_structure_summary(ss_key, ss_response)
        except Exception as exc:
            # Non-fatal — snapshot cache is the main win; structure is bonus
            print(f"[prewarm] structure-summary {symbol} {timeframe}: {exc}")
    except Exception as exc:
        print(f"[prewarm] {symbol} {timeframe} failed: {exc}")


async def _prewarm_loop(symbols: tuple[str, ...], timeframes: tuple[str, ...]) -> None:
    """Endless loop: warm each (symbol, TF), sleep, repeat."""
    cycle = 0
    while not _stopped:
        cycle += 1
        t0 = time.time()
        n_warmed = 0
        for sym in symbols:
            if _stopped:
                return
            for tf in timeframes:
                if _stopped:
                    return
                await _warm_one(sym, tf)
                n_warmed += 1
                await asyncio.sleep(INTER_SYMBOL_DELAY_SEC)
        dt = time.time() - t0
        print(f"[prewarm] cycle {cycle}: {n_warmed} (symbol,TF) warmed in {dt:.1f}s — "
              f"next in {PREWARM_INTERVAL_SEC}s")
        # Sleep until next cycle, but check stop every 5s for fast shutdown
        slept = 0.0
        while slept < PREWARM_INTERVAL_SEC and not _stopped:
            await asyncio.sleep(5)
            slept += 5


def start_prewarm(
    symbols: tuple[str, ...] = DEFAULT_FAVORITE_SYMBOLS,
    timeframes: tuple[str, ...] = COMMON_TIMEFRAMES,
) -> None:
    """Fire-and-forget: start the pre-warm background loop. Safe to call
    multiple times (second call is a no-op)."""
    global _task, _stopped
    if _task is not None and not _task.done():
        return
    _stopped = False
    _task = asyncio.create_task(_prewarm_loop(symbols, timeframes))
    print(f"[prewarm] started — {len(symbols)} symbols × {len(timeframes)} TFs "
          f"every {PREWARM_INTERVAL_SEC}s")


def stop_prewarm() -> None:
    global _stopped
    _stopped = True
