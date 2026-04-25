"""Async, concurrency-controlled Bitget OHLCV loader.

Designed for the live web panel: fetch hundreds of (symbol, TF) pairs
from Bitget v2 USDT-FUTURES in parallel via httpx.AsyncClient + a
bounded semaphore, then merge into a {(symbol, TF): DataFrame} dict
that the scan engine can consume directly (no CSV round-trip).

In-process, time-aware cache keeps results for `cache_ttl_seconds`
(default 300s) so repeated panel clicks within 5 minutes don't
hammer the exchange.
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

import httpx
import pandas as pd


_LOG = logging.getLogger(__name__)

_BITGET_TF_MAP: dict[str, str] = {
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1H",
    "4h":  "4H",
}

_BITGET_HISTORY_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"
_BITGET_TICKERS_URL = "https://api.bitget.com/api/v2/mix/market/tickers"

_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


# ───────────────────────── in-process cache ─────────────────────────

_CACHE: dict[tuple[str, str], tuple[pd.DataFrame, float]] = {}
_UNIVERSE_CACHE: tuple[list[str], float] | None = None


@dataclass
class AsyncLoaderConfig:
    pages_per_symbol: int = 5         # 5 * 200 = 1000 bars per (sym, TF)
    concurrency: int = 10              # Bitget public limit ~20 req/s
    request_timeout_seconds: float = 12.0
    cache_ttl_seconds: int = 300       # 5 min in-process cache


# ───────────────────────── universe (all USDT perps) ─────────────────────────


async def fetch_all_usdt_perp_symbols(
    client: httpx.AsyncClient,
    cfg: AsyncLoaderConfig,
    min_quote_volume_24h: float = 1_000_000.0,
) -> list[str]:
    """Returns sorted list of USDT-perp symbol strings filtered by 24h quote volume.
    Uses Bitget tickers endpoint. Cached for 5 minutes in process memory."""
    global _UNIVERSE_CACHE
    now = time.time()
    if _UNIVERSE_CACHE and (now - _UNIVERSE_CACHE[1] < cfg.cache_ttl_seconds):
        return _UNIVERSE_CACHE[0]

    r = await client.get(
        _BITGET_TICKERS_URL,
        params={"productType": "USDT-FUTURES"},
        timeout=cfg.request_timeout_seconds,
    )
    r.raise_for_status()
    data = r.json().get("data", []) or []
    out: list[str] = []
    for row in data:
        sym = row.get("symbol")
        try:
            qv = float(row.get("usdtVolume") or row.get("quoteVolume") or 0)
        except (TypeError, ValueError):
            qv = 0.0
        if sym and sym.endswith("USDT") and qv >= min_quote_volume_24h:
            out.append(sym)
    out.sort()
    _UNIVERSE_CACHE = (out, now)
    _LOG.info("bitget USDT-perp universe: %d symbols (>= $%.0f 24h vol)",
              len(out), min_quote_volume_24h)
    return out


# ───────────────────────── single (symbol, TF) fetch ─────────────────────────


async def fetch_ohlcv_async(
    client: httpx.AsyncClient,
    symbol: str,
    tf: str,
    cfg: AsyncLoaderConfig,
) -> pd.DataFrame:
    """Fetch up to cfg.pages_per_symbol pages of OHLCV for one (symbol, TF).
    Returns ascending-sorted DataFrame or an empty one."""
    if tf not in _BITGET_TF_MAP:
        raise ValueError(f"unsupported tf {tf!r}")
    bar = _BITGET_TF_MAP[tf]

    rows: list[list] = []
    end_ts: int | None = None
    for _ in range(cfg.pages_per_symbol):
        params: dict[str, str] = {
            "symbol":      symbol,
            "productType": "USDT-FUTURES",
            "granularity": bar,
            "limit":       "200",
        }
        if end_ts is not None:
            params["endTime"] = str(end_ts)
        try:
            r = await client.get(
                _BITGET_HISTORY_URL, params=params,
                timeout=cfg.request_timeout_seconds,
            )
            r.raise_for_status()
            data = r.json().get("data", []) or []
        except (httpx.HTTPError, ValueError) as exc:
            _LOG.warning("bitget fetch %s %s page failed: %s", symbol, tf, exc)
            break
        if not data:
            break
        rows.extend(data)
        end_ts = int(data[-1][0]) - 1

    if not rows:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame([r[:6] for r in rows], columns=_COLS)
    df = df.astype({"timestamp": "int64"}).astype(
        {c: float for c in ["open", "high", "low", "close", "volume"]}
    )
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ───────────────────────── batched universe fetch ─────────────────────────


@dataclass
class FetchProgress:
    total: int = 0
    done: int = 0
    started_at: float = field(default_factory=time.time)
    last_symbol: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def to_dict(self) -> dict:
        return {
            "total": self.total, "done": self.done,
            "elapsed_seconds": round(self.elapsed, 2),
            "last_symbol": self.last_symbol,
            "errors_count": len(self.errors),
        }


async def fetch_universe_async(
    symbols: Iterable[str],
    tfs: Iterable[str],
    cfg: AsyncLoaderConfig | None = None,
    progress: FetchProgress | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Fetch many (symbol, TF) pairs in parallel. Returns dict keyed by (symbol, TF).
    Pairs that fail or return empty are still keyed but with empty DataFrame.
    Caches successful fetches in-process for cfg.cache_ttl_seconds.
    """
    cfg = cfg or AsyncLoaderConfig()
    pairs = [(s, t) for s in symbols for t in tfs]
    out: dict[tuple[str, str], pd.DataFrame] = {}
    sem = asyncio.Semaphore(cfg.concurrency)
    now = time.time()
    if progress is None:
        progress = FetchProgress()
    progress.total = len(pairs)
    progress.done = 0

    # First pass: pull anything fresh in cache
    to_fetch: list[tuple[str, str]] = []
    for key in pairs:
        cached = _CACHE.get(key)
        if cached and (now - cached[1] < cfg.cache_ttl_seconds):
            out[key] = cached[0]
            progress.done += 1
        else:
            to_fetch.append(key)

    if not to_fetch:
        return out

    async with httpx.AsyncClient() as client:
        async def one(sym: str, tf: str):
            async with sem:
                try:
                    df = await fetch_ohlcv_async(client, sym, tf, cfg)
                    out[(sym, tf)] = df
                    if not df.empty:
                        _CACHE[(sym, tf)] = (df, time.time())
                except Exception as exc:  # noqa: BLE001
                    progress.errors.append(f"{sym} {tf}: {exc}")
                    out[(sym, tf)] = pd.DataFrame(columns=_COLS)
                finally:
                    progress.done += 1
                    progress.last_symbol = sym

        await asyncio.gather(*(one(s, t) for s, t in to_fetch))
    return out
