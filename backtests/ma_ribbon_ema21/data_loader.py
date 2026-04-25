"""OHLCV data loader. CSV cache primary; Bitget fetch when cache empty / stale.

PRINCIPLES.md §P1: pull max available depth from Bitget. No tfDays cap.
PRINCIPLES.md §P2: if cache is shorter than requested, refetch.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import time

import pandas as pd


_LOG = logging.getLogger(__name__)


_BITGET_TF_MAP: dict[str, str] = {
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1H",
    "4h":  "4H",
}

# Bitget v2 USDT-M perp candles endpoint (publicly documented).
_BITGET_HISTORY_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"

_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DataLoaderConfig:
    cache_dir: str = "data/csv_cache/ma_ribbon_ema21"
    bitget_request_limit: int = 200            # Bitget cap per request
    bitget_pages_per_symbol: int = 1000        # safe upper bound; actual stops at empty page
    bitget_sleep_seconds: float = 0.05
    request_timeout_seconds: float = 15.0


def csv_path(symbol: str, tf: str, cfg: DataLoaderConfig) -> Path:
    return Path(cfg.cache_dir) / f"{symbol}_{tf}.csv"


def load_ohlcv_from_csv(symbol: str, tf: str, cfg: DataLoaderConfig) -> pd.DataFrame:
    """Read CSV cache. Empty DataFrame if missing/empty."""
    p = csv_path(symbol, tf, cfg)
    if not p.exists():
        return pd.DataFrame(columns=_COLS)
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=_COLS)
    df = df[_COLS].sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def fetch_ohlcv_from_bitget(symbol: str, tf: str, cfg: DataLoaderConfig) -> pd.DataFrame:
    """Fetch up to max-depth OHLCV from Bitget. Paginates until empty page."""
    import httpx  # imported here so test_data_loader can run without network

    if tf not in _BITGET_TF_MAP:
        raise ValueError(f"unsupported tf {tf!r}")
    bar = _BITGET_TF_MAP[tf]
    rows: list[list] = []
    end_ts: int | None = None
    for page in range(cfg.bitget_pages_per_symbol):
        params: dict[str, str] = {
            "symbol":      symbol,
            "productType": "USDT-FUTURES",
            "granularity": bar,
            "limit":       str(cfg.bitget_request_limit),
        }
        if end_ts is not None:
            params["endTime"] = str(end_ts)
        try:
            r = httpx.get(_BITGET_HISTORY_URL, params=params,
                          timeout=cfg.request_timeout_seconds)
            r.raise_for_status()
            data = r.json().get("data", []) or []
        except (httpx.HTTPError, ValueError) as exc:
            _LOG.error("bitget fetch failed for %s %s page %d: %s",
                       symbol, tf, page, exc)
            raise
        if not data:
            break
        rows.extend(data)
        # Bitget returns DESC; the smallest timestamp is the last item.
        end_ts = int(data[-1][0]) - 1
        time.sleep(cfg.bitget_sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=_COLS)
    # Bitget v2 returns up to 7 columns: ts, o, h, l, c, baseVol, quoteVol (varies).
    # Take the first 6 by position.
    df = pd.DataFrame([r[:6] for r in rows], columns=_COLS)
    df = df.astype({"timestamp": "int64"}).astype(
        {c: float for c in ["open", "high", "low", "close", "volume"]}
    )
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    _LOG.info("fetched %d bars for %s %s from Bitget", len(df), symbol, tf)
    return df


def write_csv_cache(df: pd.DataFrame, symbol: str, tf: str, cfg: DataLoaderConfig) -> Path:
    p = csv_path(symbol, tf, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def load_or_fetch(symbol: str, tf: str, cfg: DataLoaderConfig) -> pd.DataFrame:
    """Per PRINCIPLES §P2: if cache is empty, fetch and write. Always returns
    the freshest available DataFrame."""
    cached = load_ohlcv_from_csv(symbol, tf, cfg)
    if not cached.empty:
        return cached
    fresh = fetch_ohlcv_from_bitget(symbol, tf, cfg)
    if not fresh.empty:
        write_csv_cache(fresh, symbol, tf, cfg)
    return fresh
