"""Phase 1 engine: scan formation events for (symbol, TF) pairs and attach
distance features + forward returns. Aggregation across the universe."""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import math
from typing import Iterable

import pandas as pd

from backtests.ma_ribbon_ema21.indicators import sma, ema
from backtests.ma_ribbon_ema21.ma_alignment import (
    AlignmentConfig,
    bullish_aligned,
    formation_events,
)
from backtests.ma_ribbon_ema21.distance_features import (
    distance_to_ma_pct,
)
from backtests.ma_ribbon_ema21.forward_returns import (
    forward_returns_at,
    apply_round_trip_cost,
)
from backtests.ma_ribbon_ema21.data_loader import (
    DataLoaderConfig,
    load_or_fetch,
)


_LOG = logging.getLogger(__name__)
_DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20, 50)


@dataclass
class Phase1Event:
    symbol: str
    tf: str
    formation_bar_idx: int
    formation_timestamp: int
    formation_close: float
    distance_to_ma5_pct:   float
    distance_to_ma8_pct:   float
    distance_to_ema21_pct: float
    distance_to_ma55_pct:  float
    ribbon_width_pct:      float
    forward_returns:          dict[int, float] = field(default_factory=dict)
    forward_returns_post_fee: dict[int, float] = field(default_factory=dict)
    split: str = "unknown"   # "train" or "test"


@dataclass
class UniverseConfig:
    symbols: list[str]
    timeframes: list[str]
    loader: DataLoaderConfig
    alignment_cfg: AlignmentConfig | None = None
    forward_horizons: tuple[int, ...] = _DEFAULT_HORIZONS
    fee_per_side: float = 0.0005
    slippage_per_fill: float = 0.0001
    train_pct: float = 0.70


def _enrich_with_indicators(
    df: pd.DataFrame,
    ma5_period: int = 5,
    ma8_period: int = 8,
    ema21_period: int = 21,
    ma55_period: int = 55,
) -> pd.DataFrame:
    out = df.copy()
    out["ma5"]   = sma(out["close"], ma5_period)
    out["ma8"]   = sma(out["close"], ma8_period)
    out["ema21"] = ema(out["close"], ema21_period)
    out["ma55"]  = sma(out["close"], ma55_period)
    return out


def scan_symbol_tf(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    alignment_cfg: AlignmentConfig | None = None,
    forward_horizons: Iterable[int] = _DEFAULT_HORIZONS,
    fee_per_side: float = 0.0005,
    slippage_per_fill: float = 0.0001,
    train_pct: float = 0.70,
) -> list[Phase1Event]:
    """Scan one (symbol, TF) DataFrame for formation events. Returns list."""
    if df.empty:
        return []
    if "close" not in df.columns:
        raise ValueError("scan_symbol_tf: df missing 'close' column")
    enriched = _enrich_with_indicators(df)
    cfg = alignment_cfg or AlignmentConfig.default()
    aligned = bullish_aligned(enriched, cfg)
    event_idx = formation_events(aligned)

    horizons = tuple(forward_horizons)
    n = len(enriched)
    cutoff = int(n * train_pct)

    d5_series  = distance_to_ma_pct(enriched["close"], enriched["ma5"])
    d8_series  = distance_to_ma_pct(enriched["close"], enriched["ma8"])
    d21_series = distance_to_ma_pct(enriched["close"], enriched["ema21"])
    d55_series = distance_to_ma_pct(enriched["close"], enriched["ma55"])

    out: list[Phase1Event] = []
    for i in event_idx:
        i = int(i)
        row = enriched.iloc[i]
        ts = int(row["timestamp"])
        close_i = float(row["close"])
        ma55_i  = float(row["ma55"])
        rib_w = (float(row["ma5"]) - ma55_i) / ma55_i \
                if not math.isnan(ma55_i) and ma55_i != 0 else math.nan

        raw  = forward_returns_at(enriched["close"], i, list(horizons))
        post = {h: apply_round_trip_cost(r, fee_per_side, slippage_per_fill)
                for h, r in raw.items()}

        out.append(Phase1Event(
            symbol=symbol, tf=tf,
            formation_bar_idx=i,
            formation_timestamp=ts,
            formation_close=close_i,
            distance_to_ma5_pct=float(d5_series.iloc[i]),
            distance_to_ma8_pct=float(d8_series.iloc[i]),
            distance_to_ema21_pct=float(d21_series.iloc[i]),
            distance_to_ma55_pct=float(d55_series.iloc[i]),
            ribbon_width_pct=float(rib_w),
            forward_returns=raw,
            forward_returns_post_fee=post,
            split=("train" if i < cutoff else "test"),
        ))
    return out


def scan_universe(cfg: UniverseConfig) -> list[Phase1Event]:
    """Scan every (symbol, TF) pair. Skip pairs whose data is unavailable.
    Per PRINCIPLES §P11: log every load result.
    """
    out: list[Phase1Event] = []
    for symbol in cfg.symbols:
        for tf in cfg.timeframes:
            df = load_or_fetch(symbol, tf, cfg.loader)
            if df.empty:
                _LOG.warning("no data for %s %s — skipping", symbol, tf)
                continue
            _LOG.info("scanning %s %s (%d bars)", symbol, tf, len(df))
            events = scan_symbol_tf(
                df, symbol=symbol, tf=tf,
                alignment_cfg=cfg.alignment_cfg,
                forward_horizons=cfg.forward_horizons,
                fee_per_side=cfg.fee_per_side,
                slippage_per_fill=cfg.slippage_per_fill,
                train_pct=cfg.train_pct,
            )
            out.extend(events)
            _LOG.info("  -> %d events for %s %s", len(events), symbol, tf)
    return out
