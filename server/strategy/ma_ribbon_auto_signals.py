"""Live formation detection for MA-ribbon auto strategy.

Bull detector: re-uses Phase 1's `bullish_aligned` + `formation_events`.
Bear detector: mirror — ribbon stack flipped (close < MA5 < MA8 < EMA21 < MA55).
Both emit Phase1Signal with a UUID per event.
"""
from __future__ import annotations
import uuid
import numpy as np
import pandas as pd

from backtests.ma_ribbon_ema21.indicators import sma, ema
from backtests.ma_ribbon_ema21.ma_alignment import AlignmentConfig, bullish_aligned
from server.strategy.ma_ribbon_auto_adapter import Phase1Signal


_LONG_CFG = AlignmentConfig.default()


def _bear_aligned(df: pd.DataFrame) -> pd.Series:
    """Strict bear stack: close < MA5 < MA8 < EMA21 < MA55."""
    required = ["close", "ma5", "ma8", "ema21", "ma55"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"_bear_aligned: df missing {c}")
    aligned = (
        (df["close"] < df["ma5"]) &
        (df["ma5"]   < df["ma8"]) &
        (df["ma8"]   < df["ema21"]) &
        (df["ema21"] < df["ma55"])
    )
    nan_mask = df[required].isna().any(axis=1)
    aligned = aligned.fillna(False)
    aligned[nan_mask] = False
    return aligned.astype(bool)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma5"]   = sma(out["close"], 5)
    out["ma8"]   = sma(out["close"], 8)
    out["ema21"] = ema(out["close"], 21)
    out["ma55"]  = sma(out["close"], 55)
    return out


def _formation_idx_set(aligned: pd.Series) -> list[int]:
    a = aligned.to_numpy(dtype=bool)
    if len(a) == 0:
        return []
    prev = np.concatenate(([False], a[:-1]))
    transitions = a & ~prev
    transitions[0] = False
    return [int(i) for i in np.where(transitions)[0]]


def detect_new_signals_for_pair(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    direction: str,
    last_processed_bar_ts: int,
) -> list[Phase1Signal]:
    if df is None or df.empty or len(df) < 60:
        return []
    enriched = _enrich(df)
    if direction == "long":
        aligned = bullish_aligned(enriched, _LONG_CFG)
    elif direction == "short":
        aligned = _bear_aligned(enriched)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    formation_idxs = _formation_idx_set(aligned)
    out: list[Phase1Signal] = []
    n = len(enriched)
    closes = enriched["close"].to_numpy(dtype=float)
    ema21_arr = enriched["ema21"].to_numpy(dtype=float)
    ts_arr = enriched["timestamp"].to_numpy(dtype="int64")
    for i in formation_idxs:
        if int(ts_arr[i]) <= last_processed_bar_ts:
            continue
        next_open = float(closes[i + 1]) if i + 1 < n else float(closes[i])
        out.append(Phase1Signal(
            signal_id=uuid.uuid4().hex,
            symbol=symbol,
            tf=tf,
            direction=direction,
            signal_bar_ts=int(ts_arr[i]),
            next_bar_open_estimate=next_open,
            ema21_at_signal=float(ema21_arr[i]),
        ))
    return out


class BullSignalDetector:
    @staticmethod
    def detect(df, symbol, tf, last_processed_bar_ts):
        return detect_new_signals_for_pair(df, symbol, tf, "long", last_processed_bar_ts)


class BearSignalDetector:
    @staticmethod
    def detect(df, symbol, tf, last_processed_bar_ts):
        return detect_new_signals_for_pair(df, symbol, tf, "short", last_processed_bar_ts)
