"""(price_window, token_seq, targets) examples for the fusion model.

Strict no-lookahead invariant:
    every TrendlineRecord in token_seq must have end_bar_index <
    prediction_bar_index.
The "next" target is the FIRST record whose end_bar_index >=
prediction_bar_index - that's what the model is trying to forecast.

Each input record produces THREE parallel representations:
  rule_*    - from tokenizer/rule.py
  learned_* - from learned/vqvae.py if a checkpoint is provided, else zeros
  raw_feat  - from features/vector.py (36-dim continuous)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..schemas.trendline import TrendlineRecord
from ..tokenizer.rule import encode_rule
from ..features.vector import build_feature_vector


@dataclass
class SequenceExample:
    price_window: np.ndarray            # (T_price, F)
    price_pad: np.ndarray               # (T_price,) bool
    rule_coarse_ids: np.ndarray         # (T_token,) int64
    rule_fine_ids: np.ndarray           # (T_token,) int64
    learned_coarse_ids: np.ndarray      # (T_token,) int64 - zeros if no vqvae
    learned_fine_ids: np.ndarray        # (T_token,) int64 - zeros if no vqvae
    raw_feats: np.ndarray               # (T_token, raw_feat_dim) float32
    token_pad: np.ndarray               # (T_token,) bool
    prediction_bar_index: int
    input_records: list[TrendlineRecord] = field(default_factory=list)
    next_coarse: int = 0
    next_fine: int = 0
    bounce: int = 0
    brk: int = 0
    cont: int = 0
    buffer_pct: float = 0.0
    # Phase 2 supervised targets (derived from the target record)
    # regime:        0=low_vol, 1=normal_vol, 2=high_vol
    # invalidation:  0=valid, 1=weak_pen, 2=confirmed_break,
    #                3=break_retest, 4=failed_breakout
    regime: int = 0
    invalidation: int = 0


def _price_window(df: pd.DataFrame, end_idx: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (window, pad_mask). end_idx is exclusive.

    All features are scale-stable so the linear projection in the
    price encoder behaves no matter the absolute price level:
      - OHLC -> log-ratio vs last close in window (centered ~0)
      - volume -> log1p(vol / median_vol)
      - ema21, ema50 -> log(ema / close)
      - atr14 -> atr / close (relative ATR)
      - rsi14 -> rsi / 100 (in [0, 1])
      - ret1, ret5 -> already returns
      - vol_z, dist_ma_atr -> already normalized
    """
    start = max(0, end_idx - length)
    raw = df.iloc[start:end_idx]
    if len(raw) == 0:
        feats = np.zeros((0, 13), dtype=np.float32)
    else:
        o = raw["open"].to_numpy(dtype=np.float64)
        h = raw["high"].to_numpy(dtype=np.float64)
        lo = raw["low"].to_numpy(dtype=np.float64)
        c = raw["close"].to_numpy(dtype=np.float64)
        v = raw["volume"].to_numpy(dtype=np.float64)

        # reference close for OHLC log-ratios
        ref = c[-1] if c[-1] > 0 else 1.0
        log_o = np.log(np.maximum(o, 1e-9) / ref)
        log_h = np.log(np.maximum(h, 1e-9) / ref)
        log_l = np.log(np.maximum(lo, 1e-9) / ref)
        log_c = np.log(np.maximum(c, 1e-9) / ref)

        # volume: log1p(vol/median)
        med_v = np.median(v) if np.median(v) > 0 else 1.0
        log_v = np.log1p(v / med_v) - np.log1p(1.0)  # centered around 0 at median

        # indicators
        ema21 = pd.Series(c).ewm(span=21, adjust=False).mean().to_numpy()
        ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
        ema21_rel = np.log(np.maximum(ema21, 1e-9) / np.maximum(c, 1e-9))
        ema50_rel = np.log(np.maximum(ema50, 1e-9) / np.maximum(c, 1e-9))

        prev_close = np.roll(c, 1); prev_close[0] = c[0]
        tr = np.maximum.reduce([h - lo, np.abs(h - prev_close), np.abs(lo - prev_close)])
        atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
        atr_rel = atr14 / np.maximum(c, 1e-9)

        delta = np.diff(c, prepend=c[:1])
        up = np.where(delta > 0, delta, 0.0)
        dn = np.where(delta < 0, -delta, 0.0)
        up_avg = pd.Series(up).rolling(14, min_periods=1).mean()
        dn_avg = pd.Series(dn).rolling(14, min_periods=1).mean().replace(0, 1e-9)
        rs = (up_avg / dn_avg).to_numpy()
        rsi14 = (100 - 100 / (1 + rs)) / 100.0  # in [0, 1]

        ret1 = np.diff(c, prepend=c[:1]) / np.maximum(c, 1e-9)
        if len(c) > 5:
            ret5 = (c - np.roll(c, 5)) / np.maximum(c, 1e-9)
            ret5[:5] = 0.0
        else:
            ret5 = np.zeros_like(c)

        vol_std = v.std()
        vol_z = (v - v.mean()) / (vol_std if vol_std > 0 else 1.0)
        vol_z = np.clip(vol_z, -5, 5)

        dist_ma_atr = (c - ema21) / np.maximum(atr14, 1e-9)
        dist_ma_atr = np.clip(dist_ma_atr, -10, 10)

        feats = np.column_stack([
            log_o, log_h, log_l, log_c, log_v,            # 5 OHLCV
            ema21_rel, ema50_rel, atr_rel, rsi14,         # 4 indicators
            np.clip(vol_z, -5, 5),
            np.clip(dist_ma_atr, -10, 10),
            np.clip(ret1, -0.5, 0.5),
            np.clip(ret5, -0.5, 0.5),                     # 4 more = 8 total
        ]).astype(np.float32)
        # Sanity: replace any residual NaN/inf with 0
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    pad = length - feats.shape[0]
    if pad > 0:
        feats = np.concatenate([np.zeros((pad, 13), dtype=np.float32), feats], axis=0)
        mask = np.concatenate([np.ones(pad, dtype=bool), np.zeros(length - pad, dtype=bool)])
    else:
        mask = np.zeros(length, dtype=bool)
    return feats, mask


def _regime_class(rec: TrendlineRecord) -> int:
    """Phase 2 regime label: 0=low_vol, 1=normal_vol, 2=high_vol."""
    v = rec.volatility_atr_pct
    if v is None:
        return 1
    if v < 0.005:
        return 0
    if v < 0.015:
        return 1
    return 2


def _invalidation_class(rec: TrendlineRecord) -> int:
    """Phase 2 invalidation label:
      0=valid, 1=weak_pen, 2=confirmed_break,
      3=break_retest, 4=failed_breakout

    Derived from break_after / retested_after_break / break_distance_atr.
    Best-effort — many records won't have full outcome metadata; default
    to 0 (valid) when unsure.
    """
    if not rec.break_after:
        # Touched but didn't break -> weak penetration only if
        # break_distance_atr is present and small
        if rec.break_distance_atr is not None and 0 < rec.break_distance_atr < 0.5:
            return 1   # weak_pen
        return 0       # valid
    # Broke
    if rec.retested_after_break:
        return 3       # break + retest
    # Broken but not retested. If distance is huge, it's a confirmed break;
    # if tiny, it's likely a failed breakout (price reverted)
    if rec.break_distance_atr is not None and rec.break_distance_atr < 0.3:
        return 4       # failed_breakout
    return 2           # confirmed_break


def _label_outcomes(rec: TrendlineRecord, df: pd.DataFrame, horizon_bars: int) -> dict:
    regime = _regime_class(rec)
    invalidation = _invalidation_class(rec)
    if rec.bounce_after is not None and rec.break_after is not None:
        bounce = int(bool(rec.bounce_after))
        brk = int(bool(rec.break_after))
        cont = int(not brk)
        # bounce_strength_atr can be 0..100+, but the target buffer is
        # a price-percent in [0, 0.05]. Clip aggressively to avoid MSE
        # gradient explosion (root cause of training NaN at 30k records).
        raw_strength = float(rec.bounce_strength_atr or 0.0)
        buffer_pct = max(0.0, min(0.05, raw_strength * 0.005))
        return {"bounce": bounce, "brk": brk, "cont": cont,
                "buffer_pct": buffer_pct, "regime": regime,
                "invalidation": invalidation}
    end = min(len(df), rec.end_bar_index + horizon_bars)
    seg = df.iloc[rec.end_bar_index:end]
    if len(seg) == 0 or rec.start_price <= 0 or rec.end_price <= 0:
        return {"bounce": 0, "brk": 0, "cont": 1, "buffer_pct": 0.005,
                "regime": regime, "invalidation": invalidation}
    moved = (float(seg["close"].iloc[-1]) - rec.end_price) / max(rec.end_price, 1e-9)
    if rec.line_role == "support":
        bounce = int(moved > 0.005); brk = int(moved < -0.005)
    elif rec.line_role == "resistance":
        bounce = int(moved < -0.005); brk = int(moved > 0.005)
    else:
        bounce = brk = 0
    cont = int(not brk)
    buffer_pct = float(seg["high"].max() - seg["low"].min()) / max(rec.end_price, 1e-9)
    return {"bounce": bounce, "brk": brk, "cont": cont,
            "buffer_pct": min(0.05, max(0.0, buffer_pct)),
            "regime": regime, "invalidation": invalidation}


def build_examples(
    df: pd.DataFrame,
    records: Sequence[TrendlineRecord],
    *,
    price_seq_len: int,
    token_seq_len: int,
    horizon_bars: int,
    raw_feat_dim: int,
    vqvae=None,
) -> list[SequenceExample]:
    """One example per record (the record is the prediction target)."""
    sorted_recs = sorted(records, key=lambda r: r.end_bar_index)
    examples: list[SequenceExample] = []
    for i, target in enumerate(sorted_recs):
        pred_bar = target.end_bar_index
        input_recs = [r for r in sorted_recs[:i] if r.end_bar_index < pred_bar]
        input_recs = input_recs[-token_seq_len:]

        rule_coarse = np.zeros(token_seq_len, dtype=np.int64)
        rule_fine = np.zeros(token_seq_len, dtype=np.int64)
        learned_coarse = np.zeros(token_seq_len, dtype=np.int64)
        learned_fine = np.zeros(token_seq_len, dtype=np.int64)
        raw_feats = np.zeros((token_seq_len, raw_feat_dim), dtype=np.float32)
        token_pad = np.ones(token_seq_len, dtype=bool)

        if input_recs:
            feats = np.stack([build_feature_vector(r) for r in input_recs], axis=0)
            for j, r in enumerate(input_recs):
                slot = token_seq_len - len(input_recs) + j
                tok = encode_rule(r)
                rule_coarse[slot] = tok.coarse_token_id
                rule_fine[slot] = tok.fine_token_id
                raw_feats[slot] = feats[j]
                token_pad[slot] = False
            if vqvae is not None:
                with torch.no_grad():
                    feat_t = torch.from_numpy(feats).float()
                    c_idx, f_idx = vqvae.tokenize(feat_t)
                for j in range(len(input_recs)):
                    slot = token_seq_len - len(input_recs) + j
                    learned_coarse[slot] = int(c_idx[j].item())
                    learned_fine[slot] = int(f_idx[j].item())

        price_window, price_pad = _price_window(df, pred_bar, price_seq_len)
        target_tok = encode_rule(target)
        outcomes = _label_outcomes(target, df, horizon_bars)
        examples.append(SequenceExample(
            price_window=price_window, price_pad=price_pad,
            rule_coarse_ids=rule_coarse, rule_fine_ids=rule_fine,
            learned_coarse_ids=learned_coarse, learned_fine_ids=learned_fine,
            raw_feats=raw_feats, token_pad=token_pad,
            prediction_bar_index=pred_bar,
            input_records=list(input_recs),
            next_coarse=target_tok.coarse_token_id,
            next_fine=target_tok.fine_token_id,
            bounce=outcomes["bounce"], brk=outcomes["brk"],
            cont=outcomes["cont"], buffer_pct=outcomes["buffer_pct"],
            regime=outcomes.get("regime", 1),
            invalidation=outcomes.get("invalidation", 0),
        ))
    return examples


class SequenceDataset(Dataset):
    def __init__(self, examples: list[SequenceExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "price": torch.from_numpy(ex.price_window),
            "price_pad": torch.from_numpy(ex.price_pad),
            "rule_coarse": torch.from_numpy(ex.rule_coarse_ids),
            "rule_fine": torch.from_numpy(ex.rule_fine_ids),
            "learned_coarse": torch.from_numpy(ex.learned_coarse_ids),
            "learned_fine": torch.from_numpy(ex.learned_fine_ids),
            "raw_feat": torch.from_numpy(ex.raw_feats),
            "token_pad": torch.from_numpy(ex.token_pad),
            "next_coarse": torch.tensor(ex.next_coarse, dtype=torch.long),
            "next_fine": torch.tensor(ex.next_fine, dtype=torch.long),
            "bounce": torch.tensor(ex.bounce, dtype=torch.long),
            "brk": torch.tensor(ex.brk, dtype=torch.long),
            "cont": torch.tensor(ex.cont, dtype=torch.long),
            "buffer_pct": torch.tensor(ex.buffer_pct, dtype=torch.float32),
            "regime": torch.tensor(ex.regime, dtype=torch.long),
            "invalidation": torch.tensor(ex.invalidation, dtype=torch.long),
        }
