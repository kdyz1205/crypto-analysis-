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


def _price_window(df: pd.DataFrame, end_idx: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (window, pad_mask). end_idx is exclusive."""
    start = max(0, end_idx - length)
    raw = df.iloc[start:end_idx]
    if len(raw) == 0:
        feats = np.zeros((0, 13), dtype=np.float32)
    else:
        ohlcv = raw[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
        closes = ohlcv[:, 3]
        ema21 = pd.Series(closes).ewm(span=21, adjust=False).mean().to_numpy()
        ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().to_numpy()
        ret1 = np.diff(closes, prepend=closes[:1]) / np.maximum(closes, 1e-9)
        ret5 = (closes - np.roll(closes, 5)) / np.maximum(closes, 1e-9)
        if len(ret5) > 5:
            ret5[:5] = 0.0
        else:
            ret5[:] = 0.0
        high = ohlcv[:, 1]; low = ohlcv[:, 2]
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr = np.maximum.reduce([high - low,
                                np.abs(high - prev_close),
                                np.abs(low - prev_close)])
        atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
        delta = np.diff(closes, prepend=closes[:1])
        up = np.where(delta > 0, delta, 0.0)
        dn = np.where(delta < 0, -delta, 0.0)
        up_avg = pd.Series(up).rolling(14, min_periods=1).mean()
        dn_avg = pd.Series(dn).rolling(14, min_periods=1).mean().replace(0, 1e-9)
        rs = (up_avg / dn_avg).to_numpy()
        rsi14 = 100 - 100 / (1 + rs)
        vol = ohlcv[:, 4]
        vol_std = vol.std()
        vol_z = (vol - vol.mean()) / (vol_std if vol_std > 0 else 1e-9)
        dist_ma_atr = (closes - ema21) / np.maximum(atr14, 1e-9)

        feats = np.column_stack([
            ohlcv,
            ema21, ema50, atr14, rsi14, vol_z, dist_ma_atr, ret1, ret5,
        ]).astype(np.float32)

    pad = length - feats.shape[0]
    if pad > 0:
        feats = np.concatenate([np.zeros((pad, 13), dtype=np.float32), feats], axis=0)
        mask = np.concatenate([np.ones(pad, dtype=bool), np.zeros(length - pad, dtype=bool)])
    else:
        mask = np.zeros(length, dtype=bool)
    return feats, mask


def _label_outcomes(rec: TrendlineRecord, df: pd.DataFrame, horizon_bars: int) -> dict:
    if rec.bounce_after is not None and rec.break_after is not None:
        bounce = int(bool(rec.bounce_after))
        brk = int(bool(rec.break_after))
        cont = int(not brk)
        buffer_pct = float(rec.bounce_strength_atr or 0.0) * 0.005
        return {"bounce": bounce, "brk": brk, "cont": cont, "buffer_pct": buffer_pct}
    end = min(len(df), rec.end_bar_index + horizon_bars)
    seg = df.iloc[rec.end_bar_index:end]
    if len(seg) == 0 or rec.start_price <= 0 or rec.end_price <= 0:
        return {"bounce": 0, "brk": 0, "cont": 1, "buffer_pct": 0.005}
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
            "buffer_pct": min(0.05, max(0.0, buffer_pct))}


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
        }
