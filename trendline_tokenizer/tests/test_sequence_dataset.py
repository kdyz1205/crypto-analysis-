"""SequenceDataset tests: shapes + multi-stream presence."""
import numpy as np
import pandas as pd

from trendline_tokenizer.schemas.trendline import TrendlineRecord
from trendline_tokenizer.training.sequence_dataset import (
    SequenceDataset, build_examples,
)


def _toy_ohlcv(n=400):
    return pd.DataFrame({
        "time": np.arange(n) * 60,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100.5, 110.5, n),
        "volume": np.full(n, 1.0),
    })


def _toy_record(start_idx: int, end_idx: int, role="support") -> TrendlineRecord:
    return TrendlineRecord(
        id=f"r-{start_idx}-{end_idx}",
        symbol="BTCUSDT", exchange="bitget", timeframe="5m",
        start_time=start_idx * 60, end_time=end_idx * 60,
        start_bar_index=start_idx, end_bar_index=end_idx,
        start_price=100.0, end_price=101.0,
        line_role=role, direction="up", touch_count=2,
        bounce_after=True, bounce_strength_atr=1.0,
        break_after=False, break_distance_atr=None,
        retested_after_break=False,
        volatility_atr_pct=0.01, volume_z_score=0.0,
        label_source="auto", auto_method="test",
        created_at=0,
    )


def test_dataset_length_matches_examples():
    df = _toy_ohlcv()
    records = [_toy_record(50, 80), _toy_record(120, 150), _toy_record(220, 250)]
    examples = build_examples(df, records, price_seq_len=64, token_seq_len=8,
                              horizon_bars=20, raw_feat_dim=36)
    ds = SequenceDataset(examples)
    assert len(ds) == len(examples) > 0


def test_dataset_returns_correct_shapes():
    df = _toy_ohlcv()
    records = [_toy_record(50 + 30 * i, 80 + 30 * i) for i in range(8)]
    examples = build_examples(df, records, price_seq_len=64, token_seq_len=8,
                              horizon_bars=20, raw_feat_dim=36)
    ds = SequenceDataset(examples)
    sample = ds[0]
    assert sample["price"].shape == (64, 13)
    # rule stream
    assert sample["rule_coarse"].shape == (8,)
    assert sample["rule_fine"].shape == (8,)
    # learned stream (zeros if no vqvae provided)
    assert sample["learned_coarse"].shape == (8,)
    assert sample["learned_fine"].shape == (8,)
    # raw stream
    assert sample["raw_feat"].shape == (8, 36)
    # padding
    assert sample["price_pad"].shape == (64,)
    assert sample["token_pad"].shape == (8,)
    # targets
    assert "next_coarse" in sample and "next_fine" in sample
    assert "bounce" in sample and "brk" in sample and "cont" in sample
    assert "buffer_pct" in sample


def test_dataset_learned_tokens_zero_without_vqvae():
    df = _toy_ohlcv()
    records = [_toy_record(50 + 30 * i, 80 + 30 * i) for i in range(4)]
    examples = build_examples(df, records, price_seq_len=64, token_seq_len=8,
                              horizon_bars=20, raw_feat_dim=36, vqvae=None)
    ds = SequenceDataset(examples)
    for i in range(len(ds)):
        s = ds[i]
        assert (s["learned_coarse"] == 0).all()
        assert (s["learned_fine"] == 0).all()
