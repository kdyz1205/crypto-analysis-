"""The single most important property of the dataset: no token in
the input sequence may have an end_bar_index >= the prediction bar.
"""
import numpy as np
import pandas as pd

from trendline_tokenizer.schemas.trendline import TrendlineRecord
from trendline_tokenizer.training.sequence_dataset import build_examples


def _df(n=300):
    return pd.DataFrame({
        "time": np.arange(n) * 60,
        "open": np.full(n, 100.0), "high": np.full(n, 101.0),
        "low": np.full(n, 99.0), "close": np.full(n, 100.5),
        "volume": np.full(n, 1.0),
    })


def _rec(s, e):
    return TrendlineRecord(
        id=f"x-{s}-{e}", symbol="BTC", exchange="bitget", timeframe="5m",
        start_time=s, end_time=e, start_bar_index=s, end_bar_index=e,
        start_price=100.0, end_price=101.0,
        line_role="support", direction="up", touch_count=2,
        label_source="auto", created_at=0,
    )


def test_input_tokens_strictly_before_prediction_bar():
    records = [_rec(s, s + 10) for s in (10, 30, 50, 70, 90, 110, 130, 150)]
    examples = build_examples(_df(), records, price_seq_len=64,
                              token_seq_len=8, horizon_bars=20, raw_feat_dim=36)
    for ex in examples:
        pred_bar = ex.prediction_bar_index
        for r in ex.input_records:
            assert r.end_bar_index < pred_bar, (
                f"leak: token end_bar {r.end_bar_index} >= pred {pred_bar}"
            )


def test_overlapping_records_still_no_lookahead():
    """Records with overlapping bar ranges must still respect the
    end_bar_index < pred_bar invariant."""
    records = [
        _rec(10, 50), _rec(20, 60), _rec(30, 70), _rec(40, 80),
        _rec(50, 90), _rec(60, 100),
    ]
    examples = build_examples(_df(), records, price_seq_len=32,
                              token_seq_len=4, horizon_bars=10, raw_feat_dim=36)
    for ex in examples:
        pred_bar = ex.prediction_bar_index
        for r in ex.input_records:
            assert r.end_bar_index < pred_bar
