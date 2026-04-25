from __future__ import annotations
import pandas as pd
import pytest
from backtests.ma_ribbon_ema21.data_split import split_by_time, label_split_column


def test_split_by_time_70_30_count():
    ts = list(range(100))
    is_train, is_test = split_by_time(ts, train_pct=0.70)
    assert sum(is_train) == 70
    assert sum(is_test) == 30


def test_split_by_time_train_is_first_70_pct():
    ts = list(range(10))
    is_train, is_test = split_by_time(ts, train_pct=0.70)
    assert is_train == [True]*7 + [False]*3
    assert is_test  == [False]*7 + [True]*3


def test_split_by_time_invalid_pct_raises():
    with pytest.raises(ValueError):
        split_by_time([1, 2, 3], train_pct=1.5)
    with pytest.raises(ValueError):
        split_by_time([1, 2, 3], train_pct=-0.1)


def test_label_split_column_attaches_train_test_labels():
    df = pd.DataFrame({"timestamp": list(range(10)), "close": list(range(10))})
    out = label_split_column(df, train_pct=0.70)
    assert "split" in out.columns
    assert (out.iloc[:7]["split"] == "train").all()
    assert (out.iloc[7:]["split"] == "test").all()


def test_label_split_column_does_not_mutate_input():
    df = pd.DataFrame({"timestamp": list(range(10)), "close": list(range(10))})
    _ = label_split_column(df, train_pct=0.70)
    assert "split" not in df.columns
