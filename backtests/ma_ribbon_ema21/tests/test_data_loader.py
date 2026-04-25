from __future__ import annotations
from pathlib import Path
import pandas as pd
from backtests.ma_ribbon_ema21.data_loader import (
    load_ohlcv_from_csv,
    DataLoaderConfig,
)


def _write_small_fixture(tmp_path: Path, name: str = "BTCUSDT_1h.csv") -> Path:
    csv = tmp_path / name
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1700000000,100,101,99,100,1.0\n"
        "1700003600,100,101,99,101,1.0\n"
        "1700007200,101,102,100,102,1.0\n"
    )
    return csv


def test_load_ohlcv_from_csv_returns_correct_dataframe(tmp_path):
    _write_small_fixture(tmp_path)
    cfg = DataLoaderConfig(cache_dir=str(tmp_path))
    df = load_ohlcv_from_csv("BTCUSDT", "1h", cfg)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df.iloc[2]["close"] == 102.0


def test_load_ohlcv_from_csv_missing_returns_empty(tmp_path):
    cfg = DataLoaderConfig(cache_dir=str(tmp_path))
    df = load_ohlcv_from_csv("DOESNOTEXIST", "1h", cfg)
    assert df.empty
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_load_ohlcv_from_csv_sorts_by_timestamp(tmp_path):
    csv = tmp_path / "FOOUSDT_1h.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1700007200,103,104,102,103,1.0\n"
        "1700000000,100,101,99,100,1.0\n"
        "1700003600,101,102,100,101,1.0\n"
    )
    cfg = DataLoaderConfig(cache_dir=str(tmp_path))
    df = load_ohlcv_from_csv("FOOUSDT", "1h", cfg)
    assert list(df["timestamp"]) == [1700000000, 1700003600, 1700007200]
