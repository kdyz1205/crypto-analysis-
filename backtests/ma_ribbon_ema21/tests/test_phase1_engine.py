from __future__ import annotations
import pandas as pd
import pytest

from backtests.ma_ribbon_ema21.phase1_engine import (
    scan_symbol_tf,
    scan_universe,
    Phase1Event,
    UniverseConfig,
)
from backtests.ma_ribbon_ema21.data_loader import DataLoaderConfig
from backtests.ma_ribbon_ema21.tests.fixtures import (
    make_uptrend_with_formation,
    make_flat_ohlcv,
)


def test_scan_symbol_tf_returns_at_least_one_event_on_uptrend():
    df, formation_bar = make_uptrend_with_formation(
        n_bars=300, formation_at_bar=120, base_price=100.0
    )
    events = scan_symbol_tf(df, symbol="TESTUSDT", tf="1h")
    assert len(events) >= 1
    e0 = events[0]
    assert isinstance(e0, Phase1Event)
    assert e0.symbol == "TESTUSDT"
    assert e0.tf == "1h"
    # Event should land near the regime change. Loose bound (noise + MA lag).
    assert e0.formation_bar_idx >= formation_bar - 5
    assert e0.formation_bar_idx <= formation_bar + 80


def test_scan_symbol_tf_returns_no_events_on_flat():
    df = make_flat_ohlcv(n_bars=300, base_price=100.0)
    events = scan_symbol_tf(df, symbol="FLATUSDT", tf="1h")
    assert events == []


def test_scan_symbol_tf_event_has_distance_features_and_forward_returns():
    df, _ = make_uptrend_with_formation(
        n_bars=300, formation_at_bar=120, base_price=100.0
    )
    events = scan_symbol_tf(df, symbol="X", tf="1h", forward_horizons=[5, 10, 20])
    assert events
    e = events[0]
    assert e.distance_to_ma5_pct  is not None
    assert e.distance_to_ma8_pct  is not None
    assert e.distance_to_ema21_pct is not None
    assert e.distance_to_ma55_pct is not None
    assert set(e.forward_returns.keys()) == {5, 10, 20}
    assert set(e.forward_returns_post_fee.keys()) == {5, 10, 20}


def test_scan_symbol_tf_skips_events_without_enough_history():
    df, _ = make_uptrend_with_formation(
        n_bars=80, formation_at_bar=30, base_price=100.0
    )
    # MA55 needs 55 bars of history → events before bar 54 cannot fire.
    events = scan_symbol_tf(df, symbol="X", tf="1h")
    for e in events:
        assert e.formation_bar_idx >= 54


def test_scan_universe_returns_events_for_each_symbol_tf_pair(tmp_path, monkeypatch):
    # Patch Bitget fetch so missing-cache pairs don't hit the real network.
    from backtests.ma_ribbon_ema21 import data_loader as dl
    monkeypatch.setattr(
        dl, "fetch_ohlcv_from_bitget",
        lambda symbol, tf, cfg: pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        ),
    )
    df_a, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=100, tf="1h")
    df_b, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=120, tf="4h")
    (tmp_path / "AAAUSDT_1h.csv").write_text(df_a.to_csv(index=False))
    (tmp_path / "BBBUSDT_4h.csv").write_text(df_b.to_csv(index=False))

    cfg = UniverseConfig(
        symbols=["AAAUSDT", "BBBUSDT"],
        timeframes=["1h", "4h"],
        loader=DataLoaderConfig(cache_dir=str(tmp_path)),
    )
    events = scan_universe(cfg)
    syms_tfs = {(e.symbol, e.tf) for e in events}
    # AAA only has 1h CSV, BBB only has 4h CSV; other pairs return empty (mocked).
    assert ("AAAUSDT", "1h") in syms_tfs
    assert ("BBBUSDT", "4h") in syms_tfs


def test_scan_universe_handles_missing_symbol_gracefully(tmp_path, monkeypatch):
    # Patch network fetch to return empty so missing symbol doesn't reach Bitget.
    from backtests.ma_ribbon_ema21 import data_loader as dl
    monkeypatch.setattr(dl, "fetch_ohlcv_from_bitget",
                        lambda symbol, tf, cfg: pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]))
    cfg = UniverseConfig(
        symbols=["DOESNOTEXIST"],
        timeframes=["1h"],
        loader=DataLoaderConfig(cache_dir=str(tmp_path)),
    )
    events = scan_universe(cfg)
    assert events == []
