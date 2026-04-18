from __future__ import annotations

import pandas as pd

from server.strategy.drawing_outcome_labeler import ReplayConfig, simulate_line_outcome


def test_simulate_line_outcome_fills_and_hits_tp() -> None:
    drawing = {
        "manual_line_id": "line-1",
        "symbol": "TESTUSDT",
        "timeframe": "1h",
        "side": "support",
        "t_start": 0,
        "t_end": 3600,
        "price_start": 100.0,
        "price_end": 100.0,
        "ts": 1,
    }
    df = pd.DataFrame([
        {"timestamp": 60, "open": 101.0, "high": 101.0, "low": 100.05, "close": 100.2, "volume": 1.0},
    ])

    outcome = simulate_line_outcome(drawing, df, ReplayConfig(buffer_pct=0.001, rr=2.0))

    assert outcome["status"] == "closed"
    assert outcome["exit_reason"] == "tp"
    assert outcome["filled"] is True
    assert outcome["realized_r"] == 2.0
    assert outcome["label_trade_win"] == 1


def test_simulate_line_outcome_moves_stop_on_trade_timeframe() -> None:
    drawing = {
        "manual_line_id": "line-2",
        "symbol": "TESTUSDT",
        "timeframe": "1h",
        "side": "support",
        "t_start": 0,
        "t_end": 3600,
        "price_start": 100.0,
        "price_end": 101.0,
        "ts": 1,
    }
    df = pd.DataFrame([
        {"timestamp": 60, "open": 100.4, "high": 100.4, "low": 100.05, "close": 100.2, "volume": 1.0},
        {"timestamp": 120, "open": 100.2, "high": 100.5, "low": 100.2, "close": 100.4, "volume": 1.0},
        {"timestamp": 3600, "open": 101.1, "high": 101.2, "low": 100.95, "close": 101.0, "volume": 1.0},
    ])

    outcome = simulate_line_outcome(
        drawing,
        df,
        ReplayConfig(buffer_pct=0.001, rr=50.0, trailing_enabled=True),
    )

    assert outcome["status"] == "closed"
    assert outcome["exit_reason"] == "stop"
    assert outcome["walking_stop_updates"] == 1
    assert outcome["realized_r"] > 0
