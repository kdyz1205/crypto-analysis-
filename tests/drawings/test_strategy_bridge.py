from datetime import datetime, timedelta, timezone

import pandas as pd

from server.drawings.strategy_bridge import augment_snapshot_with_manual_signals, build_manual_signal_lines
from server.drawings.types import ManualTrendline
from server.strategy import StrategyConfig, build_latest_snapshot
from server.strategy.types import StrategySignal, stable_id


def _sample_candles_df() -> pd.DataFrame:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for index in range(6):
        ts = int((start + timedelta(hours=index)).timestamp())
        rows.append(
            {
                "timestamp": ts,
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1000.0 + (index * 10),
            }
        )
    return pd.DataFrame(rows)


def _manual_line(*, manual_line_id: str, symbol: str, comparison_status: str, override_mode: str, nearest_auto_line_id: str | None) -> ManualTrendline:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    return ManualTrendline(
        manual_line_id=manual_line_id,
        symbol=symbol,
        timeframe="1h",
        side="resistance",
        source="manual",
        t_start=int(start.timestamp()),
        t_end=int((start + timedelta(hours=5)).timestamp()),
        price_start=105.0,
        price_end=110.0,
        extend_left=False,
        extend_right=True,
        locked=False,
        label=manual_line_id,
        notes="",
        comparison_status=comparison_status,
        override_mode=override_mode,
        nearest_auto_line_id=nearest_auto_line_id,
        slope_diff=0.0,
        projected_price_diff=0.0,
        overlap_ratio=1.0,
        created_at=int(start.timestamp()),
        updated_at=int((start + timedelta(minutes=5)).timestamp()),
    )


def test_build_manual_signal_lines_marks_manual_and_hybrid_sources() -> None:
    candles_df = _sample_candles_df()
    cfg = StrategyConfig()
    drawings = [
        _manual_line(
            manual_line_id="manual-only",
            symbol="BTCUSDT",
            comparison_status="conflicts_auto",
            override_mode="strategy_input_enabled",
            nearest_auto_line_id=None,
        ),
        _manual_line(
            manual_line_id="hybrid-line",
            symbol="BTCUSDT",
            comparison_status="supports_auto",
            override_mode="promote_to_active",
            nearest_auto_line_id="auto-line-1",
        ),
    ]

    lines = build_manual_signal_lines(drawings, candles_df, cfg, symbol="BTCUSDT", timeframe="1h")

    assert len(lines) == 2
    sources = {line.line_id: line.source for line in lines}
    assert any(source == "manual" for source in sources.values())
    assert any(source == "hybrid" for source in sources.values())
    assert all(line.state == "confirmed" for line in lines)


def test_augment_snapshot_with_manual_signals_preserves_manual_source(monkeypatch) -> None:
    candles_df = _sample_candles_df()
    cfg = StrategyConfig()
    snapshot = build_latest_snapshot(candles_df, cfg, symbol="BTCUSDT", timeframe="1h")
    drawing = _manual_line(
        manual_line_id="manual-signal",
        symbol="BTCUSDT",
        comparison_status="conflicts_auto",
        override_mode="strategy_input_enabled",
        nearest_auto_line_id=None,
    )

    def fake_pre_limit(candles, lines, config):
        manual_lines = [line for line in lines if line.source != "auto"]
        if not manual_lines:
            return []
        line = manual_lines[0]
        signal_id = stable_id(line.symbol, line.timeframe, line.line_id, candles.iloc[-1]["timestamp"], "MANUAL_TEST")
        return [
            StrategySignal(
                signal_id=signal_id,
                line_id=line.line_id,
                symbol=line.symbol,
                timeframe=line.timeframe,
                source=line.source,
                signal_type="MANUAL_TEST",
                direction="short",
                trigger_mode="pre_limit",
                timestamp=candles.iloc[-1]["timestamp"],
                trigger_bar_index=len(candles) - 1,
                score=0.95,
                priority_rank=None,
                entry_price=float(candles.iloc[-1]["close"]),
                stop_price=float(candles.iloc[-1]["high"]),
                tp_price=float(candles.iloc[-1]["close"]) - 2.0,
                risk_reward=2.0,
                confirming_touch_count=line.confirming_touch_count,
                bars_since_last_confirming_touch=line.bars_since_last_confirming_touch,
                distance_to_line=0.0,
                line_side=line.side,
                reason_code="manual_test",
                factor_components={},
            )
        ]

    monkeypatch.setattr("server.drawings.strategy_bridge.generate_pre_limit_signals", fake_pre_limit)
    monkeypatch.setattr("server.drawings.strategy_bridge.generate_rejection_signals", lambda *args, **kwargs: [])
    monkeypatch.setattr("server.drawings.strategy_bridge.generate_failed_breakout_signals", lambda *args, **kwargs: [])

    augmented = augment_snapshot_with_manual_signals(
        snapshot,
        candles_df,
        cfg,
        symbol="BTCUSDT",
        timeframe="1h",
        drawings=[drawing],
    )

    assert len(augmented.signals) == 1
    assert augmented.signals[0].source == "manual"
    assert augmented.signal_states[0].state == "active"
