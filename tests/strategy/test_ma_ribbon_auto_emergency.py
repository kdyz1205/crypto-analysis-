from __future__ import annotations
import pytest
from unittest.mock import patch, AsyncMock
from server.strategy.ma_ribbon_auto_state import AutoState
from server.strategy.ma_ribbon_auto_scanner import (
    emergency_stop, _LOCK_DURATION_SECONDS,
)


@pytest.mark.asyncio
async def test_emergency_stop_sets_24h_lock():
    s = AutoState.default()
    s.enabled = True
    with patch("server.strategy.ma_ribbon_auto_scanner.flatten_all_ribbon_positions",
               new=AsyncMock(return_value={"cancelled": 0, "closed": 0})):
        await emergency_stop(s, now_utc=1_700_000_000, reason="user click")
    assert s.locked_until_utc == 1_700_000_000 + _LOCK_DURATION_SECONDS
    assert s.halted is True


@pytest.mark.asyncio
async def test_emergency_stop_calls_flatten():
    s = AutoState.default()
    flatten = AsyncMock(return_value={"cancelled": 3, "closed": 2})
    with patch("server.strategy.ma_ribbon_auto_scanner.flatten_all_ribbon_positions", flatten):
        await emergency_stop(s, now_utc=1_700_000_000, reason="manual")
    flatten.assert_called_once()


@pytest.mark.asyncio
async def test_emergency_stop_clears_pending_signals():
    s = AutoState.default()
    s.pending_signals = [{
        "signal_id": "x",
        "pending_layers": [{"layer": "LV2", "tf": "15m",
                            "trigger_at_bar_close_after_ts": 1}]
    }]
    with patch("server.strategy.ma_ribbon_auto_scanner.flatten_all_ribbon_positions",
               new=AsyncMock(return_value={"cancelled": 0, "closed": 0})):
        await emergency_stop(s, now_utc=1_700_000_000, reason="x")
    assert s.pending_signals == []


@pytest.mark.asyncio
async def test_emergency_stop_idempotent():
    s = AutoState.default()
    flatten = AsyncMock(return_value={"cancelled": 0, "closed": 0})
    with patch("server.strategy.ma_ribbon_auto_scanner.flatten_all_ribbon_positions", flatten):
        await emergency_stop(s, now_utc=1_700_000_000, reason="x")
        await emergency_stop(s, now_utc=1_700_000_001, reason="x")
    assert flatten.call_count == 2
    assert s.halted is True


@pytest.mark.asyncio
async def test_emergency_stop_records_to_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_scanner._EMERGENCY_LOG_PATH",
        tmp_path / "ma_ribbon_emergency_stop.log",
    )
    s = AutoState.default()
    s.ledger.open_positions = [{"symbol": "BTCUSDT", "layer": "LV1"}]
    with patch("server.strategy.ma_ribbon_auto_scanner.flatten_all_ribbon_positions",
               new=AsyncMock(return_value={"cancelled": 1, "closed": 1})):
        await emergency_stop(s, now_utc=1_700_000_000, reason="user click")
    log_file = tmp_path / "ma_ribbon_emergency_stop.log"
    assert log_file.exists()
    text = log_file.read_text()
    assert "1700000000" in text
    assert "user click" in text
