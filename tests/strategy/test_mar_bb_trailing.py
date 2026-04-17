import time

import pytest

from server.strategy import mar_bb_runner as runner


class _FakeSLAdapter:
    instances: list["_FakeSLAdapter"] = []
    actual_sl: float | None = None

    def __init__(self) -> None:
        self.updates: list[tuple[str, str, float | None, float | None, str]] = []
        _FakeSLAdapter.instances.append(self)

    async def get_position_sl_trigger_price(self, symbol, hold_side, mode="live", *, entry_price=None):
        return type(self).actual_sl

    async def update_position_sl_tp(self, symbol, hold_side, new_sl=None, new_tp=None, mode="live"):
        self.updates.append((symbol, hold_side, new_sl, new_tp, mode))
        type(self).actual_sl = float(new_sl)
        return {
            "ok": True,
            "new_sl": str(new_sl),
            "actual_sl_after": float(new_sl),
            "sl_verified": True,
            "updates": ["sl_ok"],
            "cancelled_order_ids": ["old-sl"],
        }


@pytest.mark.asyncio
async def test_trailing_sl_moves_once_per_new_bar_and_never_touches_tp(monkeypatch):
    _FakeSLAdapter.instances.clear()
    _FakeSLAdapter.actual_sl = 100.0
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeSLAdapter)

    async def fake_positions():
        return ([{
            "symbol": "LINKUSDT",
            "holdSide": "long",
            "total": "1",
            "averageOpenPrice": "100",
        }], True)

    monkeypatch.setattr(runner, "_get_bitget_positions", fake_positions)
    runner._trendline_params.clear()
    runner.register_trendline_params(
        "LINKUSDT",
        slope=1.0,
        intercept=90.0,
        entry_bar=10,
        entry_price=100.0,
        side="long",
        tf="1h",
        created_ts=int(time.time()) - 3600,
        tp_price=130.0,
        last_sl_set=100.0,
    )

    for bars_since in (1, 2, 3):
        runner._trendline_params["LINKUSDT"]["opened_ts"] = int(time.time()) - bars_since * 3600
        updated = await runner._update_trailing_stops({"mode": "demo"})
        assert updated == 1
        assert runner._trendline_params["LINKUSDT"]["last_update_bar"] == bars_since

    updates = [update for adapter in _FakeSLAdapter.instances for update in adapter.updates]
    assert [u[2] for u in updates] == [101.0, 102.0, 103.0]
    assert all(u[3] is None for u in updates)
    assert runner._trendline_params["LINKUSDT"]["last_sl_set"] == 103.0


def test_register_trendline_params_restores_initial_sl_and_position_open_time():
    runner._trendline_params.clear()

    runner.register_trendline_params(
        "LINKUSDT",
        slope=0.5,
        intercept=95.0,
        entry_bar=10,
        entry_price=100.0,
        side="long",
        tf="1h",
        created_ts=1_776_390_000,
        tp_price=130.0,
        last_sl_set=99.5,
    )

    params = runner._trendline_params["LINKUSDT"]
    assert params["opened_ts"] == 1_776_390_000
    assert params["last_sl_set"] == 99.5
    assert params["tp_price"] == 130.0


def test_trailing_projection_can_use_timestamp_reference():
    runner._trendline_params.clear()
    ref_ts = int(time.time()) - 7200

    runner.register_trendline_params(
        "LINKUSDT",
        slope=1.0,
        intercept=0.0,
        entry_bar=999,
        entry_price=100.0,
        side="long",
        tf="1h",
        created_ts=ref_ts,
        tp_price=130.0,
        last_sl_set=100.0,
        line_ref_ts=ref_ts,
        line_ref_price=100.0,
    )

    assert runner._calc_trendline_trailing_sl("LINKUSDT", 999, now_ts=ref_ts + 7200) == 102.0
