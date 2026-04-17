import json
import time
from types import SimpleNamespace

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


@pytest.mark.asyncio
async def test_trailing_sl_moves_two_tick_short_change(monkeypatch):
    _FakeSLAdapter.instances.clear()
    _FakeSLAdapter.actual_sl = 200.98
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeSLAdapter)
    now_ts = int(time.time())

    async def fake_positions():
        return ([{
            "symbol": "NVDAUSDT",
            "holdSide": "short",
            "total": "0.15",
            "averageOpenPrice": "200.91",
        }], True)

    monkeypatch.setattr(runner, "_get_bitget_positions", fake_positions)
    runner._trendline_params.clear()
    runner.register_trendline_params(
        "NVDAUSDT",
        slope=-0.01,
        intercept=0.0,
        entry_bar=0,
        entry_price=200.91,
        side="short",
        tf="5m",
        created_ts=now_ts - 600,
        tp_price=199.38,
        last_sl_set=200.98,
        line_ref_ts=now_ts - 600,
        line_ref_price=200.98,
    )

    updated = await runner._update_trailing_stops({"mode": "demo"})

    updates = [update for adapter in _FakeSLAdapter.instances for update in adapter.updates]
    assert updated == 1
    assert updates[-1][0] == "NVDAUSDT"
    assert updates[-1][1] == "short"
    assert updates[-1][2] == pytest.approx(200.96)


def test_select_active_order_for_position_ignores_stale_and_prefers_latest_placed():
    orders = [
        SimpleNamespace(symbol="ENAUSDT", status="stale", last_updated_ts=999, created_ts=999),
        SimpleNamespace(symbol="ENAUSDT", status="filled", last_updated_ts=100, created_ts=100),
        SimpleNamespace(symbol="ENAUSDT", status="placed", last_updated_ts=90, created_ts=90),
        SimpleNamespace(symbol="BTCUSDT", status="placed", last_updated_ts=200, created_ts=200),
    ]

    selected = runner._select_active_order_for_position(orders, "ENAUSDT")

    assert selected is orders[2]


def test_select_active_order_for_position_restores_latest_filled_after_restart():
    orders = [
        SimpleNamespace(symbol="ENAUSDT", status="filled", last_updated_ts=100, created_ts=100),
        SimpleNamespace(symbol="ENAUSDT", status="filled", last_updated_ts=200, created_ts=200),
        SimpleNamespace(symbol="ENAUSDT", status="stale", last_updated_ts=300, created_ts=300),
    ]

    selected = runner._select_active_order_for_position(orders, "ENAUSDT")

    assert selected is orders[1]


def test_daily_dd_baseline_persists_across_restart(monkeypatch, tmp_path):
    risk_file = tmp_path / "mar_bb_daily_risk.json"
    monkeypatch.setattr(runner, "_daily_risk_file", lambda: risk_file)
    monkeypatch.setattr(runner, "_utc_day", lambda: "2026-04-17")
    runner._daily_equity_start = 0.0
    runner._daily_date = ""
    runner._state.daily_risk = None
    cfg = {"daily_dd_tiers": [(0, 0.50)]}

    halted, dd, limit = runner._check_daily_dd(1000.0, cfg)

    assert (halted, dd, limit) == (False, 0.0, 0.50)
    saved = json.loads(risk_file.read_text(encoding="utf-8"))
    assert saved["date"] == "2026-04-17"
    assert saved["equity_start"] == 1000.0

    # Simulate a process restart: globals are gone but the file remains.
    runner._daily_equity_start = 0.0
    runner._daily_date = ""

    halted, dd, limit = runner._check_daily_dd(800.0, cfg)

    assert halted is False
    assert dd == pytest.approx(0.20)
    assert limit == 0.50
    saved = json.loads(risk_file.read_text(encoding="utf-8"))
    assert saved["equity_start"] == 1000.0
    assert saved["last_equity"] == 800.0
    assert saved["last_dd_pct"] == pytest.approx(0.20)


def test_daily_dd_halts_after_restart_when_persisted_limit_is_hit(monkeypatch, tmp_path):
    risk_file = tmp_path / "mar_bb_daily_risk.json"
    risk_file.write_text(json.dumps({
        "date": "2026-04-17",
        "equity_start": 1000.0,
        "last_equity": 1000.0,
        "last_dd_pct": 0.0,
        "limit_pct": 0.10,
        "halted": False,
        "updated_ts": 1,
    }), encoding="utf-8")
    monkeypatch.setattr(runner, "_daily_risk_file", lambda: risk_file)
    monkeypatch.setattr(runner, "_utc_day", lambda: "2026-04-17")
    runner._daily_equity_start = 0.0
    runner._daily_date = ""
    cfg = {"daily_dd_tiers": [(0, 0.10)]}

    halted, dd, limit = runner._check_daily_dd(899.0, cfg)

    assert halted is True
    assert dd == pytest.approx(0.101)
    assert limit == 0.10
    saved = json.loads(risk_file.read_text(encoding="utf-8"))
    assert saved["halted"] is True
    assert saved["equity_start"] == 1000.0


def test_daily_dd_resets_on_new_utc_day(monkeypatch, tmp_path):
    risk_file = tmp_path / "mar_bb_daily_risk.json"
    risk_file.write_text(json.dumps({
        "date": "2026-04-16",
        "equity_start": 1000.0,
        "last_equity": 700.0,
        "last_dd_pct": 0.30,
        "limit_pct": 0.50,
        "halted": False,
        "updated_ts": 1,
    }), encoding="utf-8")
    monkeypatch.setattr(runner, "_daily_risk_file", lambda: risk_file)
    monkeypatch.setattr(runner, "_utc_day", lambda: "2026-04-17")
    runner._daily_equity_start = 1000.0
    runner._daily_date = "2026-04-16"
    cfg = {"daily_dd_tiers": [(0, 0.50)]}

    halted, dd, limit = runner._check_daily_dd(900.0, cfg)

    assert (halted, dd, limit) == (False, 0.0, 0.50)
    saved = json.loads(risk_file.read_text(encoding="utf-8"))
    assert saved["date"] == "2026-04-17"
    assert saved["equity_start"] == 900.0


@pytest.mark.asyncio
async def test_daily_dd_halt_cancels_managed_trendline_plans(monkeypatch):
    called = {}

    async def fake_get_equity():
        return 500.0

    async def fake_cancel(cfg, *, status):
        called["cfg"] = cfg
        called["status"] = status
        return {"cancelled": 2, "failed": 0, "status": status}

    monkeypatch.setattr(runner, "_get_equity", fake_get_equity)
    monkeypatch.setattr(runner, "_check_daily_dd", lambda equity, cfg: (True, 0.60, 0.50))
    monkeypatch.setattr(runner, "_save_state", lambda: None)
    monkeypatch.setattr(
        "server.strategy.trendline_order_manager.cancel_all_trendline_plan_orders",
        fake_cancel,
    )
    runner._daily_equity_start = 1000.0
    runner._state.config = {
        "top_n": 1,
        "scan_interval_s": 60,
        "timeframes": ["5m"],
        "sizing_mode": "fixed_risk",
        "risk_pct": 0.01,
        "max_concurrent_positions": 1,
        "mode": "demo",
        "leverage": 30,
    }
    before = runner._state.scans_completed

    await runner._do_scan()

    assert called["status"] == "daily_halt"
    assert called["cfg"] is runner._state.config
    assert "DAILY DD HALT" in runner._state.last_error
    assert runner._state.scans_completed == before + 1
