from __future__ import annotations
import pytest
from server.strategy.ma_ribbon_auto_state import AutoState
from server.strategy.ma_ribbon_auto_scanner import can_spawn_layer, GateResult


def _state(open_positions=0, per_sym_risk=0.0, ramp_first_enabled=1_700_000_000,
           realized_pnl=0.0, unrealized=0.0, capital=10_000.0,
           enabled=True, halted=False, locked_until=None) -> AutoState:
    s = AutoState.default()
    s.enabled = enabled
    s.halted = halted
    s.locked_until_utc = locked_until
    s.config.strategy_capital_usd = capital
    s.first_enabled_at_utc = ramp_first_enabled
    s.ledger.realized_pnl_usd_cumulative = realized_pnl
    # If caller passes per_sym_risk or unrealized > 0 but didn't bump
    # open_positions, ensure at least one position exists so those values
    # actually land on the ledger.
    n = open_positions
    if (per_sym_risk != 0.0 or unrealized != 0.0) and n == 0:
        n = 1
    s.ledger.open_positions = [{
        "symbol": "BTCUSDT", "layer": "LVx",
        "risk_pct": per_sym_risk if i == 0 else 0.0,
        "unrealized_pnl_usd": unrealized if i == 0 else 0.0,
    } for i in range(n)]
    return s


def test_concurrent_cap_25_blocks_26th_layer():
    s = _state(open_positions=25)
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14   # ramp fully unlocked
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV1", now_utc=1_700_000_000)
    assert r.ok is False
    assert "concurrent" in r.reason.lower() or "25" in r.reason


def test_per_symbol_cap_blocks_when_existing_risk_plus_new_exceeds_2pct():
    s = _state(per_sym_risk=0.018)
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    s.ledger.open_positions[0]["symbol"] = "BTCUSDT"
    # 0.018 + LV2 risk 0.0025 = 0.0205 > 0.02 → block
    r = can_spawn_layer(s, symbol="BTCUSDT", layer="LV2", now_utc=1_700_000_000)
    assert r.ok is False
    assert "per_symbol" in r.reason.lower() or "symbol" in r.reason.lower()


def test_per_layer_size_blocks_when_config_misedited_above_5pct():
    s = _state()
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    s.config.layer_risk_pct["LV4"] = 0.06
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV4", now_utc=1_700_000_000)
    assert r.ok is False
    assert "layer" in r.reason.lower() or "0.06" in r.reason or "per_layer" in r.reason.lower()


def test_dd_halt_at_negative_15pct_strategy_pnl():
    s = _state(realized_pnl=-1500.01)  # -15.0001% on $10000
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV1", now_utc=1_700_000_000)
    assert r.ok is False
    assert "dd" in r.reason.lower() or "drawdown" in r.reason.lower()


def test_dd_includes_unrealized_pnl():
    s = _state(realized_pnl=0.0, unrealized=-1500.01, open_positions=1)
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV1", now_utc=1_700_000_000)
    assert r.ok is False


def test_ramp_up_day0_caps_at_2pct_total_risk():
    s = _state()
    s.first_enabled_at_utc = 1_700_000_000   # day 0 → 2 % cap
    s.ledger.open_positions = [{"symbol": "X", "layer": "LV1",
                                "risk_pct": 0.018, "unrealized_pnl_usd": 0.0}]
    # 0.018 + LV4 (0.02) = 0.038 > 0.02 → block
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV4", now_utc=1_700_000_000)
    assert r.ok is False
    assert "ramp" in r.reason.lower()


def test_ramp_up_day13_lifts_to_15pct():
    s = _state()
    s.first_enabled_at_utc = 1_700_000_000
    now = 1_700_000_000 + 86400 * 13
    s.ledger.open_positions = [{"symbol": "X", "layer": "LV1",
                                "risk_pct": 0.10, "unrealized_pnl_usd": 0.0}]
    # 10 % + 2 % LV4 = 12 % < 15 % → OK
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV4", now_utc=now)
    assert r.ok is True


def test_halted_state_blocks_all_layers():
    s = _state(halted=True)
    s.halt_reason = "manual"
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV1", now_utc=1_700_000_000)
    assert r.ok is False
    assert "halt" in r.reason.lower()


def test_locked_state_blocks_all_layers():
    s = _state(locked_until=1_700_000_000 + 3600)
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV1", now_utc=1_700_000_000)
    assert r.ok is False
    assert "lock" in r.reason.lower()


def test_disabled_state_blocks_all_layers():
    s = _state(enabled=False)
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV1", now_utc=1_700_000_000)
    assert r.ok is False
    assert "disabled" in r.reason.lower() or "enabled" in r.reason.lower()


def test_zero_capital_blocks_all_layers():
    s = _state(capital=0.0)
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV1", now_utc=1_700_000_000)
    assert r.ok is False
    assert "capital" in r.reason.lower()


def test_unknown_layer_blocks():
    s = _state()
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV99", now_utc=1_700_000_000)
    assert r.ok is False


def test_normal_state_allows_lv1_spawn():
    s = _state()
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    r = can_spawn_layer(s, symbol="ETHUSDT", layer="LV1", now_utc=1_700_000_000)
    assert r.ok is True


def test_per_symbol_cap_calculates_across_multiple_layers():
    """Two existing layers on BTC summing to 1.5 %, plus LV2 (0.25 %) = 1.75 % < 2 % cap → OK."""
    s = _state()
    s.first_enabled_at_utc = 1_700_000_000 - 86400 * 14
    s.ledger.open_positions = [
        {"symbol": "BTCUSDT", "layer": "LV1", "risk_pct": 0.001, "unrealized_pnl_usd": 0.0},
        {"symbol": "BTCUSDT", "layer": "LV2", "risk_pct": 0.0025, "unrealized_pnl_usd": 0.0},
        {"symbol": "BTCUSDT", "layer": "LV3", "risk_pct": 0.005, "unrealized_pnl_usd": 0.0},
    ]
    # Sum so far on BTC = 0.0085. + LV2 again 0.0025 = 0.011 < 0.02 → OK
    r = can_spawn_layer(s, symbol="BTCUSDT", layer="LV2", now_utc=1_700_000_000)
    assert r.ok is True
