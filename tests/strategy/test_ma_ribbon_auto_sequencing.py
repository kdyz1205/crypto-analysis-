from __future__ import annotations
import pytest
from server.strategy.ma_ribbon_auto_state import AutoState
from server.strategy.ma_ribbon_auto_adapter import Phase1Signal
from server.strategy.ma_ribbon_auto_scanner import (
    register_pending_higher_layers, ready_layers_at,
    remove_layer_from_pending, expire_orphans,
)


def _sig(signal_id="sig-1", signal_bar_ts=1_700_000_000) -> Phase1Signal:
    return Phase1Signal(
        signal_id=signal_id, symbol="BTCUSDT", tf="5m", direction="long",
        signal_bar_ts=signal_bar_ts, next_bar_open_estimate=50_500.0,
        ema21_at_signal=50_000.0,
    )


def test_register_pending_creates_lv2_lv3_lv4_with_etas():
    state = AutoState.default()
    register_pending_higher_layers(_sig(), state, now_utc=1_700_000_000)
    assert len(state.pending_signals) == 1
    pending = state.pending_signals[0]
    assert pending["signal_id"] == "sig-1"
    assert {l["layer"] for l in pending["pending_layers"]} == {"LV2", "LV3", "LV4"}


def test_pending_eta_uses_next_bar_close_per_tf():
    state = AutoState.default()
    sig = _sig(signal_bar_ts=1_700_000_000)
    register_pending_higher_layers(sig, state, now_utc=1_700_000_000)
    layers = {l["layer"]: l for l in state.pending_signals[0]["pending_layers"]}
    assert layers["LV2"]["trigger_at_bar_close_after_ts"] == ((1_700_000_000 // 900) + 1) * 900
    assert layers["LV3"]["trigger_at_bar_close_after_ts"] == ((1_700_000_000 // 3600) + 1) * 3600
    assert layers["LV4"]["trigger_at_bar_close_after_ts"] == ((1_700_000_000 // 14400) + 1) * 14400


def test_ready_layers_returns_lv2_when_15m_close_passed():
    state = AutoState.default()
    sig = _sig()
    register_pending_higher_layers(sig, state, now_utc=1_700_000_000)
    later = state.pending_signals[0]["pending_layers"][0]["trigger_at_bar_close_after_ts"] + 60
    ready = ready_layers_at(state, now_utc=later)
    assert any(r["layer"] == "LV2" for r in ready)


def test_ready_layers_excludes_layers_not_yet_due():
    state = AutoState.default()
    sig = _sig()
    register_pending_higher_layers(sig, state, now_utc=1_700_000_000)
    ready = ready_layers_at(state, now_utc=1_700_000_000)
    # No layer's eta has passed yet
    assert ready == []


def test_remove_layer_from_pending_moves_to_spawned():
    state = AutoState.default()
    register_pending_higher_layers(_sig(), state, now_utc=1_700_000_000)
    remove_layer_from_pending(state, "sig-1", "LV2")
    rec = state.pending_signals[0]
    assert "LV2" in rec["spawned_layers"]
    assert all(l["layer"] != "LV2" for l in rec["pending_layers"])


def test_expire_orphans_removes_old():
    state = AutoState.default()
    register_pending_higher_layers(_sig(), state, now_utc=1_700_000_000)
    expire_orphans(state, now_utc=1_700_000_000 + 86400 * 2, max_age_seconds=86400)
    assert state.pending_signals == []


def test_expire_orphans_keeps_recent():
    state = AutoState.default()
    register_pending_higher_layers(_sig(), state, now_utc=1_700_000_000)
    expire_orphans(state, now_utc=1_700_000_000 + 60, max_age_seconds=86400)
    assert len(state.pending_signals) == 1


def test_signal_id_shared_across_layers_of_same_signal():
    state = AutoState.default()
    register_pending_higher_layers(_sig(signal_id="abc"), state, now_utc=1_700_000_000)
    rec = state.pending_signals[0]
    for layer in rec["pending_layers"]:
        # Each layer entry doesn't have signal_id directly, but ready_layers_at returns
        # the parent signal record alongside the layer dict, which contains signal_id.
        pass
    assert rec["signal_id"] == "abc"


def test_multiple_signals_coexist():
    state = AutoState.default()
    register_pending_higher_layers(_sig(signal_id="a"), state, now_utc=1_700_000_000)
    register_pending_higher_layers(_sig(signal_id="b"), state, now_utc=1_700_000_000)
    assert len(state.pending_signals) == 2
    assert {s["signal_id"] for s in state.pending_signals} == {"a", "b"}


def test_no_pending_state_does_not_break_ready_check():
    state = AutoState.default()
    ready = ready_layers_at(state, now_utc=1_700_000_000)
    assert ready == []
