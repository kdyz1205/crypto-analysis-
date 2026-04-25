from __future__ import annotations
import json
import pytest
from pathlib import Path
from server.strategy.ma_ribbon_auto_state import (
    AutoState, AutoStateConfig, load_state, save_state,
    StateCorruptError, current_ramp_cap_pct,
)


def test_default_state_has_safe_disabled_values():
    s = AutoState.default()
    assert s.enabled is False
    assert s.halted is False
    assert s.locked_until_utc is None
    assert s.first_enabled_at_utc is None
    assert s.config.layer_risk_pct["LV1"] == 0.001
    assert s.config.layer_risk_pct["LV4"] == 0.02
    assert s.config.dd_halt_pct == 0.15
    assert s.config.max_concurrent_orders == 25


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "state.json"
    s = AutoState.default()
    s.enabled = True
    s.first_enabled_at_utc = 1_700_000_000
    save_state(s, path=path)
    loaded = load_state(path=path)
    assert loaded.enabled is True
    assert loaded.first_enabled_at_utc == 1_700_000_000


def test_save_is_atomic_via_tmp_then_rename(tmp_path):
    path = tmp_path / "state.json"
    save_state(AutoState.default(), path=path)
    assert path.exists()
    assert not (tmp_path / "state.json.tmp").exists()


def test_load_missing_file_returns_default(tmp_path):
    path = tmp_path / "nope.json"
    s = load_state(path=path)
    assert s.enabled is False
    assert s.first_enabled_at_utc is None


def test_load_corrupt_json_raises_state_corrupt_error(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not valid json {{{")
    with pytest.raises(StateCorruptError):
        load_state(path=path)


def test_load_schema_mismatch_raises_state_corrupt_error(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"unrecognised": "fields"}))
    with pytest.raises(StateCorruptError):
        load_state(path=path)


def test_current_ramp_cap_progression():
    s = AutoState.default()
    s.first_enabled_at_utc = 1_700_000_000
    assert current_ramp_cap_pct(s, now_utc=1_700_000_000) == pytest.approx(0.02)
    assert current_ramp_cap_pct(s, now_utc=1_700_000_000 + 86400) == pytest.approx(0.03)
    assert current_ramp_cap_pct(s, now_utc=1_700_000_000 + 86400 * 13) == pytest.approx(0.15)
    assert current_ramp_cap_pct(s, now_utc=1_700_000_000 + 86400 * 30) == pytest.approx(0.15)


def test_history_append_creates_jsonl(tmp_path):
    path = tmp_path / "state.json"
    history = tmp_path / "state_history.jsonl"
    s = AutoState.default()
    save_state(s, path=path, history_path=history)
    s.enabled = True
    save_state(s, path=path, history_path=history)
    lines = history.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["enabled"] is True
