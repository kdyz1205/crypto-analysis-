"""End-to-end integration tests for MA-ribbon auto-execution scanner tick.

These exercise the full `tick()` pipeline with mocked I/O:
  - state load/save
  - universe fetch (mocked)
  - signal detection on synthetic uptrend fixture
  - LV1 spawn + LV2/LV3/LV4 pending registration
  - guard-rail regressions (no _submit_paper on ribbon lineage,
    halted state never fetches, day-0 ramp cap blocks oversize spawns)

Patch targets are pinned to the actual scanner module (not the spec's
suggested names) to avoid drift if helpers get renamed.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock

from server.strategy.ma_ribbon_auto_state import AutoState, save_state
from server.strategy.ma_ribbon_auto_scanner import tick
from backtests.ma_ribbon_ema21.tests.fixtures import make_uptrend_with_formation


@pytest.mark.asyncio
async def test_enabled_scanner_spawns_lv1_and_registers_higher_layers(tmp_path, monkeypatch):
    """Enabled state + synthetic uptrend → LV1 spawned at least once."""
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._STATE_PATH_DEFAULT",
        state_path,
    )
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._HISTORY_PATH_DEFAULT",
        tmp_path / "history.jsonl",
    )
    s = AutoState.default()
    s.enabled = True
    s.first_enabled_at_utc = 1_700_000_000
    s.config.strategy_capital_usd = 10_000.0
    save_state(s, path=state_path, history_path=tmp_path / "history.jsonl")

    df, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=120, base_price=100.0)
    fake_data = {
        ("AAAUSDT", "5m"): df,
        ("AAAUSDT", "15m"): df,
        ("AAAUSDT", "1h"): df,
        ("AAAUSDT", "4h"): df,
    }

    with patch(
        "server.strategy.ma_ribbon_auto_scanner._fetch_universe_data",
        new=AsyncMock(return_value=(["AAAUSDT"], fake_data)),
    ), patch(
        "server.strategy.ma_ribbon_auto_scanner._spawn_layer",
        new=AsyncMock(),
    ) as mock_spawn:
        await tick()

    # At least one spawn should have happened (LV1 for the formed uptrend).
    assert mock_spawn.call_count >= 1, "expected at least one LV1 spawn on a formed uptrend"

    # First spawn must be LV1 — higher layers fire only once their ETA passes
    # AND ribbon is still aligned. Inspect args robustly across positional/kw.
    first_call = mock_spawn.call_args_list[0]
    layer_arg = first_call.kwargs.get("layer")
    if layer_arg is None and len(first_call.args) >= 3:
        # _spawn_layer(state, sig, layer=..., now_utc=...) — layer is index 2
        layer_arg = first_call.args[2]
    assert layer_arg == "LV1", f"expected first spawn to be LV1, got {layer_arg!r}"


@pytest.mark.asyncio
async def test_halted_state_skips_all_spawning(tmp_path, monkeypatch):
    """halted=True must short-circuit before any fetch/spawn happens."""
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._STATE_PATH_DEFAULT",
        state_path,
    )
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._HISTORY_PATH_DEFAULT",
        tmp_path / "history.jsonl",
    )
    s = AutoState.default()
    s.enabled = True
    s.halted = True
    s.halt_reason = "test"
    save_state(s, path=state_path, history_path=tmp_path / "history.jsonl")

    with patch(
        "server.strategy.ma_ribbon_auto_scanner._fetch_universe_data",
        new=AsyncMock(),
    ) as mock_fetch, patch(
        "server.strategy.ma_ribbon_auto_scanner._spawn_layer",
        new=AsyncMock(),
    ) as mock_spawn:
        await tick()

    mock_fetch.assert_not_called()
    mock_spawn.assert_not_called()


@pytest.mark.asyncio
async def test_no_paper_submit_called_for_ribbon(tmp_path, monkeypatch):
    """Regression: _submit_paper must NEVER be invoked from the ribbon path.

    The watcher's _submit_paper is for the manual-line PAPER mode. Ribbon
    is live-only — paper validation happens on the separate backtest panel,
    not via the live watcher. Any code path that invokes _submit_paper for
    a ma_ribbon-lineage cond is a bug.
    """
    from server.conditionals import watcher

    paper_calls: list[tuple] = []
    if hasattr(watcher, "_submit_paper"):
        async def captured_paper(*args, **kwargs):
            paper_calls.append((args, kwargs))
        monkeypatch.setattr(watcher, "_submit_paper", captured_paper)

    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._STATE_PATH_DEFAULT",
        state_path,
    )
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._HISTORY_PATH_DEFAULT",
        tmp_path / "history.jsonl",
    )
    s = AutoState.default()
    s.enabled = True
    s.first_enabled_at_utc = 1_700_000_000
    s.config.strategy_capital_usd = 10_000.0
    save_state(s, path=state_path, history_path=tmp_path / "history.jsonl")

    df, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=120, base_price=100.0)
    with patch(
        "server.strategy.ma_ribbon_auto_scanner._fetch_universe_data",
        new=AsyncMock(return_value=(["AAAUSDT"], {("AAAUSDT", "5m"): df})),
    ), patch(
        "server.strategy.ma_ribbon_auto_scanner._spawn_layer",
        new=AsyncMock(),
    ):
        await tick()

    assert paper_calls == [], (
        f"_submit_paper was invoked from ribbon tick path "
        f"({len(paper_calls)} times) — this would treat ribbon as paper mode."
    )


@pytest.mark.asyncio
async def test_ramp_day0_caps_total_risk_to_2pct(tmp_path, monkeypatch):
    """Day-0 ramp cap: 2 % total. With 1.8 % open already, only the smallest
    layer (LV1 = 0.1 %) can fit; subsequent spawns same tick must be blocked.

    NOTE: the real _spawn_layer mutates state.ledger.open_positions so the
    gate engages on subsequent iterations. The mock here MUST mirror that
    side-effect, otherwise every iteration sees a stale 1.8 % open and the
    gate never trips.
    """
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._STATE_PATH_DEFAULT",
        state_path,
    )
    monkeypatch.setattr(
        "server.strategy.ma_ribbon_auto_state._HISTORY_PATH_DEFAULT",
        tmp_path / "history.jsonl",
    )
    # Day-0 is current wall-clock minus a few minutes, so the ramp cap is 2 %.
    # `first_enabled_at_utc` must be < now (else ramp returns base cap), and
    # within the same day for day 0 to apply.
    import time as _time
    now = int(_time.time())
    s = AutoState.default()
    s.enabled = True
    s.first_enabled_at_utc = now - 60  # ~1 min ago → day 0 → 2 % cap
    s.config.strategy_capital_usd = 10_000.0
    s.ledger.open_positions = [{
        "symbol": "X",
        "layer": "LV1",
        "risk_pct": 0.018,
        "unrealized_pnl_usd": 0.0,
    }]
    save_state(s, path=state_path, history_path=tmp_path / "history.jsonl")

    df, _ = make_uptrend_with_formation(n_bars=300, formation_at_bar=120, base_price=100.0)
    fake_data = {("AAAUSDT", tf): df for tf in ["5m", "15m", "1h", "4h"]}

    async def fake_spawn(state, sig, layer, now_utc):
        # Mirror the real _spawn_layer's ledger-append side effect so the
        # ramp gate engages between iterations of this same tick.
        rp = state.config.layer_risk_pct[layer]
        state.ledger.open_positions.append({
            "signal_id": sig.signal_id, "layer": layer, "tf": sig.tf,
            "symbol": sig.symbol, "direction": sig.direction,
            "risk_pct": rp,
            "unrealized_pnl_usd": 0.0,
            "spawned_at_utc": now_utc,
            "conditional_id": "cond_test",
        })

    mock_spawn = AsyncMock(side_effect=fake_spawn)
    with patch(
        "server.strategy.ma_ribbon_auto_scanner._fetch_universe_data",
        new=AsyncMock(return_value=(["AAAUSDT"], fake_data)),
    ), patch(
        "server.strategy.ma_ribbon_auto_scanner._spawn_layer",
        new=mock_spawn,
    ):
        await tick()

    # 0.018 existing + 0.001 (LV1) = 0.019 < 0.02 → first LV1 spawn passes.
    # 0.019 + 0.001 (next LV1) = 0.020 → exactly at cap, gate uses strict >
    # so a second LV1 still passes. 0.020 + 0.001 = 0.021 > 0.02 → blocked.
    # Conclusion: on a 1.8 %-open day-0 ledger, the cap admits at MOST 2
    # spawns (each LV1 = 0.1 %) before the > comparison trips.
    assert mock_spawn.call_count <= 2, (
        f"day-0 ramp cap (2 %) violated: {mock_spawn.call_count} spawns "
        f"with 1.8 % already open (each LV1=0.1 %, so >2 means cap leak)"
    )
