from __future__ import annotations
import pytest
from server.strategy.ma_ribbon_auto_state import AutoState
from server.strategy.ma_ribbon_auto_adapter import (
    Phase1Signal, signal_to_conditional, _entry_to_sl_pct,
)


def _state_at_equity(equity_usd: float) -> AutoState:
    s = AutoState.default()
    s.config.strategy_capital_usd = equity_usd
    s.first_enabled_at_utc = 1_700_000_000  # ramp day 0 → cap 2%
    return s


def _bull_signal(symbol="BTCUSDT", tf="5m", ema21=50_000.0, next_bar_open=50_500.0) -> Phase1Signal:
    return Phase1Signal(
        signal_id="sig-abc-123",
        symbol=symbol,
        tf=tf,
        direction="long",
        signal_bar_ts=1_700_000_000,
        next_bar_open_estimate=next_bar_open,
        ema21_at_signal=ema21,
    )


def test_long_lv1_creates_conditional_with_correct_lineage_and_layer():
    sig = _bull_signal()
    cond = signal_to_conditional(sig, layer="LV1", state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    assert cond.lineage == "ma_ribbon"
    assert cond.manual_line_id is None
    assert cond.symbol == "BTCUSDT"
    assert cond.timeframe == "5m"
    assert cond.direction == "long"
    assert cond.config.sl_logic == "ribbon_ema21_trailing"
    assert cond.config.ribbon_meta["signal_id"] == "sig-abc-123"
    assert cond.config.ribbon_meta["layer"] == "LV1"
    assert cond.config.ribbon_meta["reverse_on_stop"] is False


def test_long_lv1_risk_usd_is_0_1pct_of_equity():
    cond = signal_to_conditional(_bull_signal(), layer="LV1",
                                 state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    assert cond.config.risk_usd_target == pytest.approx(10.0)


def test_long_lv4_risk_usd_is_2pct_of_equity():
    cond = signal_to_conditional(_bull_signal(tf="4h", ema21=50_000.0), layer="LV4",
                                 state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    assert cond.config.risk_usd_target == pytest.approx(200.0)


def test_long_initial_sl_below_entry_by_buffer_off_ema21():
    sig = _bull_signal(ema21=50_000.0, next_bar_open=50_500.0)
    cond = signal_to_conditional(sig, layer="LV1",
                                 state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    assert cond.config.ribbon_meta["initial_sl_estimate"] == pytest.approx(49_500.0)


def test_short_initial_sl_above_entry_by_buffer_off_ema21():
    sig = _bull_signal(symbol="SOLUSDT", tf="15m", ema21=100.0, next_bar_open=99.0)
    sig.direction = "short"
    cond = signal_to_conditional(sig, layer="LV2",
                                 state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    assert cond.config.ribbon_meta["initial_sl_estimate"] == pytest.approx(104.0)


def test_qty_notional_is_risk_usd_divided_by_entry_to_sl_pct():
    sig = _bull_signal(ema21=50_000.0, next_bar_open=50_500.0)
    cond = signal_to_conditional(sig, layer="LV1",
                                 state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    assert cond.config.qty_notional_target == pytest.approx(505.0, rel=0.01)


def test_ramp_day_cap_recorded_on_meta():
    cond = signal_to_conditional(_bull_signal(), layer="LV1",
                                 state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    assert cond.config.ribbon_meta["ramp_day_cap_pct_at_spawn"] == pytest.approx(0.02)


def test_lv2_lv3_lv4_use_correct_per_layer_buffer():
    for layer, tf, expected_buffer in [
        ("LV1", "5m", 0.01), ("LV2", "15m", 0.04),
        ("LV3", "1h", 0.07), ("LV4", "4h", 0.10),
    ]:
        sig = _bull_signal(tf=tf)
        cond = signal_to_conditional(sig, layer=layer,
                                     state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
        assert cond.config.ribbon_meta["ribbon_buffer_pct"] == expected_buffer


def test_zero_strategy_capital_raises():
    sig = _bull_signal()
    with pytest.raises(ValueError, match="strategy_capital_usd"):
        signal_to_conditional(sig, layer="LV1",
                              state=_state_at_equity(0.0), now_utc=1_700_000_000)


def test_invalid_layer_raises():
    sig = _bull_signal()
    with pytest.raises(KeyError):
        signal_to_conditional(sig, layer="LV5",
                              state=_state_at_equity(10_000.0), now_utc=1_700_000_000)


def test_signal_id_stable_across_layers_of_same_signal():
    sig = _bull_signal()
    c1 = signal_to_conditional(sig, layer="LV1", state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    c2 = signal_to_conditional(sig, layer="LV2", state=_state_at_equity(10_000.0), now_utc=1_700_000_000)
    assert c1.config.ribbon_meta["signal_id"] == c2.config.ribbon_meta["signal_id"]


def test_entry_to_sl_pct_helper_handles_long_and_short():
    assert _entry_to_sl_pct(entry=100.0, sl=99.0, direction="long") == pytest.approx(0.01)
    assert _entry_to_sl_pct(entry=100.0, sl=101.0, direction="short") == pytest.approx(0.01)
