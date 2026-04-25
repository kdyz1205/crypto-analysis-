from __future__ import annotations
from server.conditionals.types import OrderConfig


def test_default_sl_logic_is_line_buffer_for_existing_callsites():
    cfg = OrderConfig(direction="long")
    assert cfg.sl_logic == "line_buffer"
    assert cfg.ribbon_meta is None


def test_can_construct_with_ribbon_ema21_trailing():
    cfg = OrderConfig(
        direction="long",
        sl_logic="ribbon_ema21_trailing",
        ribbon_meta={"signal_id": "x", "layer": "LV1"},
    )
    assert cfg.sl_logic == "ribbon_ema21_trailing"
    assert cfg.ribbon_meta == {"signal_id": "x", "layer": "LV1"}


def test_invalid_sl_logic_value_raises():
    import pytest
    with pytest.raises((ValueError, TypeError)):
        OrderConfig(direction="long", sl_logic="not_a_real_mode")  # type: ignore[arg-type]


def test_ribbon_meta_optional_for_line_buffer_mode():
    # Manual-line callers must keep working without setting ribbon_meta.
    cfg = OrderConfig(direction="long", sl_logic="line_buffer")
    assert cfg.ribbon_meta is None
