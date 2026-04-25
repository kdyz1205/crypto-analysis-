from __future__ import annotations
from server.strategy.catalog import STRATEGY_CATALOG


def test_ma_ribbon_template_present_with_correct_defaults():
    by_id = {t.template_id: t for t in STRATEGY_CATALOG}
    assert "ma_ribbon_ema21_auto" in by_id
    t = by_id["ma_ribbon_ema21_auto"]
    assert t.category == "trend"
    assert t.risk_level == "high"
    assert set(t.supported_timeframes) == {"5m", "15m", "1h", "4h"}
    assert t.default_params["max_concurrent_orders"] == 25
    assert t.default_params["dd_halt_pct"] == 0.15
    assert t.default_params["layer_risk_pct"] == {
        "LV1": 0.001, "LV2": 0.0025, "LV3": 0.005, "LV4": 0.02,
    }
    assert t.default_params["ribbon_buffer_pct"] == {
        "5m": 0.01, "15m": 0.04, "1h": 0.07, "4h": 0.10,
    }
