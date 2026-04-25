from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from backtests.ma_ribbon_ema21.phase1_cli import run_phase1
from backtests.ma_ribbon_ema21.tests.fixtures import make_uptrend_with_formation


def test_run_phase1_end_to_end_with_synthetic_data(tmp_path):
    """E2E: stage two synthetic CSVs, run the full pipeline, verify report."""
    cache = tmp_path / "cache"
    cache.mkdir()
    df1, _ = make_uptrend_with_formation(n_bars=400, formation_at_bar=120, base_price=100.0)
    df2, _ = make_uptrend_with_formation(n_bars=400, formation_at_bar=200, base_price=200.0)
    df1.to_csv(cache / "AAAUSDT_1h.csv", index=False)
    df2.to_csv(cache / "BBBUSDT_1h.csv", index=False)

    cfg_path = tmp_path / "phase1.json"
    cfg_path.write_text(json.dumps({
        "phase": "P1",
        "universe": ["AAAUSDT", "BBBUSDT"],
        "timeframes": ["1h"],
        "data_split": {"train_pct": 0.70},
        "moving_averages": {"ma_fast_1": 5, "ma_fast_2": 8, "ema_mid": 21, "ma_slow": 55},
        "bullish_alignment": {
            "require_close_above_ma5":  True,
            "require_close_above_ma8":  True,
            "require_close_above_ema21": True,
            "require_close_above_ma55":  True,
            "require_ma5_above_ma8":     True,
            "require_ma8_above_ema21":   True,
            "require_ema21_above_ma55":  True,
        },
        "forward_return_bars": [5, 10, 20, 50],
        "fees": {"per_side": 0.0005, "slippage_per_fill": 0.0001},
        "data_cache_dir": str(cache),
    }))

    output = tmp_path / "report.md"
    summary = run_phase1(config_path=str(cfg_path), output_path=str(output))

    assert output.exists()
    text = output.read_text()
    assert "AAAUSDT" in text
    assert "BBBUSDT" in text
    assert "Phase 1 cohort report" in text
    assert summary["total_events"] > 0
    assert "gate" in summary
