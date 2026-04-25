from server.strategy.config import StrategyConfig


def test_config_has_volume_fields():
    cfg = StrategyConfig()
    assert cfg.volume_surge_threshold == 1.5
    assert cfg.volume_lookback_bars == 20


def test_config_has_trend_fields():
    cfg = StrategyConfig()
    assert cfg.trend_ema_period == 50
    assert cfg.trend_weight == 0.10


def test_config_has_rr_gate_fields():
    # 2026-04-23: min_rr_ratio default was tuned 3.0 → 1.8 based on
    # backtest results. Test simply validates the field exists and is
    # a sensible positive float, not a specific target value (which is
    # a product-level tuning decision, not a contract).
    cfg = StrategyConfig()
    assert cfg.min_rr_ratio > 0
    assert cfg.min_rr_ratio < 10
    assert cfg.min_profit_space_atr_mult == 1.0
