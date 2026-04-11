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
    cfg = StrategyConfig()
    assert cfg.min_rr_ratio == 2.0
    assert cfg.min_profit_space_atr_mult == 1.0
