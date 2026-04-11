"""Tests for position sizing and risk calibration."""
from server.strategy.position_sizing import (
    kelly_fraction,
    get_calibrated_params,
    is_timeframe_verified,
    TIMEFRAME_STOP_ATR_MULT,
    BACKTEST_CALIBRATION,
)


def test_kelly_fraction_positive_ev():
    """22.9% WR x 5.62 RR should give positive Kelly."""
    kf = kelly_fraction(0.229, 5.62)
    assert kf > 0
    assert kf < 0.25  # capped at 25%


def test_kelly_fraction_negative_ev():
    """5% WR x 2.0 RR = negative EV, Kelly should be 0."""
    kf = kelly_fraction(0.05, 2.0)
    assert kf == 0.0


def test_kelly_fraction_zero_inputs():
    assert kelly_fraction(0.0, 5.0) == 0.0
    assert kelly_fraction(0.3, 0.0) == 0.0


def test_calibration_table_has_profitable_timeframes():
    """4h must be in calibration table with positive EV."""
    assert "4h" in BACKTEST_CALIBRATION
    wr, rr, ev, hk, stop = BACKTEST_CALIBRATION["4h"]
    assert ev > 0, "4h must have positive expected value"
    assert hk > 0, "4h must have positive Kelly"


def test_calibration_table_no_unprofitable():
    """5m and 15m should NOT be in calibration table."""
    for tf in BACKTEST_CALIBRATION:
        _, _, ev, _, _ = BACKTEST_CALIBRATION[tf]
        assert ev > 0, f"{tf} has negative EV but is in calibration table"


def test_get_calibrated_params_known_tf():
    wr, rr, hk = get_calibrated_params("4h")
    assert wr == 0.154
    assert rr == 5.69
    assert hk > 0


def test_get_calibrated_params_unknown_tf_returns_zero():
    """Unknown timeframes get zeros — they must NOT trade."""
    wr, rr, hk = get_calibrated_params("2h")
    assert wr == 0.0
    assert rr == 0.0
    assert hk == 0.0


def test_uncalibrated_5m_returns_zero():
    """5m is explicitly not profitable — must return zeros."""
    wr, rr, hk = get_calibrated_params("5m")
    assert wr == 0.0 and rr == 0.0 and hk == 0.0


def test_calibrated_15m_returns_values():
    """15m is profitable in 15-coin backtest — must return real values."""
    wr, rr, hk = get_calibrated_params("15m")
    assert wr > 0
    assert rr > 0
    assert hk > 0


def test_uncalibrated_1h_returns_zero():
    """1h is NOT profitable in 15-coin backtest — must return zeros."""
    wr, rr, hk = get_calibrated_params("1h")
    assert wr == 0.0 and rr == 0.0 and hk == 0.0


def test_is_timeframe_verified():
    assert is_timeframe_verified("4h") == True
    assert is_timeframe_verified("15m") == True
    # 1h was profitable in 8-coin test but NOT in 15-coin test
    assert is_timeframe_verified("1h") == False
    assert is_timeframe_verified("5m") == False
    assert is_timeframe_verified("1m") == False
    assert is_timeframe_verified("unknown") == False


def test_stop_atr_mult_increases_with_timeframe():
    """Larger timeframes should have larger ATR multipliers."""
    assert TIMEFRAME_STOP_ATR_MULT["1m"] < TIMEFRAME_STOP_ATR_MULT["5m"]
    assert TIMEFRAME_STOP_ATR_MULT["5m"] < TIMEFRAME_STOP_ATR_MULT["1h"]
    assert TIMEFRAME_STOP_ATR_MULT["1h"] < TIMEFRAME_STOP_ATR_MULT["4h"]
    assert TIMEFRAME_STOP_ATR_MULT["4h"] < TIMEFRAME_STOP_ATR_MULT["1d"]
