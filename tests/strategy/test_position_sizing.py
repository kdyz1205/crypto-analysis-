"""Tests for position sizing and risk calibration."""
from server.strategy.position_sizing import (
    kelly_fraction,
    get_calibrated_params,
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
    assert wr == 0.229
    assert rr == 5.62
    assert hk > 0


def test_get_calibrated_params_unknown_tf():
    """Unknown timeframes get conservative defaults."""
    wr, rr, hk = get_calibrated_params("2h")
    assert wr > 0
    assert rr > 0
    assert hk > 0
    # Should be more conservative than 4h
    assert hk <= get_calibrated_params("4h")[2]


def test_stop_atr_mult_increases_with_timeframe():
    """Larger timeframes should have larger ATR multipliers."""
    assert TIMEFRAME_STOP_ATR_MULT["1m"] < TIMEFRAME_STOP_ATR_MULT["5m"]
    assert TIMEFRAME_STOP_ATR_MULT["5m"] < TIMEFRAME_STOP_ATR_MULT["1h"]
    assert TIMEFRAME_STOP_ATR_MULT["1h"] < TIMEFRAME_STOP_ATR_MULT["4h"]
    assert TIMEFRAME_STOP_ATR_MULT["4h"] < TIMEFRAME_STOP_ATR_MULT["1d"]
