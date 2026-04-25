"""Role-aware signal-engine tests: the 4-cell trade-type matrix from
TA_BASICS.md must hold for every (role, behaviour) combo.
"""
from __future__ import annotations

import pytest

from trendline_tokenizer.tokenizer.rule import _compose
from trendline_tokenizer.tokenizer.vocab import (
    LINE_ROLES, DIRECTIONS, TIMEFRAMES,
    DURATION_LABELS, SLOPE_COARSE_LABELS, coarse_cardinalities,
)
from trendline_tokenizer.inference.signal_engine import (
    SignalEngine, SignalEngineConfig, decode_role_from_coarse, effective_role,
)
from trendline_tokenizer.inference.inference_service import PredictionRecord


def _coarse_for_role(role: str) -> int:
    role_idx = LINE_ROLES.index(role)
    indices = [role_idx, 0, 0, 0, 0]   # role, dir=up, tf=1m, dur=short, slope=flat
    return _compose(indices, coarse_cardinalities())


def _pred(role: str, bounce_prob: float, break_prob: float,
          continuation_prob: float = 0.5) -> PredictionRecord:
    return PredictionRecord(
        symbol="BTC", timeframe="5m", timestamp=0,
        artifact_name="x", tokenizer_version="rule.v1",
        next_coarse_id=_coarse_for_role(role), next_fine_id=0,
        bounce_prob=bounce_prob, break_prob=break_prob,
        continuation_prob=continuation_prob, suggested_buffer_pct=0.01,
        n_input_records=1, n_bars_in_cache=100,
    )


def test_decode_role_round_trip():
    for role in LINE_ROLES:
        cid = _coarse_for_role(role)
        assert decode_role_from_coarse(cid) == role


def test_effective_role_collapses_channels():
    assert effective_role("channel_upper") == "resistance"
    assert effective_role("channel_lower") == "support"
    assert effective_role("support") == "support"
    assert effective_role("resistance") == "resistance"
    assert effective_role("unknown") == "unknown"


def test_support_bounce_is_long():
    se = SignalEngine()
    sig = se.evaluate(_pred("support", bounce_prob=0.8, break_prob=0.1))
    assert sig.action == "LONG"
    assert sig.trade_type == "bounce_long"
    assert sig.predicted_role == "support"


def test_support_break_is_short():
    se = SignalEngine()
    sig = se.evaluate(_pred("support", bounce_prob=0.1, break_prob=0.8))
    assert sig.action == "SHORT"
    assert sig.trade_type == "breakdown_short"


def test_resistance_bounce_is_short():
    se = SignalEngine()
    sig = se.evaluate(_pred("resistance", bounce_prob=0.8, break_prob=0.1))
    assert sig.action == "SHORT"
    assert sig.trade_type == "bounce_short"


def test_resistance_break_is_long():
    se = SignalEngine()
    sig = se.evaluate(_pred("resistance", bounce_prob=0.1, break_prob=0.8))
    assert sig.action == "LONG"
    assert sig.trade_type == "breakout_long"


def test_channel_upper_acts_as_resistance():
    se = SignalEngine()
    sig = se.evaluate(_pred("channel_upper", bounce_prob=0.8, break_prob=0.1))
    assert sig.action == "SHORT"
    assert sig.trade_type == "bounce_short"
    assert sig.predicted_role == "channel_upper"
    assert sig.extras["effective_role"] == "resistance"


def test_channel_lower_acts_as_support():
    se = SignalEngine()
    sig = se.evaluate(_pred("channel_lower", bounce_prob=0.1, break_prob=0.8))
    assert sig.action == "SHORT"
    assert sig.trade_type == "breakdown_short"
    assert sig.extras["effective_role"] == "support"


def test_unknown_role_yields_wait():
    se = SignalEngine()
    sig = se.evaluate(_pred("unknown", bounce_prob=0.9, break_prob=0.05))
    assert sig.action == "WAIT"


def test_indecisive_yields_wait_regardless_of_role():
    se = SignalEngine()
    sig = se.evaluate(_pred("support", bounce_prob=0.5, break_prob=0.45))
    assert sig.action == "WAIT"


def test_threshold_below_min_yields_wait():
    se = SignalEngine(SignalEngineConfig(bounce_threshold=0.7, break_threshold=0.7))
    sig = se.evaluate(_pred("support", bounce_prob=0.6, break_prob=0.1))
    assert sig.action == "WAIT"
