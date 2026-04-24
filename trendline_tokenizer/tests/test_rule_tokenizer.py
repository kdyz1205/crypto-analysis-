"""Rule tokeniser: determinism, bounded vocab, round-trip stability."""
from trendline_tokenizer.schemas.trendline import TrendlineRecord
from trendline_tokenizer.tokenizer import encode_rule, decode_rule
from trendline_tokenizer.tokenizer.vocab import (
    coarse_vocab_size, fine_vocab_size, default_config,
)


def _rec(**overrides):
    d = dict(
        id="r1", symbol="BTCUSDT", timeframe="4h",
        start_time=0, end_time=86400,
        start_bar_index=0, end_bar_index=24,
        start_price=60_000.0, end_price=63_000.0,
        line_role="support", direction="up",
        touch_count=3, bounce_after=True, bounce_strength_atr=1.0,
        break_after=False, volatility_atr_pct=0.012, volume_z_score=0.1,
        label_source="auto", created_at=0,
    )
    d.update(overrides)
    return TrendlineRecord(**d)


def test_vocab_sizes_fixed():
    assert coarse_vocab_size() == 5040
    assert fine_vocab_size() == 21600


def test_encode_deterministic():
    r = _rec()
    t1 = encode_rule(r)
    t2 = encode_rule(r)
    assert t1.coarse_token_id == t2.coarse_token_id
    assert t1.fine_token_id == t2.fine_token_id


def test_encode_within_vocab():
    r = _rec()
    tok = encode_rule(r)
    assert 0 <= tok.coarse_token_id < coarse_vocab_size()
    assert 0 <= tok.fine_token_id < fine_vocab_size()


def test_different_role_yields_different_coarse_token():
    r1 = _rec(line_role="support")
    r2 = _rec(id="r2", line_role="resistance")
    t1, t2 = encode_rule(r1), encode_rule(r2)
    assert t1.coarse_token_id != t2.coarse_token_id


def test_decode_preserves_role_direction_timeframe():
    r = _rec()
    tok = encode_rule(r)
    dec = decode_rule(tok, reference_record=r)
    assert dec.line_role == r.line_role
    assert dec.direction == r.direction
    assert dec.timeframe == r.timeframe


def test_encode_decode_encode_stable():
    """Re-encoding a decoded record must give the same tokens."""
    r = _rec()
    t1 = encode_rule(r)
    dec = decode_rule(t1, reference_record=r)
    t2 = encode_rule(dec)
    assert t1.coarse_token_id == t2.coarse_token_id
    # Fine token may differ slightly because decode snaps to bucket
    # midpoints; re-encode then lands in the same bin by construction.
    assert t1.fine_token_id == t2.fine_token_id


def test_mid_slope_vs_flat_get_different_coarse_slope():
    steep = _rec(start_price=100.0, end_price=130.0)
    flat = _rec(id="r_flat", start_price=100.0, end_price=100.1, direction="flat")
    t_steep, t_flat = encode_rule(steep), encode_rule(flat)
    assert t_steep.coarse_token_id != t_flat.coarse_token_id
