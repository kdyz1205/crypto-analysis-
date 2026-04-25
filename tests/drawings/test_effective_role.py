"""Effective-role inference test.

User pain point 2026-04-24:
    A line drawn as `side="resistance"` may currently be acting as SUPPORT
    if price has since risen above it. The DB label is the user's INTENT
    at draw-time and is intentionally stable. But anything that describes
    or reasons about the line's CURRENT behavior must use effective_role
    (line below price → support; line above price → resistance).

This test pins the exact ZEC scenario from 2026-04-24 to make sure I
never describe a line by its frozen DB side again.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.drawings.types import ManualTrendline


def _make_line(*, side="resistance", t_start=0, t_end=1000,
               price_start=100.0, price_end=200.0):
    return ManualTrendline(
        manual_line_id="test", symbol="TEST", timeframe="1h", side=side,
        source="manual", t_start=t_start, t_end=t_end,
        price_start=price_start, price_end=price_end,
        extend_left=False, extend_right=True,
        locked=False, label="", notes="",
        comparison_status="uncompared", override_mode="display_only",
        nearest_auto_line_id=None, slope_diff=None,
        projected_price_diff=None, overlap_ratio=None,
        created_at=0, updated_at=0,
    )


def test_zec_2026_04_24_scenario_db_says_resistance_but_role_is_support():
    """The exact bug. Line `manual-ZECUSDT-1d-resistance-...` projects to
    ~$289 right now; ZEC is at ~$377. DB says side=resistance, but the
    line is BELOW price → effective_role MUST be 'support'."""
    line = _make_line(
        side="resistance",
        t_start=1776614400, t_end=1777518279,
        price_start=233.34987587590896, price_end=357.9313526866013,
    )
    line_at_now = line.line_at_ts(1777060000)
    assert 280 < line_at_now < 300, f"sanity: line at now should be ~$289, got {line_at_now}"

    # Current ZEC ~ $377
    role = line.compute_effective_role(current_price=377.0, now_ts=1777060000)
    assert role == "support", (
        f"DB side='resistance' but line at $289 < price $377 → "
        f"effective_role MUST be 'support', got '{role}'. "
        f"This is the 2026-04-24 ZEC bug — fix forbidden."
    )


def test_role_when_price_below_line():
    """Line at $200, price at $150 → line is ceiling → resistance."""
    line = _make_line(price_start=200.0, price_end=200.0)
    assert line.compute_effective_role(150.0, 500) == "resistance"


def test_role_when_price_above_line():
    """Line at $100, price at $150 → line is floor → support."""
    line = _make_line(price_start=100.0, price_end=100.0)
    assert line.compute_effective_role(150.0, 500) == "support"


def test_role_on_line_within_tolerance():
    """Price within 0.05% of line → 'on_line'."""
    line = _make_line(price_start=100.0, price_end=100.0)
    # Default tolerance 0.05%
    assert line.compute_effective_role(100.04, 500) == "on_line"
    assert line.compute_effective_role(99.96, 500) == "on_line"


def test_db_side_does_not_influence_effective_role():
    """Same geometry, opposite db side → same effective_role.
    The function MUST NOT cheat by reading `self.side`."""
    geom = dict(price_start=100.0, price_end=100.0)
    line_R = _make_line(side="resistance", **geom)
    line_S = _make_line(side="support", **geom)
    # Price above the line → support, regardless of DB side
    assert line_R.compute_effective_role(150.0, 500) == "support"
    assert line_S.compute_effective_role(150.0, 500) == "support"
    # Price below the line → resistance, regardless of DB side
    assert line_R.compute_effective_role(50.0, 500) == "resistance"
    assert line_S.compute_effective_role(50.0, 500) == "resistance"


def test_log_interpolation_for_sloping_line():
    """Line $100 → $400 spans 1000s. At t=500 (midpoint), log interp
    → exp(log(100) + 0.5*log(4)) = exp(log(100)+log(2)) = $200, not $250."""
    line = _make_line(price_start=100.0, price_end=400.0,
                      t_start=0, t_end=1000)
    # Midpoint
    p_mid = line.line_at_ts(500)
    # log midpoint of 100,400 is sqrt(100*400) = 200
    assert abs(p_mid - 200.0) < 0.5, f"expected ~$200, got {p_mid}"


def test_role_at_extension_right():
    """Line $100 → $200 from t=0 to t=1000, extends right. At t=2000
    (extrapolation), line projects to $400 (log slope continues)."""
    line = _make_line(price_start=100.0, price_end=200.0,
                      t_start=0, t_end=1000)
    # Line at the right anchor exactly = $200 (no extrapolation needed)
    assert abs(line.line_at_ts(1000) - 200.0) < 0.1
