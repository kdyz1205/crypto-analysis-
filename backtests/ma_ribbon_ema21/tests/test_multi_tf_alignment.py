from __future__ import annotations
from backtests.ma_ribbon_ema21.multi_tf import (
    bar_close_time,
    latest_closed_bar_idx,
    TF_SECONDS,
)


def test_tf_seconds_table():
    assert TF_SECONDS["5m"]  == 300
    assert TF_SECONDS["15m"] == 900
    assert TF_SECONDS["1h"]  == 3600
    assert TF_SECONDS["4h"]  == 14400


def test_bar_close_time_5m():
    open_ts = 1_700_000_300
    assert bar_close_time(open_ts, "5m") == 1_700_000_300 + 300


def test_latest_closed_bar_idx_15m_at_specific_time():
    # Anchor base = 09:45. 15m bars open at 09:45, 10:00, 10:15. Close: 10:00, 10:15, 10:30.
    base = 1_700_000_000
    open_times = [base + 0, base + 900, base + 1800]
    # T = 10:10 = anchor + 25 minutes = base + 1500.
    # Latest closed bar at 10:10 = the [09:45,10:00] bar (idx 0).
    idx = latest_closed_bar_idx(open_times, "15m", T=base + 1500)
    assert idx == 0


def test_latest_closed_bar_idx_at_boundary_includes_closed_bar():
    open_times = [1_700_000_000 + 0,
                  1_700_000_000 + 900]
    idx = latest_closed_bar_idx(open_times, "15m", T=1_700_000_000 + 900)
    assert idx == 0


def test_latest_closed_bar_idx_before_any_close_returns_negative_one():
    open_times = [1_700_000_000]
    idx = latest_closed_bar_idx(open_times, "15m", T=1_700_000_000 + 100)
    assert idx == -1


def test_latest_closed_bar_idx_real_user_example():
    """At T = 10:10 UTC, latest 5m closed bar is [10:05, 10:10] (idx 4),
    latest 15m closed bar is [09:45, 10:00] (idx 0).
    """
    base = 1_700_000_000  # treat as 09:45 UTC anchor
    five_m_opens   = [base + i * 300 for i in range(10)]
    fifteen_m_opens = [base + i * 900 for i in range(4)]
    T = base + 25 * 60  # 10:10

    assert latest_closed_bar_idx(five_m_opens, "5m", T) == 4
    assert latest_closed_bar_idx(fifteen_m_opens, "15m", T) == 0
