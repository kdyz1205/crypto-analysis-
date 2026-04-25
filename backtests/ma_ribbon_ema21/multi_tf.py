"""Multi-timeframe time alignment. Critical for no-lookahead across TFs."""
from __future__ import annotations
import bisect


TF_SECONDS: dict[str, int] = {
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
}


def bar_close_time(open_ts: int, tf: str) -> int:
    """Bar's close time = open_ts + tf duration."""
    if tf not in TF_SECONDS:
        raise ValueError(f"unknown tf {tf!r}; expected one of {list(TF_SECONDS)}")
    return open_ts + TF_SECONDS[tf]


def latest_closed_bar_idx(open_times: list[int], tf: str, T: int) -> int:
    """Index of the most recently closed bar at simulation time T.
    Returns -1 if no bar has closed yet at time T.
    `open_times` must be sorted ascending.
    """
    if tf not in TF_SECONDS:
        raise ValueError(f"unknown tf {tf!r}")
    duration = TF_SECONDS[tf]
    # Bar i is "closed at T" iff open_times[i] + duration <= T  <=>  open_times[i] <= T - duration.
    threshold = T - duration
    pos = bisect.bisect_right(open_times, threshold)
    return pos - 1
