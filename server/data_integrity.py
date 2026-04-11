"""
Data Integrity Checker — proves data is complete or explains why not.

For each symbol + timeframe:
- Checks expected vs actual bar count
- Detects gaps, duplicates, ordering issues
- Reports earliest/latest timestamps
- Returns structured result for logging
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import polars as pl

INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800,
}


@dataclass(frozen=True, slots=True)
class DataIntegrityResult:
    symbol: str
    timeframe: str
    status: str  # "COMPLETE" | "INCOMPLETE" | "ERROR"
    received_bars: int
    expected_bars: int
    missing_bars: int
    has_gaps: bool
    gap_count: int
    has_duplicates: bool
    duplicate_count: int
    is_ordered: bool
    earliest_timestamp: str
    latest_timestamp: str
    requested_days: int
    issues: list[str]

    def to_log_line(self) -> str:
        return (
            f"[DATA][{self.symbol}][{self.timeframe}] "
            f"status={self.status} | "
            f"bars={self.received_bars}/{self.expected_bars} | "
            f"missing={self.missing_bars} | "
            f"gaps={self.gap_count} | "
            f"range={self.earliest_timestamp} -> {self.latest_timestamp}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "status": self.status,
            "received_bars": self.received_bars,
            "expected_bars": self.expected_bars,
            "missing_bars": self.missing_bars,
            "has_gaps": self.has_gaps,
            "gap_count": self.gap_count,
            "has_duplicates": self.has_duplicates,
            "duplicate_count": self.duplicate_count,
            "is_ordered": self.is_ordered,
            "earliest_timestamp": self.earliest_timestamp,
            "latest_timestamp": self.latest_timestamp,
            "issues": self.issues,
        }


def check_data_integrity(
    df: pl.DataFrame,
    symbol: str,
    timeframe: str,
    requested_days: int = 365,
) -> DataIntegrityResult:
    """Check a Polars DataFrame for completeness and quality."""
    issues: list[str] = []

    if df is None or df.is_empty():
        return DataIntegrityResult(
            symbol=symbol, timeframe=timeframe, status="ERROR",
            received_bars=0, expected_bars=0, missing_bars=0,
            has_gaps=False, gap_count=0, has_duplicates=False,
            duplicate_count=0, is_ordered=True,
            earliest_timestamp="", latest_timestamp="",
            requested_days=requested_days, issues=["no_data_received"],
        )

    n = len(df)
    interval_s = INTERVAL_SECONDS.get(timeframe, 3600)

    # Get timestamps
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    try:
        timestamps = df[ts_col].to_list()
        # Convert to epoch seconds if datetime
        if timestamps and hasattr(timestamps[0], 'timestamp'):
            ts_seconds = [int(t.timestamp()) for t in timestamps]
        else:
            ts_seconds = [int(t) for t in timestamps]
    except Exception as e:
        issues.append(f"timestamp_parse_error: {e}")
        return DataIntegrityResult(
            symbol=symbol, timeframe=timeframe, status="ERROR",
            received_bars=n, expected_bars=0, missing_bars=0,
            has_gaps=False, gap_count=0, has_duplicates=False,
            duplicate_count=0, is_ordered=True,
            earliest_timestamp="", latest_timestamp="",
            requested_days=requested_days, issues=issues,
        )

    earliest = min(ts_seconds)
    latest = max(ts_seconds)
    earliest_str = datetime.fromtimestamp(earliest, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    latest_str = datetime.fromtimestamp(latest, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    # Expected bars
    time_span = latest - earliest
    expected_bars = (time_span // interval_s) + 1 if interval_s > 0 else n
    missing_bars = max(0, expected_bars - n)

    # Check ordering
    is_ordered = all(ts_seconds[i] <= ts_seconds[i + 1] for i in range(len(ts_seconds) - 1))
    if not is_ordered:
        issues.append("timestamps_not_ordered")

    # Check duplicates
    unique_ts = set(ts_seconds)
    duplicate_count = n - len(unique_ts)
    has_duplicates = duplicate_count > 0
    if has_duplicates:
        issues.append(f"duplicate_timestamps: {duplicate_count}")

    # Check gaps
    gap_count = 0
    if len(ts_seconds) >= 2:
        sorted_ts = sorted(ts_seconds)
        for i in range(1, len(sorted_ts)):
            delta = sorted_ts[i] - sorted_ts[i - 1]
            # Allow up to 2x interval (some gaps are normal for weekends/maintenance)
            if delta > interval_s * 2:
                gap_count += 1
    has_gaps = gap_count > 0
    if has_gaps:
        issues.append(f"gaps_detected: {gap_count}")

    # Missing bars threshold
    if missing_bars > expected_bars * 0.05:
        issues.append(f"significant_missing_bars: {missing_bars}/{expected_bars} ({missing_bars/max(expected_bars,1)*100:.1f}%)")

    # Determine status
    if not issues:
        status = "COMPLETE"
    elif missing_bars > expected_bars * 0.20:
        status = "INCOMPLETE"
    else:
        status = "COMPLETE"  # minor issues are OK

    return DataIntegrityResult(
        symbol=symbol, timeframe=timeframe, status=status,
        received_bars=n, expected_bars=expected_bars,
        missing_bars=missing_bars, has_gaps=has_gaps, gap_count=gap_count,
        has_duplicates=has_duplicates, duplicate_count=duplicate_count,
        is_ordered=is_ordered,
        earliest_timestamp=earliest_str, latest_timestamp=latest_str,
        requested_days=requested_days, issues=issues,
    )


__all__ = ["DataIntegrityResult", "check_data_integrity"]
