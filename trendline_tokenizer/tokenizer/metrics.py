"""Reconstruction-error metrics — the only way to judge the tokeniser."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..schemas.trendline import TrendlineRecord


@dataclass
class RoundTripError:
    slope_err: float
    role_err: int                 # 0 or 1
    duration_err: float
    touch_err: float
    aggregate: float

    def as_dict(self) -> dict:
        return {
            "slope_err": self.slope_err,
            "role_err": self.role_err,
            "duration_err": self.duration_err,
            "touch_err": self.touch_err,
            "aggregate": self.aggregate,
        }


def round_trip_error(orig: TrendlineRecord, decoded: TrendlineRecord) -> RoundTripError:
    # log-slope relative error
    s_orig = orig.log_slope_per_bar()
    s_dec = decoded.log_slope_per_bar()
    if abs(s_orig) < 1e-9 and abs(s_dec) < 1e-9:
        slope_err = 0.0
    elif abs(s_orig) < 1e-9:
        slope_err = abs(s_dec)
    else:
        slope_err = abs(math.log((abs(s_dec) + 1e-9) / (abs(s_orig) + 1e-9)))

    role_err = 0 if decoded.line_role == orig.line_role else 1

    dur_o = max(1, orig.duration_bars())
    dur_d = max(1, decoded.duration_bars())
    duration_err = abs(dur_d - dur_o) / dur_o

    touch_err = abs(decoded.touch_count - orig.touch_count)

    aggregate = 0.4 * slope_err + 0.3 * role_err + 0.2 * duration_err + 0.1 * (touch_err / 5)
    return RoundTripError(slope_err, role_err, duration_err, touch_err, aggregate)


def summarize(errors: Iterable[RoundTripError]) -> dict:
    errs = list(errors)
    if not errs:
        return {"n": 0}
    def pct(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        if not xs:
            return 0.0
        k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
        return xs[k]
    agg = [e.aggregate for e in errs]
    slope = [e.slope_err for e in errs]
    role = [e.role_err for e in errs]
    return {
        "n": len(errs),
        "aggregate_median": pct(agg, 0.5),
        "aggregate_p95": pct(agg, 0.95),
        "slope_err_median": pct(slope, 0.5),
        "slope_err_p95": pct(slope, 0.95),
        "role_accuracy": 1.0 - sum(role) / len(role),
    }
