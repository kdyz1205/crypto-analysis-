"""Rule-based hierarchical tokeniser — transparent, deterministic,
mixed-radix integer encoding.

No ML. No randomness. Given a TrendlineRecord and a TokenizerConfig,
encode_rule returns exactly one (coarse_id, fine_id) tuple, and
decode_rule reconstructs a TrendlineRecord whose continuous fields
are the midpoints of the bucket that produced the tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..schemas.trendline import TrendlineRecord, TokenizedTrendline, TokenizerConfig
from .vocab import (
    LINE_ROLES, DIRECTIONS, TIMEFRAMES,
    DURATION_EDGES, DURATION_LABELS,
    SLOPE_COARSE_EDGES, SLOPE_COARSE_LABELS,
    SLOPE_FINE_QUANTILES,
    TOUCH_LABELS,
    BOUNCE_EDGES, BOUNCE_LABELS,
    BREAK_LABELS,
    ANCHOR_LABELS,
    VOLATILITY_LABELS, VOLUME_LABELS,
    coarse_cardinalities, fine_cardinalities,
)


# ── axis containers (integer indices into each dimension) ───────────────

@dataclass
class CoarseAxes:
    line_role_idx: int
    direction_idx: int
    timeframe_idx: int
    duration_idx: int
    slope_coarse_idx: int


@dataclass
class FineAxes:
    slope_fine_idx: int
    touch_idx: int
    bounce_idx: int
    break_idx: int
    anchor_idx: int
    volatility_idx: int
    volume_idx: int


# ── bucket helpers ──────────────────────────────────────────────────────

def _bucket_linear(value: float, edges: list[float]) -> int:
    """Return index i s.t. edges[i] <= value < edges[i+1]. Last bucket
    includes the top edge."""
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2  # clamp high


def _enum_idx(value: str, values: list[str]) -> int:
    try:
        return values.index(value)
    except ValueError:
        return values.index("unknown") if "unknown" in values else 0


def _duration_idx(duration_bars: int) -> int:
    return _bucket_linear(float(duration_bars), DURATION_EDGES)


def _slope_coarse_idx(abs_log_slope: float) -> int:
    return _bucket_linear(abs_log_slope, SLOPE_COARSE_EDGES)


def _slope_fine_idx(log_slope_per_bar: float, coarse_idx: int) -> int:
    """Within a coarse slope bin, split the *signed* log-slope range into
    10 sub-bins. Sign carries direction; but direction is already its own
    coarse dim, so within a coarse-slope bin |slope| is what distinguishes
    residual geometry. Fine bin = percentile within the coarse bin,
    approximated by assuming uniform distribution over (edge_lo, edge_hi)."""
    lo = SLOPE_COARSE_EDGES[coarse_idx]
    hi = SLOPE_COARSE_EDGES[coarse_idx + 1]
    if hi - lo <= 0:
        return 0
    abs_s = min(max(abs(log_slope_per_bar), lo), hi - 1e-12)
    frac = (abs_s - lo) / (hi - lo)
    return min(SLOPE_FINE_QUANTILES - 1, max(0, int(frac * SLOPE_FINE_QUANTILES)))


def _touch_idx(touch_count: int) -> int:
    if touch_count <= 1:
        return 0
    if touch_count == 2:
        return 1
    if touch_count == 3:
        return 2
    if touch_count == 4:
        return 3
    return 4  # 5+


def _bounce_idx(bounce_after: Optional[bool], bounce_strength_atr: Optional[float]) -> int:
    if bounce_after is None or not bounce_after or bounce_strength_atr is None:
        return 0  # none
    return _bucket_linear(float(bounce_strength_atr), BOUNCE_EDGES) + 1  # shift past "none"


def _break_idx(break_after: Optional[bool], retested: Optional[bool], break_distance_atr: Optional[float]) -> int:
    if not break_after:
        # distinguish "never-tested" vs "touched-but-not-broken"
        if break_distance_atr is not None and break_distance_atr > 0:
            return 1  # touched
        return 0  # none
    # broken
    if retested:
        return 3
    return 2  # broken_weak


def _anchor_idx(record: TrendlineRecord, window_bars: int = 400) -> int:
    """Position of the line's midpoint within a rolling context window.
    Heuristic: use end_bar_index's distance from the current visible edge."""
    dur = record.duration_bars()
    mid_ratio = (record.start_bar_index + dur / 2) / max(1.0, record.end_bar_index + 1)
    if mid_ratio < 0.33:
        return 0
    if mid_ratio < 0.66:
        return 1
    return 2


def _tercile(value: Optional[float], low: float, high: float) -> int:
    if value is None:
        return 1  # normal
    if value < low:
        return 0
    if value >= high:
        return 2
    return 1


def _volatility_idx(record: TrendlineRecord) -> int:
    # ATR/price terciles: <0.5 % low, >1.5 % high (crypto-tuned).
    return _tercile(record.volatility_atr_pct, 0.005, 0.015)


def _volume_idx(record: TrendlineRecord) -> int:
    # Volume z-score terciles.
    return _tercile(record.volume_z_score, -0.3, 0.3)


# ── mixed-radix integer composition ─────────────────────────────────────

def _compose(indices: list[int], cardinalities: list[int]) -> int:
    assert len(indices) == len(cardinalities), "shape mismatch"
    acc = 0
    for idx, card in zip(indices, cardinalities):
        if not (0 <= idx < card):
            raise ValueError(f"index {idx} out of range [0, {card})")
        acc = acc * card + idx
    return acc


def _decompose(token_id: int, cardinalities: list[int]) -> list[int]:
    out = [0] * len(cardinalities)
    for i in range(len(cardinalities) - 1, -1, -1):
        out[i] = token_id % cardinalities[i]
        token_id //= cardinalities[i]
    return out


# ── public encode / decode ──────────────────────────────────────────────

def encode_rule(record: TrendlineRecord, cfg: TokenizerConfig | None = None) -> TokenizedTrendline:
    """Produce (coarse_id, fine_id) for a record under the rule tokeniser.

    cfg is accepted for future versioning but currently the axes are
    hardcoded to match vocab.py.
    """
    version = cfg.version if cfg is not None else "rule.v1"

    # coarse axes
    coarse = CoarseAxes(
        line_role_idx=_enum_idx(record.line_role, LINE_ROLES),
        direction_idx=_enum_idx(record.direction, DIRECTIONS),
        timeframe_idx=_enum_idx(record.timeframe, TIMEFRAMES),
        duration_idx=_duration_idx(record.duration_bars()),
        slope_coarse_idx=_slope_coarse_idx(abs(record.log_slope_per_bar())),
    )
    coarse_id = _compose(
        [coarse.line_role_idx, coarse.direction_idx, coarse.timeframe_idx,
         coarse.duration_idx, coarse.slope_coarse_idx],
        coarse_cardinalities(),
    )

    # fine axes
    fine = FineAxes(
        slope_fine_idx=_slope_fine_idx(record.log_slope_per_bar(), coarse.slope_coarse_idx),
        touch_idx=_touch_idx(record.touch_count),
        bounce_idx=_bounce_idx(record.bounce_after, record.bounce_strength_atr),
        break_idx=_break_idx(record.break_after, record.retested_after_break, record.break_distance_atr),
        anchor_idx=_anchor_idx(record),
        volatility_idx=_volatility_idx(record),
        volume_idx=_volume_idx(record),
    )
    fine_id = _compose(
        [fine.slope_fine_idx, fine.touch_idx, fine.bounce_idx,
         fine.break_idx, fine.anchor_idx, fine.volatility_idx, fine.volume_idx],
        fine_cardinalities(),
    )

    return TokenizedTrendline(
        record_id=record.id,
        coarse_token_id=coarse_id,
        fine_token_id=fine_id,
        tokenizer_version=version,
    )


def decode_rule(tok: TokenizedTrendline, reference_record: TrendlineRecord | None = None) -> TrendlineRecord:
    """Reconstruct a TrendlineRecord from (coarse_id, fine_id).

    Continuous fields become bucket midpoints; geometry that was lost
    during bucketing is replaced with a representative value.

    If `reference_record` is provided, its start_time/end_time/symbol/
    exchange are carried through (they are outside the token schema).
    """
    ci = _decompose(tok.coarse_token_id, coarse_cardinalities())
    fi = _decompose(tok.fine_token_id, fine_cardinalities())

    line_role = LINE_ROLES[ci[0]]
    direction = DIRECTIONS[ci[1]]
    timeframe = TIMEFRAMES[ci[2]]
    dur_idx = ci[3]
    slope_coarse_idx = ci[4]

    # duration midpoint
    dur_lo, dur_hi = DURATION_EDGES[dur_idx], DURATION_EDGES[dur_idx + 1]
    if dur_hi > 1_000_000:
        dur_mid = dur_lo * 1.5 if dur_lo > 0 else 200
    else:
        dur_mid = (dur_lo + dur_hi) / 2
    duration = max(1, int(round(dur_mid)))

    # slope midpoint within coarse × fine quantile
    slope_fine_idx = fi[0]
    s_lo = SLOPE_COARSE_EDGES[slope_coarse_idx]
    s_hi = SLOPE_COARSE_EDGES[slope_coarse_idx + 1]
    if s_hi > 1e6:
        s_hi = s_lo * 2 if s_lo > 0 else 0.08
    q_lo = s_lo + (s_hi - s_lo) * slope_fine_idx / SLOPE_FINE_QUANTILES
    q_hi = s_lo + (s_hi - s_lo) * (slope_fine_idx + 1) / SLOPE_FINE_QUANTILES
    abs_slope = (q_lo + q_hi) / 2
    signed_slope = abs_slope if direction == "up" else (-abs_slope if direction == "down" else 0.0)

    # synthesise a concrete record
    base_price = reference_record.start_price if reference_record else 100.0
    start_idx = reference_record.start_bar_index if reference_record else 0
    start_time = reference_record.start_time if reference_record else 0
    end_idx = start_idx + duration
    # log-space: end_price = start_price * exp(signed_slope * duration)
    end_price = max(1e-9, base_price * math.exp(signed_slope * duration))

    # fine-axis reconstructions
    touch_count = [1, 2, 3, 4, 5][fi[1]]
    bounce_bucket = BOUNCE_LABELS[fi[2]]
    bounce_after = bounce_bucket != "none"
    bounce_strength = {"none": None, "weak": 0.25, "medium": 1.0, "strong": 2.5}[bounce_bucket]
    break_bucket = BREAK_LABELS[fi[3]]
    break_after = break_bucket in ("broken_weak", "broken_retested")
    retested = break_bucket == "broken_retested"
    break_distance = {"none": None, "touched": 0.1, "broken_weak": 0.6, "broken_retested": 1.2}[break_bucket]

    reconstructed = TrendlineRecord(
        id=f"decoded-{tok.record_id}",
        symbol=reference_record.symbol if reference_record else "RECON",
        exchange=reference_record.exchange if reference_record else "bitget",
        timeframe=timeframe,
        start_time=start_time,
        end_time=start_time + duration * 60,  # nominal — caller supplies real tf seconds
        start_bar_index=start_idx,
        end_bar_index=end_idx,
        start_price=base_price,
        end_price=end_price,
        line_role=line_role,
        direction=direction,
        touch_count=touch_count,
        rejection_strength_atr=None,
        bounce_after=bounce_after,
        bounce_strength_atr=bounce_strength,
        break_after=break_after,
        break_distance_atr=break_distance,
        retested_after_break=retested,
        volatility_atr_pct={"vol_low": 0.003, "vol_normal": 0.01, "vol_high": 0.025}[VOLATILITY_LABELS[fi[5]]],
        volume_z_score={"volu_low": -1.0, "volu_normal": 0.0, "volu_high": 1.0}[VOLUME_LABELS[fi[6]]],
        label_source="auto",
        auto_method=f"decode/{tok.tokenizer_version}",
        score=None,
        created_at=reference_record.created_at if reference_record else 0,
    )
    return reconstructed
