"""Default rule tokenizer vocabulary (v1).

Coarse token   =  5040 values  (line_role × direction × timeframe × duration × slope_coarse)
Fine token     = 21600 values  (slope_fine × touch × bounce × break × anchor × volatility × volume)

Every dimension is enumerable. No hash tables, no random sampling.
Encoding is a mixed-radix integer composition.
"""
from __future__ import annotations

from ..schemas.trendline import BucketSpec, TokenizerConfig


# ── coarse dimensions (order matters for the mixed-radix encoding) ───────

LINE_ROLES = [
    "support", "resistance",
    "channel_upper", "channel_lower",
    "wedge_side", "triangle_side",
    "unknown",
]
DIRECTIONS = ["up", "down", "flat"]
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]

# Duration in bars (half-open intervals: v in [edges[i], edges[i+1]))
DURATION_EDGES = [0, 16, 48, 128, 10_000_000]
DURATION_LABELS = ["short", "medium", "long", "very_long"]

# Coarse slope bucket based on |log-price-slope-per-bar|
SLOPE_COARSE_EDGES = [0.0, 0.002, 0.005, 0.015, 0.04, 1e9]
SLOPE_COARSE_LABELS = ["flat", "low", "mid", "high", "very_high"]

COARSE_DIM_ORDER = [
    "line_role",
    "direction",
    "timeframe",
    "duration_bucket",
    "slope_coarse",
]

# ── fine dimensions ───────────────────────────────────────────────────────

SLOPE_FINE_QUANTILES = 10  # 10 sub-buckets within each coarse slope bin

TOUCH_LABELS = ["t1", "t2", "t3", "t4", "t5plus"]

BOUNCE_EDGES = [0.0, 0.5, 1.5, 1e9]  # in ATR units; "none" is a separate value
BOUNCE_LABELS = ["none", "weak", "medium", "strong"]

BREAK_LABELS = ["none", "touched", "broken_weak", "broken_retested"]

ANCHOR_LABELS = ["early", "middle", "late"]

VOLATILITY_LABELS = ["vol_low", "vol_normal", "vol_high"]
VOLUME_LABELS = ["volu_low", "volu_normal", "volu_high"]

FINE_DIM_ORDER = [
    "slope_fine",
    "touch_bucket",
    "bounce_bucket",
    "break_bucket",
    "anchor_bucket",
    "volatility_bucket",
    "volume_bucket",
]


def _enum(name: str, values: list[str]) -> BucketSpec:
    return BucketSpec(name=name, kind="enum", values=values)


def _linear(name: str, edges: list[float]) -> BucketSpec:
    return BucketSpec(name=name, kind="linear", edges=edges)


def default_config() -> TokenizerConfig:
    """The v1 rule tokeniser configuration. Bumping this version means
    all stored tokens from a prior version must be re-encoded."""
    coarse = [
        _enum("line_role", LINE_ROLES),
        _enum("direction", DIRECTIONS),
        _enum("timeframe", TIMEFRAMES),
        BucketSpec(name="duration_bucket", kind="linear",
                   edges=DURATION_EDGES, values=DURATION_LABELS),
        BucketSpec(name="slope_coarse", kind="linear",
                   edges=SLOPE_COARSE_EDGES, values=SLOPE_COARSE_LABELS),
    ]
    fine = [
        BucketSpec(name="slope_fine", kind="quantile",
                   edges=[i / SLOPE_FINE_QUANTILES for i in range(SLOPE_FINE_QUANTILES + 1)],
                   values=[f"sf{i}" for i in range(SLOPE_FINE_QUANTILES)]),
        _enum("touch_bucket", TOUCH_LABELS),
        BucketSpec(name="bounce_bucket", kind="linear",
                   edges=BOUNCE_EDGES, values=BOUNCE_LABELS[1:]),   # 3 non-"none" buckets
        _enum("break_bucket", BREAK_LABELS),
        _enum("anchor_bucket", ANCHOR_LABELS),
        _enum("volatility_bucket", VOLATILITY_LABELS),
        _enum("volume_bucket", VOLUME_LABELS),
    ]
    return TokenizerConfig(
        version="rule.v1",
        coarse_dims=coarse,
        fine_dims=fine,
        feature_vector_keys=[
            "normalized_start_price", "normalized_end_price",
            "normalized_start_bar_index", "normalized_end_bar_index",
            "duration_bars_log", "slope", "angle_rad",
            "touch_count", "rejection_strength_atr",
            "break_distance_atr", "bounce_strength_atr",
            "retested_after_break",
            "volatility_atr_pct", "volume_z_score",
            "distance_to_ma20_atr",
            "distance_to_recent_high_atr", "distance_to_recent_low_atr",
            # + role_onehot[7] + timeframe_onehot[12] appended by the feature extractor
        ],
    )


def coarse_cardinalities() -> list[int]:
    return [
        len(LINE_ROLES), len(DIRECTIONS), len(TIMEFRAMES),
        len(DURATION_LABELS), len(SLOPE_COARSE_LABELS),
    ]


def fine_cardinalities() -> list[int]:
    return [
        SLOPE_FINE_QUANTILES,
        len(TOUCH_LABELS),
        len(BOUNCE_LABELS),           # includes "none"
        len(BREAK_LABELS),
        len(ANCHOR_LABELS),
        len(VOLATILITY_LABELS),
        len(VOLUME_LABELS),
    ]


def coarse_vocab_size() -> int:
    s = 1
    for c in coarse_cardinalities():
        s *= c
    return s


def fine_vocab_size() -> int:
    s = 1
    for c in fine_cardinalities():
        s *= c
    return s
