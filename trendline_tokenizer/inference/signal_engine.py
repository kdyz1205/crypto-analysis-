"""Signal engine: PredictionRecord -> SignalRecord.

Combines (predicted line role) with (bounce vs break) probabilities to
emit a LONG / SHORT / WAIT decision per the canonical 4-cell trade-type
matrix from TA_BASICS.md:

    Line role @ now | bounce | break
    ---------------+---------+---------
    support        | LONG    | SHORT       (bounce-long  / breakdown-short)
    resistance     | SHORT   | LONG        (bounce-short / breakout-long)

Channels collapse to support/resistance:
    channel_upper -> resistance,  channel_lower -> support

Roles we cannot assign a directional thesis to (wedge_side, triangle_side,
unknown) yield WAIT regardless of probabilities. The model is also
suppressed when the role is ambiguous.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal

from ..tokenizer.rule import _decompose
from ..tokenizer.vocab import (
    LINE_ROLES, DIRECTIONS, TIMEFRAMES,
    DURATION_LABELS, SLOPE_COARSE_LABELS,
    coarse_cardinalities,
)
from .inference_service import PredictionRecord


SignalAction = Literal["LONG", "SHORT", "WAIT"]
TradeType = Literal[
    "bounce_long", "breakdown_short",
    "breakout_long", "bounce_short",
    "wait",
]


def decode_role_from_coarse(coarse_id: int) -> str:
    """Decompose a rule-coarse token id back to its line_role string."""
    indices = _decompose(coarse_id, coarse_cardinalities())
    role_idx = indices[0]
    if 0 <= role_idx < len(LINE_ROLES):
        return LINE_ROLES[role_idx]
    return "unknown"


def effective_role(line_role: str) -> str:
    """Collapse channel labels to plain support/resistance."""
    if line_role == "channel_upper":
        return "resistance"
    if line_role == "channel_lower":
        return "support"
    return line_role


@dataclass
class SignalRecord:
    symbol: str
    timeframe: str
    timestamp: int
    artifact_name: str
    tokenizer_version: str
    action: SignalAction
    trade_type: TradeType
    confidence: float
    suggested_buffer_pct: float
    bounce_prob: float
    break_prob: float
    continuation_prob: float
    next_coarse_id: int
    next_fine_id: int
    predicted_role: str
    reason: str
    # Geometry of the predicted next line — passed through from
    # PredictionRecord so the UI / strategy layer can draw / project
    # the line without re-decoding.
    decoded_role: str = "unknown"
    decoded_direction: str = "flat"
    decoded_log_slope_per_bar: float = 0.0
    decoded_duration_bars: int = 1
    # See PredictionRecord.line_endpoint_pct_change — this is the LINE's
    # endpoint % change, NOT the trade's expected return.
    line_endpoint_pct_change: float = 0.0
    horizon_seconds: int = 0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SignalEngineConfig:
    bounce_threshold: float = 0.55
    break_threshold: float = 0.55
    edge_min: float = 0.10
    min_buffer_pct: float = 0.001
    max_buffer_pct: float = 0.05


# 4-cell decision table: (effective_role, behaviour) -> (SignalAction, TradeType)
_DECISION_TABLE: dict[tuple[str, str], tuple[SignalAction, TradeType]] = {
    ("support",    "bounce"): ("LONG",  "bounce_long"),
    ("support",    "break"):  ("SHORT", "breakdown_short"),
    ("resistance", "bounce"): ("SHORT", "bounce_short"),
    ("resistance", "break"):  ("LONG",  "breakout_long"),
}


class SignalEngine:
    def __init__(self, cfg: SignalEngineConfig | None = None):
        self.cfg = cfg or SignalEngineConfig()

    def evaluate(self, pred: PredictionRecord) -> SignalRecord:
        cfg = self.cfg
        role_raw = decode_role_from_coarse(pred.next_coarse_id)
        role = effective_role(role_raw)
        bo = pred.bounce_prob
        br = pred.break_prob
        edge = bo - br

        # Decide the behavioural side first (bounce vs break vs neither)
        if bo >= cfg.bounce_threshold and edge >= cfg.edge_min:
            behaviour = "bounce"
            confidence = bo
            edge_phrase = f"edge={edge:+.2f}>={cfg.edge_min:.2f}"
        elif br >= cfg.break_threshold and -edge >= cfg.edge_min:
            behaviour = "break"
            confidence = br
            edge_phrase = f"edge={-edge:+.2f}>={cfg.edge_min:.2f}"
        else:
            behaviour = "neither"
            confidence = 1.0 - max(bo, br)
            edge_phrase = f"|edge|={abs(edge):.2f}<{cfg.edge_min:.2f}"

        # Map (role, behaviour) to action + trade_type
        action: SignalAction
        trade_type: TradeType
        if behaviour == "neither":
            action, trade_type = "WAIT", "wait"
            reason = f"no decisive edge: bounce={bo:.2f}, break={br:.2f}, {edge_phrase}"
        elif role not in ("support", "resistance"):
            action, trade_type = "WAIT", "wait"
            reason = (f"predicted role={role_raw!r} has no directional thesis; "
                      f"behaviour={behaviour}, confidence={confidence:.2f}")
        else:
            action, trade_type = _DECISION_TABLE[(role, behaviour)]
            reason = (f"role={role}/{behaviour}: confidence={confidence:.2f}, "
                      f"{edge_phrase} -> {trade_type}")

        suggested_buf = max(cfg.min_buffer_pct,
                            min(cfg.max_buffer_pct, pred.suggested_buffer_pct))

        return SignalRecord(
            symbol=pred.symbol, timeframe=pred.timeframe,
            timestamp=pred.timestamp,
            artifact_name=pred.artifact_name,
            tokenizer_version=pred.tokenizer_version,
            action=action, trade_type=trade_type,
            confidence=float(confidence),
            suggested_buffer_pct=float(suggested_buf),
            bounce_prob=bo, break_prob=br, continuation_prob=pred.continuation_prob,
            next_coarse_id=pred.next_coarse_id,
            next_fine_id=pred.next_fine_id,
            predicted_role=role_raw,
            reason=reason,
            decoded_role=pred.decoded_role,
            decoded_direction=pred.decoded_direction,
            decoded_log_slope_per_bar=pred.decoded_log_slope_per_bar,
            decoded_duration_bars=pred.decoded_duration_bars,
            line_endpoint_pct_change=pred.line_endpoint_pct_change,
            horizon_seconds=pred.horizon_seconds,
            extras={"n_input_records": pred.n_input_records,
                    "n_bars_in_cache": pred.n_bars_in_cache,
                    "effective_role": role,
                    "behaviour": behaviour,
                    "anchor_close": (pred.extras or {}).get("anchor_close", 0.0),
                    "anchor_open_time_ms": (pred.extras or {}).get("anchor_open_time_ms", 0)},
        )
