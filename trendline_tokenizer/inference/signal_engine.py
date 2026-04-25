"""Signal engine: PredictionRecord -> SignalRecord.

Turns model probabilities into a discrete trading-relevant decision plus
a confidence score and a human-readable reason. The engine NEVER places
orders - it just emits a SignalRecord that downstream paper / live
dispatchers consume.

Decision rule (intentionally simple - the model has the nuance):
    bounce_prob >= bounce_threshold and bounce - break >= edge_min:
        action = "BOUNCE"
    break_prob  >= break_threshold and break - bounce >= edge_min:
        action = "BREAK"
    else:
        action = "WAIT"

confidence = max(bounce, break) when action != WAIT, else 1 - max.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal

from .inference_service import PredictionRecord


SignalAction = Literal["BOUNCE", "BREAK", "WAIT"]


@dataclass
class SignalRecord:
    symbol: str
    timeframe: str
    timestamp: int
    artifact_name: str
    tokenizer_version: str
    action: SignalAction
    confidence: float
    suggested_buffer_pct: float
    bounce_prob: float
    break_prob: float
    continuation_prob: float
    next_coarse_id: int
    next_fine_id: int
    reason: str
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SignalEngineConfig:
    bounce_threshold: float = 0.55
    break_threshold: float = 0.55
    edge_min: float = 0.10            # min |bounce - break| to be decisive
    min_buffer_pct: float = 0.001     # 0.1% floor
    max_buffer_pct: float = 0.05      # 5% ceiling


class SignalEngine:
    def __init__(self, cfg: SignalEngineConfig | None = None):
        self.cfg = cfg or SignalEngineConfig()

    def evaluate(self, pred: PredictionRecord) -> SignalRecord:
        cfg = self.cfg
        bo = pred.bounce_prob
        br = pred.break_prob
        co = pred.continuation_prob
        edge = bo - br
        action: SignalAction
        if bo >= cfg.bounce_threshold and edge >= cfg.edge_min:
            action = "BOUNCE"
            confidence = bo
            reason = (f"bounce_prob={bo:.2f} >= {cfg.bounce_threshold:.2f} and "
                      f"edge={edge:+.2f} >= {cfg.edge_min:.2f}")
        elif br >= cfg.break_threshold and -edge >= cfg.edge_min:
            action = "BREAK"
            confidence = br
            reason = (f"break_prob={br:.2f} >= {cfg.break_threshold:.2f} and "
                      f"edge={-edge:+.2f} >= {cfg.edge_min:.2f}")
        else:
            action = "WAIT"
            confidence = 1.0 - max(bo, br)
            reason = (f"no decisive edge: bounce={bo:.2f}, break={br:.2f}, "
                      f"|edge|={abs(edge):.2f} < {cfg.edge_min:.2f}")

        suggested_buf = max(cfg.min_buffer_pct,
                            min(cfg.max_buffer_pct, pred.suggested_buffer_pct))

        return SignalRecord(
            symbol=pred.symbol, timeframe=pred.timeframe,
            timestamp=pred.timestamp,
            artifact_name=pred.artifact_name,
            tokenizer_version=pred.tokenizer_version,
            action=action, confidence=float(confidence),
            suggested_buffer_pct=float(suggested_buf),
            bounce_prob=bo, break_prob=br, continuation_prob=co,
            next_coarse_id=pred.next_coarse_id,
            next_fine_id=pred.next_fine_id,
            reason=reason,
            extras={"n_input_records": pred.n_input_records,
                    "n_bars_in_cache": pred.n_bars_in_cache},
        )
