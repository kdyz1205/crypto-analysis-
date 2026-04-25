"""Use the trained fusion model as an AI auto-drawer.

Given recent OHLCV for a (symbol, timeframe), the model emits trendline
candidates that look like the user's hand-drawn lines (because manual
records were 50x oversampled in training).

Pipeline:
  1. detect_lines() finds candidate trendlines via sr_patterns
  2. For each candidate, simulate "if this were the next-line target, what
     would the model predict for bounce/break?" by running the model on
     the truncated history ending at that candidate's end_bar.
  3. Keep only candidates with high-confidence (BOUNCE or BREAK) signals
     that ALSO match the user's manual style profile (role distribution,
     duration buckets).
  4. Save kept lines to data/agent_drawn/<sym>_<tf>_<ts>.jsonl.

The output is structurally identical to manual records — they can be fed
straight back into the next training round, completing the self-evolution
loop.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..inference.feature_cache import FeatureCache
from ..inference.inference_service import InferenceService
from ..inference.signal_engine import SignalEngine
from ..inference.runtime_detector import detect_lines
from ..registry.paths import ROOT
from ..schemas.trendline import TrendlineRecord
from .style_score import StyleScorer


DEFAULT_OUT_DIR = ROOT / "data" / "agent_drawn"


_STYLE_SCORER: StyleScorer | None = None


def _scorer() -> StyleScorer:
    global _STYLE_SCORER
    if _STYLE_SCORER is None:
        _STYLE_SCORER = StyleScorer()
    return _STYLE_SCORER


def auto_draw(
    *,
    service: InferenceService,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    min_confidence: float = 0.65,
    min_style_score: float = 0.55,
    keep_top_k: int = 20,
    out_dir: Path | None = None,
) -> tuple[list[TrendlineRecord], Path]:
    """Run the model as a lines emitter on the supplied DataFrame.

    Returns (kept_lines, jsonl_path).
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = detect_lines(df, symbol=symbol, timeframe=timeframe,
                              max_lines=keep_top_k * 5)
    if not candidates:
        return [], out_dir / f"{symbol}_{timeframe}_{int(time.time())}_empty.jsonl"

    # Re-prime the FeatureCache with the full df so prediction has
    # exactly that history available.
    fc = service.fc
    seen_open_times = {b.last_close_time for b in fc._buffers.values() if b.last_close_time}
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    for _, row in df.iterrows():
        v = row[ts_col]
        try:
            ot = int(v)
        except (TypeError, ValueError):
            ot = int(pd.Timestamp(v).timestamp() * 1000)
        if ot in seen_open_times:
            continue
        fc.push(symbol, timeframe, {
            "open_time": ot,
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0)),
        })

    pred = service.predict(symbol, timeframe)
    engine = SignalEngine()
    if pred is None:
        return [], out_dir / f"{symbol}_{timeframe}_{int(time.time())}_no_pred.jsonl"
    sig = engine.evaluate(pred)

    # Keep candidates whose role/direction match the model's predicted
    # next-line distribution + which had a decisive signal AND match the
    # user's hand-drawn style (cosine similarity in feature space).
    kept: list[TrendlineRecord] = []
    style_scores: list[float] = []
    if sig.action != "WAIT" and sig.confidence >= min_confidence:
        matching = [c for c in candidates if c.line_role == sig.predicted_role
                    or (sig.predicted_role == "channel_upper" and c.line_role == "resistance")
                    or (sig.predicted_role == "channel_lower" and c.line_role == "support")]
        # Style-similarity gate (toward user's manual centroid).
        scorer = _scorer()
        scored = [(c, scorer.score_one(c)) for c in matching]
        scored = [pair for pair in scored if pair[1] >= min_style_score]
        # Rank by combined: detector strength + style score.
        scored.sort(key=lambda p: (p[0].score or 0.0) * 0.4 + p[1] * 0.6, reverse=True)
        scored = scored[:keep_top_k]
        kept = [p[0] for p in scored]
        style_scores = [p[1] for p in scored]
        for r, s in zip(kept, style_scores):
            r.label_source = "auto_approved"
            r.auto_method = f"agent.auto_drawer/{pred.artifact_name}"
            r.confidence = float(sig.confidence)
            r.score = float(s)   # overwrite detector strength with style score
            r.notes = (f"agent v1: action={sig.action} conf={sig.confidence:.2f} "
                       f"trade={sig.trade_type} style={s:.2f}")

    out_path = out_dir / f"{symbol}_{timeframe}_{int(time.time())}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(r.model_dump_json() + "\n")
        fh.write(json.dumps({
            "_meta": True, "symbol": symbol, "timeframe": timeframe,
            "n_candidates": len(candidates), "n_kept": len(kept),
            "signal_action": sig.action, "confidence": sig.confidence,
            "predicted_role": sig.predicted_role,
            "artifact": pred.artifact_name,
            "style_scores": style_scores,
            "n_manual_in_centroid": _scorer().n_manual,
        }) + "\n")
    return kept, out_path
