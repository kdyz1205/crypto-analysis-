"""FastAPI router for the trendline fusion model.

Endpoints:
    GET  /api/trendline/health
        Returns the loaded model's manifest + readiness.

    POST /api/trendline/predict
        body: {"symbol": "BTCUSDT", "timeframe": "5m"}
        Pulls the latest cached OHLCV from the existing data_service,
        feeds it into the InferenceService, runs the model, and returns
        a SignalRecord-shaped JSON.

    POST /api/trendline/feedback
        body: {"event_type": "signal_accepted"|"signal_rejected"|"corrected_trendline",
               ...event-specific fields}
        Appends to data/trendline_feedback.jsonl via FeedbackStore.

    GET  /api/trendline/feedback/count
        Returns counts per event_type.

This router is paper-only. It never places live orders.
"""
from __future__ import annotations
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from trendline_tokenizer.feedback.schemas import (
    CorrectedTrendline, SignalAccepted, SignalRejected, parse_feedback_line,
)
from trendline_tokenizer.feedback.store import FeedbackStore
from trendline_tokenizer.inference.feature_cache import FeatureCache
from trendline_tokenizer.inference.inference_service import InferenceService
from trendline_tokenizer.inference.signal_engine import SignalEngine
from trendline_tokenizer.inference.paper_dispatcher import PaperDispatcher
from trendline_tokenizer.registry.paths import latest_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEEDBACK = PROJECT_ROOT / "data" / "trendline_feedback.jsonl"
DEFAULT_SIGNAL_LOG = PROJECT_ROOT / "data" / "trendline_signals.jsonl"


router = APIRouter(prefix="/api/trendline", tags=["trendline"])


# Singleton state (loaded lazily on first call)
_state: dict = {
    "service": None,                # InferenceService
    "engine": None,                 # SignalEngine
    "dispatcher": None,             # PaperDispatcher
    "feedback_store": None,         # FeedbackStore
    "manifest_path": None,
}


def _ensure_loaded() -> tuple[InferenceService, SignalEngine, PaperDispatcher, FeedbackStore]:
    if _state["service"] is None:
        latest = latest_fusion("fusion.v0.1")
        if latest is None:
            raise HTTPException(503, "no fusion model trained yet; run train_fusion first")
        manifest_path = latest / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(503, f"manifest missing at {manifest_path}")
        _state["manifest_path"] = manifest_path
        _state["service"] = InferenceService(manifest_path, device="cpu")
        _state["engine"] = SignalEngine()
        _state["dispatcher"] = PaperDispatcher(DEFAULT_SIGNAL_LOG)
        _state["feedback_store"] = FeedbackStore(DEFAULT_FEEDBACK)
    return (_state["service"], _state["engine"],
            _state["dispatcher"], _state["feedback_store"])


@router.get("/health")
def health() -> dict:
    try:
        svc, _, _, _ = _ensure_loaded()
        return {
            "ready": True,
            "manifest": svc.manifest.model_dump(),
            "device": svc.device,
        }
    except HTTPException as exc:
        return {"ready": False, "reason": exc.detail}


class PredictRequest(BaseModel):
    symbol: str
    timeframe: str
    push_recent_bars: int = 256   # how many most-recent OHLCV bars to push into the cache


@router.post("/predict")
def predict(req: PredictRequest) -> dict:
    svc, engine, dispatcher, _ = _ensure_loaded()

    # Pull recent OHLCV via the existing data_service helper.
    try:
        from server.data_service import _find_csv, _load_csv
        p = _find_csv(req.symbol, req.timeframe)
        if p is None:
            raise HTTPException(404, f"no OHLCV cache for {req.symbol} {req.timeframe}")
        df = _load_csv(p)
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"failed to load OHLCV: {exc}")

    # Push the tail bars into the cache.
    n = min(req.push_recent_bars, len(df))
    tail = df.tail(n)
    ts_col = "open_time" if "open_time" in tail.columns else "timestamp"
    for _, row in tail.iterrows():
        v = row[ts_col]
        try:
            ot = int(v)
        except (TypeError, ValueError):
            import pandas as pd
            ot = int(pd.Timestamp(v).timestamp() * 1000)
        svc.push_bar(req.symbol, req.timeframe, {
            "open_time": ot,
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0)),
        })

    pred = svc.predict(req.symbol, req.timeframe)
    if pred is None:
        raise HTTPException(503, "not enough bars in cache yet")

    sig = engine.evaluate(pred)
    dispatcher.dispatch(sig)
    return sig.to_dict()


@router.post("/feedback")
def feedback(payload: dict = Body(...)) -> dict:
    _, _, _, store = _ensure_loaded()
    if "created_at" not in payload:
        payload["created_at"] = int(time.time())
    try:
        event = parse_feedback_line(payload)
    except Exception as exc:
        raise HTTPException(400, f"invalid feedback payload: {exc}")
    store.append(event)
    return {"ok": True, "stored": event.event_type}


@router.get("/feedback/count")
def feedback_count() -> dict:
    _, _, _, store = _ensure_loaded()
    return store.count_by_type()


@router.get("/signals/recent")
def recent_signals(limit: int = 50) -> dict:
    _, _, dispatcher, _ = _ensure_loaded()
    rows = dispatcher.read_all()[-limit:]
    return {"count": len(rows), "signals": rows}


@router.get("/agent/state")
def agent_state() -> dict:
    """Current self-evolving agent state."""
    from trendline_tokenizer.agent.state import AgentState
    s = AgentState.load()
    return {
        "iteration": s.iteration,
        "last_run_ts": s.last_run_ts,
        "last_train_ts": s.last_train_ts,
        "last_train_artifact": s.last_train_artifact,
        "n_lines_auto_drawn_total": s.n_lines_auto_drawn_total,
        "n_retrain_triggered": s.n_retrain_triggered,
        "recent_log": s.log[-10:],
    }


@router.get("/agent/reports")
def agent_reports(limit: int = 20) -> dict:
    """Tail of the agent's iteration reports."""
    from trendline_tokenizer.agent.report import tail_reports
    return {"reports": tail_reports(n=limit)}


@router.post("/agent/run")
def agent_run(req: dict | None = Body(None)) -> dict:
    """Run ONE iteration of the self-evolving agent on demand.

    Body (all optional):
      {"symbols": ["BTCUSDT"], "timeframes": ["5m"],
       "feedback_threshold": 5, "force_retrain": false}

    Returns the IterationReport as a dict. Blocks until the iteration
    finishes; for long-running iterations, prefer running the loop
    standalone via `python -m trendline_tokenizer.agent.loop ...`.
    """
    from trendline_tokenizer.agent.loop import iteration_step
    from trendline_tokenizer.agent.state import AgentState

    req = req or {}
    symbols = req.get("symbols") or ["BTCUSDT"]
    timeframes = req.get("timeframes") or ["5m"]
    feedback_threshold = int(req.get("feedback_threshold", 5))
    force_retrain = bool(req.get("force_retrain", False))

    state = AgentState.load()
    state.iteration += 1
    rep = iteration_step(
        state=state, symbols=symbols, timeframes=timeframes,
        feedback_threshold=feedback_threshold,
        force_retrain=force_retrain,
    )
    state.last_run_ts = int(time.time())
    state.save()
    return rep.as_dict()


@router.get("/agent/auto_drawn")
def agent_auto_drawn(symbol: str | None = None, timeframe: str | None = None,
                     limit: int = 50) -> dict:
    """List the most-recent auto-drawn lines from data/agent_drawn/."""
    from pathlib import Path
    import json as _json
    drawn_dir = PROJECT_ROOT / "data" / "agent_drawn"
    if not drawn_dir.exists():
        return {"files": [], "lines": []}
    files = sorted(drawn_dir.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if symbol:
        files = [f for f in files if f.name.upper().startswith(symbol.upper())]
    if timeframe:
        files = [f for f in files if f"_{timeframe}_" in f.name]
    files = files[:limit]
    lines: list[dict] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                if obj.get("_meta"):
                    continue
                lines.append(obj)
            except Exception:
                continue
    return {"files": [str(f.name) for f in files], "lines": lines[-200:]}
