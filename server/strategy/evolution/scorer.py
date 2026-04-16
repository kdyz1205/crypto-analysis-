"""Setup Quality Scorer (Level 5) -- gates trades by ML prediction.

Loads the latest trained model and scores new trade setups on a 0.0-1.0
scale representing the predicted probability of winning.

Usage in the scan loop:
    score = score_setup(trade_params, ohlcv_context)
    if score < threshold:
        skip this trade

If no model exists yet (< 100 historical trades), returns 0.5 (neutral)
so the system runs unfiltered while accumulating training data.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, FEATURE_COUNT, extract_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    try:
        from server.core.config import PROJECT_ROOT
        return Path(PROJECT_ROOT)
    except Exception:
        return Path(__file__).resolve().parents[3]


def _models_dir() -> Path:
    return _project_root() / "data" / "models"


MODEL_FILENAME = "trendline_classifier_latest.pkl"
META_FILENAME = "trendline_classifier_latest_meta.json"

# ---------------------------------------------------------------------------
# Model cache (avoid re-loading from disk on every call)
# ---------------------------------------------------------------------------

_cached_model: Any = None
_cached_model_mtime: float = 0.0
_CACHE_REFRESH_INTERVAL = 300  # re-check disk every 5 min
_last_cache_check: float = 0.0


def _load_model() -> Any:
    """Load the model from disk, with a simple mtime-based cache.

    Returns the classifier object, or None if no model file exists.
    """
    global _cached_model, _cached_model_mtime, _last_cache_check

    now = time.time()
    model_path = _models_dir() / MODEL_FILENAME

    # Fast path: cache is fresh
    if _cached_model is not None and (now - _last_cache_check) < _CACHE_REFRESH_INTERVAL:
        return _cached_model

    _last_cache_check = now

    if not model_path.exists():
        _cached_model = None
        return None

    try:
        mtime = model_path.stat().st_mtime
        if mtime == _cached_model_mtime and _cached_model is not None:
            return _cached_model

        with open(model_path, "rb") as f:
            _cached_model = pickle.load(f)
        _cached_model_mtime = mtime
        print(f"[scorer] Loaded model from {model_path}", flush=True)
        return _cached_model
    except Exception as e:
        print(f"[scorer] Model load err: {e}", flush=True)
        _cached_model = None
        return None


def _load_config() -> dict:
    """Load evolution config for scorer threshold and other params."""
    config_path = _project_root() / "data" / "evolution_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_setup(
    trade_params: dict[str, Any],
    ohlcv_context: dict[str, np.ndarray] | None = None,
) -> float:
    """Score a new trade setup.

    Parameters
    ----------
    trade_params : dict
        Trade parameters matching the trade_log format.  Must include
        at least ``symbol``, ``timeframe``, ``direction``, ``entry_price``,
        ``stop_price``, ``tp_price``.
    ohlcv_context : dict | None
        OHLCV arrays (o/h/l/c/v as numpy arrays) for the symbol+TF
        around the current time.  If provided, enables richer feature
        extraction (market context indicators).

    Returns
    -------
    float in [0.0, 1.0]
        Probability of the setup being a winner.
        0.5 = neutral (no model or insufficient confidence).
    """
    model = _load_model()
    if model is None:
        return 0.5  # neutral -- no model trained yet

    try:
        features = extract_features(trade_params, ohlcv_context)
        X = features.reshape(1, -1)

        # Use predict_proba if available (most classifiers have it)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # proba shape: (1, 2) for binary.  Column 1 = P(win).
            if proba.shape[1] >= 2:
                return float(np.clip(proba[0, 1], 0.0, 1.0))
            return float(np.clip(proba[0, 0], 0.0, 1.0))
        else:
            # Fallback: hard prediction -> 0.3 or 0.7
            pred = model.predict(X)[0]
            return 0.7 if pred == 1 else 0.3

    except Exception as e:
        print(f"[scorer] Prediction err: {e}", flush=True)
        return 0.5  # neutral on error


def should_take_trade(
    trade_params: dict[str, Any],
    ohlcv_context: dict[str, np.ndarray] | None = None,
    threshold: float | None = None,
) -> tuple[bool, float]:
    """Convenience wrapper: score + threshold check.

    Parameters
    ----------
    trade_params : dict
    ohlcv_context : dict | None
    threshold : float | None
        Minimum score to accept. If None, reads from evolution_config.json
        (key ``scoring_threshold``, default 0.3).

    Returns
    -------
    (take_trade: bool, score: float)
    """
    if threshold is None:
        config = _load_config()
        threshold = float(config.get("scoring_threshold", 0.3))

    score = score_setup(trade_params, ohlcv_context)
    return score >= threshold, score


def get_scorer_status() -> dict[str, Any]:
    """Return status information about the scorer.

    Useful for debugging and the dashboard.
    """
    model = _load_model()
    config = _load_config()
    meta_path = _models_dir() / META_FILENAME

    status = {
        "model_loaded": model is not None,
        "threshold": float(config.get("scoring_threshold", 0.3)),
    }

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status["model_engine"] = meta.get("engine", "unknown")
            status["model_accuracy"] = meta.get("accuracy", 0)
            status["model_n_trades"] = meta.get("n_trades", 0)
            status["model_trained_at"] = meta.get("trained_at", "")
            status["top_features"] = dict(list(meta.get("feature_importances", {}).items())[:5])
        except Exception:
            pass

    return status


__all__ = [
    "score_setup",
    "should_take_trade",
    "get_scorer_status",
]
