"""ML Learning Layer (Level 4) -- trains a classifier on trade outcomes.

Workflow:
  1. Load completed trades with outcomes from trade_log.jsonl + daily reports
  2. Extract features for each trade via features.py
  3. Train XGBoost (preferred) or sklearn GradientBoosting classifier
  4. Save model to data/models/trendline_classifier_latest.pkl
  5. Log feature importances for interpretability

Public API:
  ``train_model(min_trades=100)`` -- force a training run
  ``retrain_if_needed()`` -- checks if enough new trades have accumulated
                              and retrains weekly (called from scan loop)
"""
from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, FEATURE_COUNT, extract_features
from .daily_report import _load_all_trades, load_report, list_reports

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
    d = _project_root() / "data" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


MODEL_FILENAME = "trendline_classifier_latest.pkl"
META_FILENAME = "trendline_classifier_latest_meta.json"

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _gather_training_data() -> tuple[list[dict], list[int]]:
    """Collect all trades with known outcomes for training.

    Sources:
      1. trade_log.jsonl trades that have been enriched with ``outcome``
         by the daily report process
      2. Trades stored inside daily report JSON files

    Returns (trades, labels) where label=1 for win, 0 for loss.
    Breakeven trades are excluded (ambiguous label).
    """
    seen_ids: set[str] = set()  # deduplicate by order_id + ts
    trades: list[dict] = []
    labels: list[int] = []

    def _add(t: dict) -> None:
        outcome = t.get("outcome", "")
        if outcome not in ("win", "loss"):
            return
        # Deduplicate
        key = f"{t.get('order_id', '')}_{t.get('ts', '')}"
        if key in seen_ids:
            return
        seen_ids.add(key)
        trades.append(t)
        labels.append(1 if outcome == "win" else 0)

    # Source 1: raw trade log (only those already enriched)
    for t in _load_all_trades():
        _add(t)

    # Source 2: daily report files (trades therein have outcomes)
    for date_str in list_reports(last_n=365):
        report = load_report(date_str)
        if not report:
            continue
        for t in report.get("trades", []):
            _add(t)

    return trades, labels


def _build_feature_matrix(trades: list[dict]) -> np.ndarray:
    """Build an (N, FEATURE_COUNT) matrix from trade dicts.

    OHLCV context is not available at training time (trades are historical),
    so market-context features use whatever the trade dict already contains
    (populated by the daily-report enrichment), plus defaults for the rest.
    """
    X = np.zeros((len(trades), FEATURE_COUNT), dtype=float)
    for i, trade in enumerate(trades):
        X[i] = extract_features(trade, ohlcv=None)
    return X


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _get_classifier():
    """Return (classifier_instance, engine_name).

    Tries XGBoost first, falls back to sklearn GradientBoosting.
    """
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        return clf, "xgboost"
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        return clf, "sklearn_gbc"
    except ImportError:
        pass

    return None, "none"


def train_model(min_trades: int = 100) -> dict[str, Any]:
    """Train the setup quality classifier.

    Parameters
    ----------
    min_trades : int
        Minimum number of resolved trades required. If fewer are
        available, training is skipped and a status dict is returned.

    Returns
    -------
    dict with keys:
      ``ok``: bool
      ``reason``: str (if not ok)
      ``engine``: str (xgboost / sklearn_gbc)
      ``n_trades``: int
      ``accuracy``: float (in-sample, for sanity check only)
      ``feature_importances``: dict[str, float]
      ``model_path``: str
    """
    # 1. Gather data
    trades, labels = _gather_training_data()
    n = len(trades)
    print(f"[trainer] Gathered {n} resolved trades for training", flush=True)

    if n < min_trades:
        msg = f"Not enough trades ({n}/{min_trades}). Skipping training."
        print(f"[trainer] {msg}", flush=True)
        return {"ok": False, "reason": msg, "n_trades": n}

    # 2. Build features
    X = _build_feature_matrix(trades)
    y = np.array(labels, dtype=int)

    # 3. Get classifier
    clf, engine = _get_classifier()
    if clf is None:
        msg = "No ML backend available (install xgboost or scikit-learn)"
        print(f"[trainer] {msg}", flush=True)
        return {"ok": False, "reason": msg, "n_trades": n}

    print(f"[trainer] Training with engine={engine}, n={n}, features={FEATURE_COUNT}", flush=True)

    # 4. Train-test split (80/20) for validation
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
    except (ImportError, ValueError):
        # No sklearn or too few samples for stratified split -- train on all
        X_train, X_test, y_train, y_test = X, X, y, y

    # 5. Fit
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        msg = f"Training failed: {e}"
        print(f"[trainer] {msg}", flush=True)
        return {"ok": False, "reason": msg, "n_trades": n}

    # 6. Evaluate
    try:
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred, target_names=["loss", "win"], zero_division=0)
        print(f"[trainer] Validation accuracy: {accuracy:.3f}", flush=True)
        print(f"[trainer]\n{report_str}", flush=True)
    except ImportError:
        y_pred = clf.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test))

    # 7. Feature importances
    importances = {}
    try:
        imp = clf.feature_importances_
        for i, name in enumerate(FEATURE_NAMES):
            importances[name] = round(float(imp[i]), 6)
        # Sort by importance descending
        importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
        print("[trainer] Feature importances:", flush=True)
        for name, val in list(importances.items())[:10]:
            print(f"  {name}: {val:.4f}", flush=True)
    except Exception:
        pass

    # 8. Save model
    model_dir = _models_dir()
    model_path = model_dir / MODEL_FILENAME
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Save metadata
    meta = {
        "engine": engine,
        "n_trades": n,
        "n_features": FEATURE_COUNT,
        "feature_names": FEATURE_NAMES,
        "accuracy": round(accuracy, 4),
        "feature_importances": importances,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "win_rate_train": round(float(np.mean(y_train)) * 100, 1),
        "win_rate_test": round(float(np.mean(y_test)) * 100, 1),
    }
    meta_path = model_dir / META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[trainer] Model saved to {model_path}", flush=True)
    print(f"[trainer] Metadata saved to {meta_path}", flush=True)

    return {
        "ok": True,
        "engine": engine,
        "n_trades": n,
        "accuracy": round(accuracy, 4),
        "feature_importances": importances,
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------------
# Auto-retrain logic
# ---------------------------------------------------------------------------

_RETRAIN_INTERVAL_DAYS = 7     # retrain at most once per week
_RETRAIN_MIN_NEW_TRADES = 50   # require at least 50 new trades since last train


def retrain_if_needed() -> dict[str, Any]:
    """Check if we should retrain, and do so if conditions are met.

    Conditions:
      1. At least ``_RETRAIN_INTERVAL_DAYS`` since last training
      2. At least ``_RETRAIN_MIN_NEW_TRADES`` new resolved trades since
         last model was saved

    Returns a status dict. If training ran, it includes the train_model()
    result.
    """
    meta_path = _models_dir() / META_FILENAME
    last_trained_at = None
    last_n_trades = 0

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            last_trained_at = meta.get("trained_at")
            last_n_trades = int(meta.get("n_trades", 0))
        except Exception:
            pass

    # Check time since last train
    if last_trained_at:
        try:
            last_dt = datetime.fromisoformat(last_trained_at)
            days_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
            if days_since < _RETRAIN_INTERVAL_DAYS:
                return {
                    "ok": True,
                    "action": "skipped",
                    "reason": f"Only {days_since:.1f} days since last train (need {_RETRAIN_INTERVAL_DAYS})",
                }
        except Exception:
            pass

    # Check new trade count
    trades, labels = _gather_training_data()
    current_n = len(trades)
    new_trades = current_n - last_n_trades

    if new_trades < _RETRAIN_MIN_NEW_TRADES:
        return {
            "ok": True,
            "action": "skipped",
            "reason": f"Only {new_trades} new trades since last train (need {_RETRAIN_MIN_NEW_TRADES})",
            "current_total": current_n,
        }

    # Run training
    print(f"[trainer] Retraining: {new_trades} new trades, {current_n} total", flush=True)
    result = train_model(min_trades=50)
    result["action"] = "trained" if result.get("ok") else "failed"
    return result


def get_model_info() -> dict[str, Any]:
    """Return metadata about the current model (if any)."""
    meta_path = _models_dir() / META_FILENAME
    if not meta_path.exists():
        return {"exists": False}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["exists"] = True
        return meta
    except Exception:
        return {"exists": False}


__all__ = [
    "train_model",
    "retrain_if_needed",
    "get_model_info",
]
