"""Tests for the multi-task per-class eval (audit-grade benchmark)."""
from __future__ import annotations
import numpy as np

from trendline_tokenizer.benchmarks.multi_task_eval import (
    per_class_from_arrays, BINARY_LABELS, REGIME_LABELS, INVALIDATION_LABELS,
)


def test_per_class_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    rep = per_class_from_arrays(y_true, y_pred, 3, REGIME_LABELS)
    assert rep["overall_accuracy"] == 1.0
    assert rep["macro_f1"] == 1.0
    for label, m in rep["per_class"].items():
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0


def test_per_class_all_one_class():
    """Always predict 0 — captures the parallel_same-style failure."""
    y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)   # imbalanced
    y_pred = np.zeros_like(y_true)                     # always class 0
    rep = per_class_from_arrays(y_true, y_pred, 3, REGIME_LABELS)
    assert rep["overall_accuracy"] == 0.9              # looks great
    # But class 1 + 2 are F1=0
    assert rep["per_class"]["normal_vol"]["f1"] == 0.0
    assert rep["per_class"]["high_vol"]["f1"] == 0.0
    # Macro F1 is much worse than overall acc
    assert rep["macro_f1"] < 0.4


def test_per_class_5class_invalidation():
    n = 100
    y_true = np.array([i % 5 for i in range(n)])
    y_pred = y_true.copy()
    y_pred[:10] = (y_pred[:10] + 1) % 5   # 10% noise
    rep = per_class_from_arrays(y_true, y_pred, 5, INVALIDATION_LABELS)
    assert 0.85 <= rep["overall_accuracy"] <= 0.95
    assert rep["macro_f1"] > 0.7
    # Confusion matrix sums match
    cm = np.array(rep["confusion_matrix"])
    assert cm.sum() == n


def test_per_class_binary():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    rep = per_class_from_arrays(y_true, y_pred, 2, BINARY_LABELS)
    # 2 of 3 zeros correct, 2 of 3 ones correct -> overall 4/6
    assert rep["overall_accuracy"] == round(4 / 6, 4)
    assert "no" in rep["per_class"]
    assert "yes" in rep["per_class"]
