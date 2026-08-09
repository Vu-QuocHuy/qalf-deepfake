"""Video-level binary metrics using the project convention score=P(fake)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    index = int(np.argmax((true_positive - false_positive)[finite]))
    return float(thresholds[finite][index])


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    if np.unique(labels).size == 2:
        metrics["auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
        false_positive, true_positive, _ = roc_curve(labels, scores)
        false_negative = 1.0 - true_positive
        index = int(np.argmin(np.abs(false_positive - false_negative)))
        metrics["eer"] = float((false_positive[index] + false_negative[index]) / 2.0)
    else:
        metrics.update({"auc": float("nan"), "average_precision": float("nan"), "eer": float("nan")})
    return metrics
