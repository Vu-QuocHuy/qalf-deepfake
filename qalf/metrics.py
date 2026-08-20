"""Video-level binary metrics using the project convention score=P(fake)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    strategy: str = "eer",
) -> float:
    """Select a validation threshold without using the target test set.

    ``eer`` selects the finite ROC threshold whose FPR and FNR are closest;
    this is a reproducible closest-to-EER operating point for discrete scores.
    """

    if strategy != "eer":
        raise ValueError("Only the EER threshold strategy is supported")
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    false_negative = 1.0 - true_positive
    index = int(np.argmin(np.abs(false_positive - false_negative)[finite]))
    return float(thresholds[finite][index])


def compute_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Compute video-level metrics with ``1=fake`` as the positive label.

    The class-specific keys are kept for auditability, while the generic
    ``precision``, ``recall`` and ``f1`` aliases all refer to the positive
    (fake) class.  This matches the binary convention used by the paper tables.
    """

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))

    def safe_ratio(numerator: float | int, denominator: float | int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision_fake = safe_ratio(true_positive, true_positive + false_positive)
    precision_real = safe_ratio(true_negative, true_negative + false_negative)
    recall_fake = safe_ratio(true_positive, true_positive + false_negative)
    recall_real = safe_ratio(true_negative, true_negative + false_positive)
    f1_real = safe_ratio(
        2 * precision_real * recall_real,
        precision_real + recall_real,
    )
    f1_fake = safe_ratio(
        2 * precision_fake * recall_fake,
        precision_fake + recall_fake,
    )
    apcer = 1.0 - recall_fake
    bpcer = 1.0 - recall_real
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": precision_fake,
        "recall": recall_fake,
        "f1": f1_fake,
        "f1_fake": f1_fake,
        "f1_real": f1_real,
        "f1_macro": 0.5 * (f1_real + f1_fake),
        "precision_fake": precision_fake,
        "precision_real": precision_real,
        "recall_fake": recall_fake,
        "recall_real": recall_real,
        "specificity": recall_real,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": 0.5 * (apcer + bpcer),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "real_count": true_negative + false_positive,
        "fake_count": true_positive + false_negative,
        "sample_count": int(labels.size),
    }
    if np.unique(labels).size == 2:
        metrics["auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        false_negative_rate = 1.0 - true_positive_rate
        index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
        metrics["eer"] = float(
            (false_positive_rate[index] + false_negative_rate[index]) / 2.0
        )
    else:
        metrics.update(
            {
                "auc": float("nan"),
                "average_precision": float("nan"),
                "eer": float("nan"),
            }
        )
    return metrics
