"""Video-level binary metrics using the project convention score=P(fake).

The implementation is NumPy-only so training/evaluation does not need to
import the comparatively heavy SciPy/scikit-learn stack at startup.
"""

from __future__ import annotations

import numpy as np


def _validate_binary_inputs(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("labels and scores must have the same length")
    if labels.size == 0:
        raise ValueError("labels and scores cannot be empty")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    return labels, scores


def _threshold_groups(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), ordered_scores.size - 1]
    true_positive = np.cumsum(ordered_labels, dtype=np.int64)[ends]
    false_positive = (ends + 1) - true_positive
    return (
        ordered_scores[ends],
        true_positive.astype(np.float64),
        false_positive.astype(np.float64),
    )


def roc_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return FPR, TPR and descending score thresholds."""

    labels, scores = _validate_binary_inputs(labels, scores)
    positive = float(np.sum(labels == 1))
    negative = float(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        raise ValueError("ROC curve requires both classes")
    thresholds, true_positive, false_positive = _threshold_groups(labels, scores)
    return (
        np.r_[0.0, false_positive / negative],
        np.r_[0.0, true_positive / positive],
        np.r_[np.inf, thresholds],
    )


def precision_recall_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precision, recall and thresholds for the positive class."""

    labels, scores = _validate_binary_inputs(labels, scores)
    positive = float(np.sum(labels == 1))
    if positive == 0:
        raise ValueError("Precision-recall curve requires a positive class")
    thresholds, true_positive, false_positive = _threshold_groups(labels, scores)
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.ones_like(true_positive),
        where=(true_positive + false_positive) != 0,
    )
    recall = true_positive / positive
    # Match the conventional PR endpoint at recall=0.
    return np.r_[precision, 1.0], np.r_[recall, 0.0], thresholds


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(labels, scores)
    return float(np.trapezoid(true_positive, false_positive))


def average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _validate_binary_inputs(labels, scores)
    positive = float(np.sum(labels == 1))
    if positive == 0:
        return 0.0
    thresholds, true_positive, false_positive = _threshold_groups(labels, scores)
    del thresholds
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) != 0,
    )
    recall = true_positive / positive
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    return true_negative, false_positive, false_negative, true_positive


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _validate_binary_inputs(labels, scores)
    if np.unique(labels).size != 2:
        return 0.5
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    index = int(np.argmax((true_positive - false_positive)[finite]))
    return float(thresholds[finite][index])


def compute_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    labels, scores = _validate_binary_inputs(labels, scores)
    predictions = (scores >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = _confusion(
        labels, predictions
    )

    def safe_ratio(numerator: float | int, denominator: float | int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision_fake = safe_ratio(true_positive, true_positive + false_positive)
    precision_real = safe_ratio(true_negative, true_negative + false_negative)
    recall_fake = safe_ratio(true_positive, true_positive + false_negative)
    recall_real = safe_ratio(true_negative, true_negative + false_positive)
    f1_real = safe_ratio(2 * precision_real * recall_real, precision_real + recall_real)
    f1_fake = safe_ratio(2 * precision_fake * recall_fake, precision_fake + recall_fake)
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": safe_ratio(true_positive + true_negative, labels.size),
        "balanced_accuracy": 0.5 * (recall_real + recall_fake),
        "f1": f1_fake,
        "f1_fake": f1_fake,
        "f1_real": f1_real,
        "f1_macro": 0.5 * (f1_real + f1_fake),
        "precision_fake": precision_fake,
        "precision_real": precision_real,
        "recall_fake": recall_fake,
        "recall_real": recall_real,
        "specificity": recall_real,
        "apcer": 1.0 - recall_fake,
        "bpcer": 1.0 - recall_real,
        "acer": 0.5 * ((1.0 - recall_fake) + (1.0 - recall_real)),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "real_count": true_negative + false_positive,
        "fake_count": true_positive + false_negative,
        "sample_count": int(labels.size),
    }
    if np.unique(labels).size == 2:
        metrics["auc"] = roc_auc_score(labels, scores)
        metrics["average_precision"] = average_precision_score(labels, scores)
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        false_negative_rate = 1.0 - true_positive_rate
        index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
        metrics["eer"] = float(
            (false_positive_rate[index] + false_negative_rate[index]) / 2.0
        )
    else:
        metrics.update({"auc": float("nan"), "average_precision": float("nan"), "eer": float("nan")})
    return metrics
