"""Human-readable reports, reproducibility metadata, and diagnostic plots."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

REPORT_SECTIONS = (
    (
        "RANKING METRICS (threshold independent)",
        (
            ("auc", "AUC ROC"),
            ("average_precision", "Average precision"),
            ("eer", "Equal error rate"),
        ),
    ),
    (
        "OPERATING POINT (threshold dependent)",
        (
            ("threshold", "Decision threshold"),
            ("balanced_accuracy", "Balanced accuracy"),
            ("accuracy", "Accuracy"),
            ("f1_fake", "F1 fake"),
            ("f1_real", "F1 real"),
            ("f1_macro", "F1 macro"),
            ("precision_fake", "Precision fake"),
            ("precision_real", "Precision real"),
            ("recall_fake", "Recall fake"),
            ("recall_real", "Recall real"),
            ("apcer", "APCER (fake predicted real)"),
            ("bpcer", "BPCER (real predicted fake)"),
            ("acer", "ACER"),
        ),
    ),
    (
        "QALF BRANCH DIAGNOSTICS",
        (
            ("auxiliary_auc", "Auxiliary AUC"),
            ("texture_auc", "Texture AUC"),
            ("fused_auc", "Fused AUC"),
            ("fixed_average_auc", "Fixed-average AUC"),
            ("mean_auxiliary_weight", "Mean auxiliary weight"),
            ("mean_texture_weight", "Mean texture weight"),
            ("median_auxiliary_weight", "Median auxiliary weight"),
            ("p90_auxiliary_weight", "P90 auxiliary weight"),
            ("p95_auxiliary_weight", "P95 auxiliary weight"),
            ("max_auxiliary_weight", "Maximum auxiliary weight"),
            ("auxiliary_weight_above_0_05_fraction", "Auxiliary weight > 0.05 fraction"),
            ("mean_auxiliary_weight_real", "Mean auxiliary weight — real"),
            ("mean_auxiliary_weight_fake", "Mean auxiliary weight — fake"),
        ),
    ),
    (
        "ZERO-AUXILIARY COUNTERFACTUAL",
        (
            ("zero_auxiliary_auc", "AUC with auxiliary zeroed"),
            ("auc_gain_over_zero_auxiliary", "Normal minus zero-auxiliary AUC"),
            ("mean_abs_zero_auxiliary_score_shift", "Mean absolute score shift"),
            ("max_abs_zero_auxiliary_score_shift", "Maximum absolute score shift"),
        ),
    ),
    (
        "CONFUSION COUNTS (real=0, fake=1)",
        (
            ("true_negative", "TN — real predicted real"),
            ("false_positive", "FP — real predicted fake"),
            ("false_negative", "FN — fake predicted real"),
            ("true_positive", "TP — fake predicted fake"),
            ("real_count", "Real videos"),
            ("fake_count", "Fake videos"),
            ("sample_count", "Total videos"),
        ),
    ),
)


def format_evaluation_report(
    metrics: Mapping[str, float | int],
    *,
    title: str = "QALF EVALUATION RESULTS",
    context: Mapping[str, object] | None = None,
) -> str:
    """Render metrics in a stable, protocol-aware order."""

    width = 72
    lines = ["=" * width, title.center(width), "=" * width]
    if context:
        for label, value in context.items():
            lines.append(f"{label:<24} {value}")
        lines.append("-" * width)
    for section, entries in REPORT_SECTIONS:
        present = [(key, label) for key, label in entries if key in metrics]
        if not present:
            continue
        lines.append(section)
        for key, label in present:
            value = metrics[key]
            if isinstance(value, (int, np.integer)):
                rendered = f"{int(value):d}"
            else:
                rendered = f"{float(value):.4f}"
            lines.append(f"  {label:.<45} {rendered:>12}")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    lines.append("=" * width)
    return "\n".join(lines) + "\n"


def collect_run_metadata(
    argv: Sequence[str],
    config: Mapping[str, object],
    project_root: str | Path,
) -> dict[str, object]:
    """Collect enough environment information to identify an exact run."""

    root = Path(project_root)

    def git_output(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    try:
        torchvision_version = metadata.version("torchvision")
    except metadata.PackageNotFoundError:
        torchvision_version = None
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(argv),
        "config": dict(config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_names": gpu_names,
        },
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "dirty": bool(git_output("status", "--porcelain")),
        },
        "label_convention": "real=0,fake=1",
        "score_target": "fake",
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_training_history_plot(history: Sequence[Mapping[str, object]], path: str | Path) -> None:
    """Write a compact four-panel summary of all completed epochs."""

    if not history:
        return
    plt = _pyplot()
    epochs = [int(row["epoch"]) for row in history]

    def values(section: str, key: str) -> list[float]:
        return [float(row[section][key]) for row in history]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    loss_axis, rank_axis, operating_axis, fusion_axis = axes.flatten()
    eligible = [
        index for index, row in enumerate(history) if bool(row.get("selection_eligible", True))
    ]
    best_epoch = None
    if eligible:
        best_index = max(
            eligible,
            key=lambda index: float(history[index]["validation"]["auc"]),
        )
        best_epoch = epochs[best_index]
    for key, label in (
        ("loss", "Total"),
        ("fused", "Fused"),
        ("auxiliary", "Auxiliary"),
        ("texture", "Texture"),
    ):
        loss_axis.plot(epochs, values("train", key), label=label)
    loss_axis.set_title("Training losses")
    loss_axis.set_ylabel("BCE loss")

    for key, label in (
        ("auc", "Fused AUC"),
        ("average_precision", "Average precision"),
        ("auxiliary_auc", "Auxiliary AUC"),
        ("texture_auc", "Texture AUC"),
    ):
        rank_axis.plot(epochs, values("validation", key), label=label)
    rank_axis.set_title("Validation ranking metrics")
    rank_axis.set_ylim(0.0, 1.0)

    for key, label in (
        ("balanced_accuracy", "Balanced accuracy"),
        ("accuracy", "Accuracy"),
        ("f1", "F1 fake"),
        ("eer", "EER"),
    ):
        operating_axis.plot(epochs, values("validation", key), label=label)
    operating_axis.set_title("Validation operating metrics")
    operating_axis.set_ylim(0.0, 1.0)

    for key, label in (
        ("mean_auxiliary_weight", "Auxiliary weight"),
        ("mean_texture_weight", "Texture weight"),
        ("threshold", "Decision threshold"),
    ):
        fusion_axis.plot(epochs, values("validation", key), label=label)
    fusion_axis.set_title("Fusion and threshold")
    fusion_axis.set_ylim(0.0, 1.0)

    for axis in axes.flatten():
        if best_epoch is not None:
            axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.35)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_plots(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_dir: str | Path,
) -> list[str]:
    """Save standard ROC/PR, confusion matrices, and score distributions."""

    plt = _pyplot()
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    for values, name, value_format, colorbar_label in (
        (matrix, "confusion_matrix_counts.png", "d", "Videos"),
        (normalized, "confusion_matrix_normalized.png", ".1%", "Row proportion"),
    ):
        figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
        image = axis.imshow(
            values,
            cmap="Blues",
            vmin=0,
            vmax=1.0 if name == "confusion_matrix_normalized.png" else None,
        )
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    format(values[row, column], value_format),
                    ha="center",
                    va="center",
                    color="white" if values[row, column] > values.max() / 2 else "black",
                    fontsize=12,
                )
        axis.set_xticks((0, 1), labels=("Real", "Fake"))
        axis.set_yticks((0, 1), labels=("Real", "Fake"))
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(f"QALF confusion matrix @ threshold={threshold:.4f}")
        figure.colorbar(image, ax=axis, label=colorbar_label)
        figure.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append(name)

    if np.unique(labels).size == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        roc_auc = roc_auc_score(labels, scores)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"QALF (AUC={roc_auc:.4f})",
        )
        axis.plot((0, 1), (0, 1), "--", color="gray", label="Random")
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.savefig(output / "roc_curve.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append("roc_curve.png")

        precision, recall, _ = precision_recall_curve(labels, scores)
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        average_precision = average_precision_score(labels, scores)
        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=f"QALF (AP={average_precision:.4f})",
        )
        axis.axhline(labels.mean(), linestyle="--", color="gray", label="Class prior")
        axis.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.savefig(output / "precision_recall_curve.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append("precision_recall_curve.png")

    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.hist(scores[labels == 0], bins=30, range=(0, 1), alpha=0.65, label="Real")
    axis.hist(scores[labels == 1], bins=30, range=(0, 1), alpha=0.65, label="Fake")
    axis.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.4f}")
    axis.set(xlabel="Predicted P(fake)", ylabel="Videos", title="Video score distribution")
    axis.legend()
    figure.savefig(output / "score_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    written.append("score_distribution.png")
    return written
