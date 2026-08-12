#!/usr/bin/env python3
"""Diagnose whether SRM residual scores complement RGB texture scores."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.config import save_json
from qalf.metrics import compute_metrics, select_threshold

BRANCH_FIELDS = {
    "fused": "score",
    "auxiliary": "auxiliary_score",
    "texture": "texture_score",
}


def _read_predictions(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No predictions in {path}")
    required = {"label", *BRANCH_FIELDS.values()}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in required
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    pearson = float(np.corrcoef(left, right)[0, 1])
    spearman = float(np.corrcoef(_average_ranks(left), _average_ranks(right))[0, 1])
    return pearson, spearman


def _branch_metrics(
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float | int]], dict[str, float]]:
    val_labels = validation["label"].astype(np.int64)
    test_labels = test["label"].astype(np.int64)
    metrics: dict[str, dict[str, float | int]] = {}
    thresholds: dict[str, float] = {}
    for branch, field in BRANCH_FIELDS.items():
        threshold = float(select_threshold(val_labels, validation[field]))
        thresholds[branch] = threshold
        metrics[branch] = compute_metrics(test_labels, test[field], threshold)
    return metrics, thresholds


def _select_linear_blend(
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, float | int]:
    labels = validation["label"].astype(np.int64)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for texture_weight in np.linspace(0.0, 1.0, 21):
        scores = (
            texture_weight * validation["texture_score"]
            + (1.0 - texture_weight) * validation["auxiliary_score"]
        )
        auc = float(compute_metrics(labels, scores, 0.5)["auc"])
        candidates.append((auc, float(texture_weight), scores))
    validation_auc, texture_weight, validation_scores = max(
        candidates,
        key=lambda row: (row[0], row[1]),
    )
    threshold = float(select_threshold(labels, validation_scores))
    test_scores = (
        texture_weight * test["texture_score"]
        + (1.0 - texture_weight) * test["auxiliary_score"]
    )
    result = compute_metrics(test["label"].astype(np.int64), test_scores, threshold)
    result.update(
        {
            "texture_weight": texture_weight,
            "auxiliary_weight": 1.0 - texture_weight,
            "ffpp_validation_auc": validation_auc,
        }
    )
    return result


def _error_complementarity(
    test: dict[str, np.ndarray], thresholds: dict[str, float]
) -> dict[str, float | int]:
    labels = test["label"].astype(np.int64)
    texture_correct = (test["texture_score"] >= thresholds["texture"]).astype(int) == labels
    auxiliary_correct = (
        test["auxiliary_score"] >= thresholds["auxiliary"]
    ).astype(int) == labels
    total = len(labels)
    return {
        "both_correct": int(np.sum(texture_correct & auxiliary_correct)),
        "texture_only_correct": int(np.sum(texture_correct & ~auxiliary_correct)),
        "auxiliary_only_correct": int(np.sum(~texture_correct & auxiliary_correct)),
        "both_wrong": int(np.sum(~texture_correct & ~auxiliary_correct)),
        "disagreement_fraction": float(np.mean(texture_correct != auxiliary_correct)),
        "oracle_accuracy": float(np.mean(texture_correct | auxiliary_correct)),
        "sample_count": total,
    }


def _gate_trajectory(history_path: Path | None, output_prefix: Path) -> dict[str, object] | None:
    if history_path is None or not history_path.is_file():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = [
        {
            "epoch": int(row["epoch"]),
            "fused_auc": float(row["validation"]["auc"]),
            "auxiliary_auc": float(row["validation"]["auxiliary_auc"]),
            "texture_auc": float(row["validation"]["texture_auc"]),
            "auxiliary_weight": float(row["validation"]["mean_auxiliary_weight"]),
        }
        for row in history
    ]
    with output_prefix.with_name(output_prefix.name + "_gate_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    minimum = min(rows, key=lambda row: row["auxiliary_weight"])
    maximum = max(rows, key=lambda row: row["auxiliary_weight"])
    return {
        "first": rows[0],
        "last": rows[-1],
        "minimum_auxiliary_weight": minimum,
        "maximum_auxiliary_weight": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--history")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    evaluation = Path(args.evaluation_dir)
    validation = _read_predictions(evaluation / "threshold_predictions.csv")
    test = _read_predictions(evaluation / "predictions.csv")
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    branch_metrics, thresholds = _branch_metrics(validation, test)
    val_pearson, val_spearman = _correlation(
        validation["texture_score"], validation["auxiliary_score"]
    )
    test_pearson, test_spearman = _correlation(
        test["texture_score"], test["auxiliary_score"]
    )
    payload = {
        "protocol": {
            "threshold_selection": "Youden-J independently on FF++ validation",
            "linear_blend_selection": "texture weight grid 0.00..1.00 step 0.05 on FF++ validation",
            "status": "post-hoc diagnostic; not final confirmatory evidence",
        },
        "thresholds": thresholds,
        "celebdf_branch_metrics": branch_metrics,
        "score_correlation": {
            "ffpp_validation_pearson": val_pearson,
            "ffpp_validation_spearman": val_spearman,
            "celebdf_pearson": test_pearson,
            "celebdf_spearman": test_spearman,
        },
        "celebdf_error_complementarity": _error_complementarity(test, thresholds),
        "ffpp_selected_linear_blend": _select_linear_blend(validation, test),
        "gate_trajectory": _gate_trajectory(
            Path(args.history) if args.history else None,
            output_prefix,
        ),
    }
    save_json(payload, output_prefix.with_suffix(".json"))

    complementarity = payload["celebdf_error_complementarity"]
    blend = payload["ffpp_selected_linear_blend"]
    lines = [
        "# SRM complementarity diagnostic",
        "",
        "> Post-hoc diagnostic only. Thresholds and blend weight use FF++ validation; "
        "Celeb-DF is not used for fitting.",
        "",
        "| Score | FF++ threshold | Celeb-DF AUC | Balanced acc | F1 fake | EER |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for branch in ("texture", "auxiliary", "fused"):
        metrics = branch_metrics[branch]
        lines.append(
            f"| {branch} | {thresholds[branch]:.4f} | {float(metrics['auc']):.4f} | "
            f"{float(metrics['balanced_accuracy']):.4f} | {float(metrics['f1_fake']):.4f} | "
            f"{float(metrics['eer']):.4f} |"
        )
    lines.extend(
        (
            "",
            "## Score correlation",
            "",
            f"- FF++ validation Pearson/Spearman: {val_pearson:.4f}/{val_spearman:.4f}",
            f"- Celeb-DF Pearson/Spearman: {test_pearson:.4f}/{test_spearman:.4f}",
            "",
            "## Error complementarity on Celeb-DF",
            "",
            f"- Both correct: {complementarity['both_correct']}",
            f"- Texture only correct: {complementarity['texture_only_correct']}",
            f"- SRM only correct: {complementarity['auxiliary_only_correct']}",
            f"- Both wrong: {complementarity['both_wrong']}",
            f"- Oracle accuracy: {float(complementarity['oracle_accuracy']):.4f}",
            "",
            "## FF++-selected linear score blend",
            "",
            f"- Texture/SRM weight: {float(blend['texture_weight']):.2f}/"
            f"{float(blend['auxiliary_weight']):.2f}",
            f"- FF++ validation AUC: {float(blend['ffpp_validation_auc']):.4f}",
            f"- Celeb-DF AUC: {float(blend['auc']):.4f}",
            f"- Celeb-DF balanced accuracy: {float(blend['balanced_accuracy']):.4f}",
        )
    )
    report = "\n".join(lines) + "\n"
    output_prefix.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report)
    print(f"JSON: {output_prefix.with_suffix('.json')}")
    print(f"Markdown: {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
