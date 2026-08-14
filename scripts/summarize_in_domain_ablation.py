#!/usr/bin/env python3
"""Summarize FF++ in-domain evaluation runs for registered ablation profiles."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "auc",
    "average_precision",
    "eer",
    "balanced_accuracy",
    "accuracy",
    "f1_fake",
    "acer",
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected PROFILE=EVAL_DIR, got: {spec!r}")
    profile, directory = spec.split("=", 1)
    if not profile or not directory:
        raise ValueError(f"Expected PROFILE=EVAL_DIR, got: {spec!r}")
    return profile, Path(directory)


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="PROFILE=DIR")
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for spec in args.run:
        profile, eval_dir = _parse_run(spec)
        metrics_path = eval_dir / "metrics.json"
        row: dict[str, Any] = {"profile": profile, "eval_dir": str(eval_dir), "status": "complete"}
        if not metrics_path.is_file():
            row["status"] = "missing_metrics"
            rows.append(row)
            continue
        payload = _load(metrics_path)
        metrics = payload.get("metrics", {})
        protocol = payload.get("protocol", {})
        row.update({key: metrics.get(key) for key in METRICS})
        row["threshold"] = metrics.get("threshold")
        row["fake_method_filter"] = ",".join(protocol.get("fake_method_filter") or [])
        checkpoint = Path(protocol.get("checkpoint", ""))
        summary_path = checkpoint.parent / "training_summary.json"
        if summary_path.is_file():
            training = _load(summary_path)
            row["ffpp_val_auc"] = training.get("best_value")
            row["best_epoch"] = training.get("best_epoch")
            if row.get("ffpp_val_auc") is not None and row.get("auc") is not None:
                row["domain_gap"] = float(row["ffpp_val_auc"]) - float(row["auc"])
        rows.append(row)

    stem = Path(args.output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "profile", "status", "best_epoch", "ffpp_val_auc", "auc", "domain_gap",
        "average_precision", "eer", "balanced_accuracy", "accuracy", "f1_fake",
        "acer", "threshold", "fake_method_filter", "eval_dir",
    ]
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# QALF FF++ in-domain ablation summary",
        "",
        "The evaluation split is the official FF++ test split. Thresholds are calibrated on FF++ validation only.",
        "FaceShifter is excluded; fake methods are listed per row.",
        "",
        "| Profile | Status | FF++ val AUC | FF++ test AUC | Gap | AP | EER | Balanced acc | ACER |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['status']} | {_fmt(row.get('ffpp_val_auc'))} | "
            f"{_fmt(row.get('auc'))} | {_fmt(row.get('domain_gap'))} | "
            f"{_fmt(row.get('average_precision'))} | {_fmt(row.get('eer'))} | "
            f"{_fmt(row.get('balanced_accuracy'))} | {_fmt(row.get('acer'))} |"
        )

    complete = [row for row in rows if row["status"] == "complete"]
    lines.extend(("", "## Mean ± standard deviation by profile", ""))
    for profile in sorted({str(row["profile"]) for row in complete}):
        selected = [row for row in complete if row["profile"] == profile]
        values: list[str] = []
        for key in ("ffpp_val_auc", "auc", "average_precision", "eer", "balanced_accuracy", "acer"):
            numeric = [float(row[key]) for row in selected if row.get(key) is not None]
            if not numeric:
                values.append("NA")
            elif len(numeric) == 1:
                values.append(f"{numeric[0]:.4f} ± 0.0000")
            else:
                values.append(f"{statistics.mean(numeric):.4f} ± {statistics.stdev(numeric):.4f}")
        lines.append(
            f"- **{profile}** — val AUC {values[0]}, test AUC {values[1]}, AP {values[2]}, "
            f"EER {values[3]}, balanced acc {values[4]}, ACER {values[5]}"
        )
    lines.extend(("", f"CSV: `{stem.with_suffix('.csv')}`", ""))
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"CSV: {stem.with_suffix('.csv')}")
    print(f"Markdown: {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
