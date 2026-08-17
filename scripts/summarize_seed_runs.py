"""Summarize canonical baseline metrics across independent random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "auc",
    "average_precision",
    "eer",
    "balanced_accuracy",
    "accuracy",
    "f1_fake",
    "acer",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any) -> float:
    return float(value)


def _format(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _summary(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-prefix", required=True)
    parser.add_argument("--eval-suffix", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        train_dir = Path(f"{args.train_prefix}{seed}")
        eval_dir = Path(f"{train_dir}{args.eval_suffix}")
        train_summary_path = train_dir / "training_summary.json"
        metrics_path = eval_dir / "metrics.json"
        row: dict[str, Any] = {"seed": seed, "status": "complete"}
        missing: list[str] = []
        if train_summary_path.is_file():
            train_summary = _load_json(train_summary_path)
            row["best_epoch"] = train_summary.get("best_epoch")
            row["ffpp_val_auc"] = train_summary.get("best_value")
        else:
            missing.append("train_summary")
        if metrics_path.is_file():
            evaluation = _load_json(metrics_path).get("metrics", {})
            for key in METRIC_KEYS:
                row[key] = evaluation.get(key)
        else:
            missing.append("eval_metrics")
        if missing:
            row["status"] = "missing_" + "+".join(missing)
        rows.append(row)

    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "status", "best_epoch", "ffpp_val_auc", *METRIC_KEYS]
    with (output_stem.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_lines = [
        "# TextureSBI baseline multi-seed summary",
        "",
        "| Seed | Status | Best epoch | FF++ val AUC | Celeb-DF AUC | AP | EER | Balanced acc | Accuracy | F1 fake | ACER |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        markdown_lines.append(
            "| {seed} | {status} | {best_epoch} | {ffpp_val_auc} | {auc} | {average_precision} | "
            "{eer} | {balanced_accuracy} | {accuracy} | {f1_fake} | {acer} |".format(
                seed=row["seed"],
                status=row["status"],
                best_epoch=row.get("best_epoch", "NA"),
                ffpp_val_auc=_format(row.get("ffpp_val_auc")),
                auc=_format(row.get("auc")),
                average_precision=_format(row.get("average_precision")),
                eer=_format(row.get("eer")),
                balanced_accuracy=_format(row.get("balanced_accuracy")),
                accuracy=_format(row.get("accuracy")),
                f1_fake=_format(row.get("f1_fake")),
                acer=_format(row.get("acer")),
            )
        )

    markdown_lines.extend(("", "## Mean ± standard deviation", ""))
    complete_rows = [row for row in rows if row["status"] == "complete"]
    for key in ("ffpp_val_auc", *METRIC_KEYS):
        values = [_number(row[key]) for row in complete_rows if row.get(key) is not None]
        mean, std = _summary(values)
        label = "FF++ val AUC" if key == "ffpp_val_auc" else key
        markdown_lines.append(f"- {label}: {_format(mean)} ± {_format(std)}")
    markdown_lines.extend(("", f"CSV: {output_stem.with_suffix('.csv')}", ""))
    output_stem.with_suffix(".md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    print("# TextureSBI baseline multi-seed summary")
    for line in markdown_lines[4:]:
        if line.startswith("CSV:"):
            break
        print(line)
    print(f"CSV: {output_stem.with_suffix('.csv')}")
    print(f"Markdown: {output_stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
