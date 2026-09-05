#!/usr/bin/env python3
"""Join server and Pi4 Table IV(a) AUC CSVs and report per-seed gaps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _compare(server_rows: list[dict[str, str]], pi4_rows: list[dict[str, str]]) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    server = {int(row["seed"]): float(row["auc"]) for row in server_rows}
    pi4 = {int(row["seed"]): float(row["auc"]) for row in pi4_rows}
    if server.keys() != pi4.keys():
        raise ValueError(f"Seed mismatch: server={sorted(server)}, pi4={sorted(pi4)}")
    rows = [{"seed": seed, "server_auc": server[seed], "pi4_auc": pi4[seed],
             "delta_pp": round(100 * (server[seed] - pi4[seed]), 4)} for seed in sorted(server)]
    deltas = [float(row["delta_pp"]) for row in rows]
    return rows, {"delta_pp_mean": statistics.mean(deltas),
                  "delta_pp_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-csv", required=True, type=Path)
    parser.add_argument("--pi4-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    with args.server_csv.open(encoding="utf-8", newline="") as handle:
        server_rows = list(csv.DictReader(handle))
    with args.pi4_csv.open(encoding="utf-8", newline="") as handle:
        pi4_rows = list(csv.DictReader(handle))
    rows, summary = _compare(server_rows, pi4_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "server_auc", "pi4_auc", "delta_pp"])
        writer.writeheader()
        writer.writerows(rows)
    args.output_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Mean server-minus-Pi4 AUC gap: {summary['delta_pp_mean']:.4f} ± {summary['delta_pp_std']:.4f} pp")


if __name__ == "__main__":
    main()
