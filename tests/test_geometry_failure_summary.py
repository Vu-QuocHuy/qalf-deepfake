from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_geometry_failure import _interpret


class GeometryFailureSummaryTests(unittest.TestCase):
    def test_decision_gate_has_three_explicit_outcomes(self) -> None:
        self.assertIn("DROP-ONLY COLLAPSE", _interpret(0.005, 0.0005))
        self.assertIn("GEOMETRY PRESERVED", _interpret(0.08, 0.004))
        self.assertIn("MIXED RESULT", _interpret(0.03, 0.002))

    def test_summary_validates_protocol_and_interprets_preserved_geometry(self) -> None:
        experiments = (
            ("qalf_ffpp4_effb0_160_8f_full_face_sbi", 0.0, 0.0, 0.09, 0.004),
            (
                "qalf_ffpp4_effb0_160_8f_sbi_geometry_dropout_only",
                0.15,
                0.0,
                0.08,
                0.004,
            ),
            (
                "qalf_ffpp4_effb0_160_8f_sbi_geometry_i2_reliability",
                0.15,
                0.10,
                0.002,
                0.0,
            ),
        )
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for (
                experiment,
                dropout,
                reliability,
                geometry_weight,
                counterfactual_gain,
            ) in experiments:
                training_dir = root / experiment
                training_dir.mkdir(parents=True)
                (training_dir / "training_summary.json").write_text(
                    json.dumps({"best_value": 0.97}), encoding="utf-8"
                )
                evaluation_dir = root / (
                    f"{experiment}_to_celebdf_12f_3clips_mean_tta_ffpp_threshold"
                )
                evaluation_dir.mkdir()
                metrics = {
                    "auc": 0.83,
                    "average_precision": 0.90,
                    "geometry_auc": 0.60,
                    "texture_auc": 0.82,
                    "mean_geometry_weight": geometry_weight,
                    "median_geometry_weight": geometry_weight,
                    "p90_geometry_weight": geometry_weight + 0.01,
                    "mean_geometry_weight_real": geometry_weight,
                    "mean_geometry_weight_fake": geometry_weight,
                    "zero_geometry_auc": 0.83 - counterfactual_gain,
                    "auc_gain_over_zero_geometry": counterfactual_gain,
                    "mean_abs_zero_geometry_score_shift": 0.01,
                }
                protocol = {
                    "model": {
                        "geometry_architecture": "tcn_mean",
                        "fusion_mode": "quality",
                        "modality_dropout_probability": dropout,
                    },
                    "training_data": {
                        "reliability_gate_loss_weight": reliability,
                    },
                }
                (evaluation_dir / "metrics.json").write_text(
                    json.dumps({"metrics": metrics, "protocol": protocol}),
                    encoding="utf-8",
                )

            output_prefix = root / "report"
            result = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "summarize_geometry_failure.py"),
                    "--experiments-root",
                    str(root),
                    "--output-prefix",
                    str(output_prefix),
                ],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("GEOMETRY PRESERVED", result.stdout)
            with output_prefix.with_suffix(".csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["profile"], "C — dropout only")


if __name__ == "__main__":
    unittest.main()
