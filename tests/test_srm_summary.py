from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SRMSummaryTests(unittest.TestCase):
    def test_summary_accepts_legacy_controls_and_applies_registered_gate(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        profiles = (
            (
                "qalf_ffpp4_effb0_160_8f_full_face_sbi_texture_only",
                {"fusion_mode": "texture"},
                {
                    "auc": 0.830,
                    "average_precision": 0.900,
                    "eer": 0.250,
                    "balanced_accuracy": 0.750,
                    "texture_auc": 0.830,
                },
            ),
            (
                "qalf_ffpp4_effb0_160_8f_full_face_sbi",
                {"fusion_mode": "quality"},
                {
                    "auc": 0.832,
                    "average_precision": 0.901,
                    "eer": 0.249,
                    "balanced_accuracy": 0.751,
                    "geometry_auc": 0.600,
                    "texture_auc": 0.831,
                    "mean_geometry_weight": 0.100,
                    "zero_geometry_auc": 0.828,
                },
            ),
            (
                "qalf_ffpp4_effb0_160_8f_full_face_sbi_srm",
                {"auxiliary_branch": "srm"},
                {
                    "auc": 0.841,
                    "average_precision": 0.910,
                    "eer": 0.240,
                    "balanced_accuracy": 0.760,
                    "auxiliary_auc": 0.650,
                    "texture_auc": 0.834,
                    "mean_auxiliary_weight": 0.100,
                    "zero_auxiliary_auc": 0.835,
                },
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (experiment, model_protocol, metrics) in enumerate(profiles):
                training = root / experiment
                training.mkdir()
                if index > 0:
                    (training / "training_summary.json").write_text(
                        json.dumps({"best_value": 0.970}),
                        encoding="utf-8",
                    )
                evaluation = root / (
                    f"{experiment}_to_celebdf_12f_3clips_mean_tta_ffpp_threshold"
                )
                evaluation.mkdir()
                (evaluation / "metrics.json").write_text(
                    json.dumps(
                        {
                            "metrics": metrics,
                            "protocol": {"model": model_protocol},
                        }
                    ),
                    encoding="utf-8",
                )

            subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "summarize_srm_ablation.py"),
                    "--experiments-root",
                    str(root),
                    "--seed",
                    "42",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = (root / "srm_ablation_seed42.md").read_text(encoding="utf-8")
            self.assertIn("SRM PASSES seed-42 gate", report)
            self.assertIn("+0.0060", report)


if __name__ == "__main__":
    unittest.main()
