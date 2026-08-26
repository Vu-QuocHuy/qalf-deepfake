from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_sbi_ablation import (
    collect_runs,
    paired_differences,
    summarize_runs,
)


def _write_metrics(path: Path, *, dataset: str, auc: float, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "threshold": 0.5,
        "sample_count": 10,
        "auc": auc,
        "accuracy": accuracy,
        "balanced_accuracy": accuracy,
        "f1_macro": accuracy,
        "eer": 1.0 - auc,
        "apcer": 0.1,
        "bpcer": 0.2,
        "acer": 0.15,
    }
    path.write_text(
        json.dumps({"metrics": metrics, "protocol": {"datasets": [dataset]}}),
        encoding="utf-8",
    )


class SbiAblationSummaryTests(unittest.TestCase):
    def test_collect_summarize_and_pair_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ffpp_root = root / "ffpp_test"
            for dataset, dataset_name, dataset_root in (
                ("celebdf", "celebdf_v2", root),
                ("ffpp", "ffpp", ffpp_root),
            ):
                for profile, offset in (("baseline", 0.0), ("sbi_frame", 0.01)):
                    for seed, auc in ((0, 0.80), (17, 0.84)):
                        if dataset == "celebdf":
                            directory = (
                                f"{profile}_seed{seed}_to_celebdf_8f_3clips_"
                                "mean_eer_tta_ffpp_threshold"
                            )
                        else:
                            directory = (
                                f"{profile}_seed{seed}_ffpp_test_8f_3clips_"
                                "mean_eer_tta"
                            )
                        _write_metrics(
                            dataset_root / directory / "metrics.json",
                            dataset=dataset_name,
                            auc=auc + offset,
                            accuracy=0.70 + offset,
                        )

            runs, missing = collect_runs(
                celebdf_root=root,
                ffpp_root=ffpp_root,
                profiles=["baseline", "sbi_frame"],
                seeds=[0, 17],
                texture_frames=8,
                clips_per_video=3,
                aggregation="mean",
                threshold_selection="eer",
                flip_tta=True,
                allow_missing=False,
            )
            self.assertEqual(len(runs), 8)
            self.assertEqual(missing, [])

            summary = summarize_runs(runs)
            celeb_baseline = next(
                row
                for row in summary
                if row["dataset"] == "celebdf" and row["profile"] == "baseline"
            )
            self.assertEqual(celeb_baseline["n_seeds"], 2)
            self.assertAlmostEqual(celeb_baseline["auc_mean"], 0.82)

            details, paired = paired_differences(runs, reference="baseline")
            self.assertEqual(len(details), 4)
            celeb_delta = next(row for row in paired if row["dataset"] == "celebdf")
            self.assertEqual(celeb_delta["n_paired_seeds"], 2)
            self.assertAlmostEqual(celeb_delta["auc_delta_mean"], 0.01)

    def test_missing_runs_are_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                collect_runs(
                    celebdf_root=root,
                    ffpp_root=root / "ffpp_test",
                    profiles=["baseline"],
                    seeds=[0],
                    texture_frames=8,
                    clips_per_video=3,
                    aggregation="mean",
                    threshold_selection="eer",
                    flip_tta=True,
                    allow_missing=False,
                )


if __name__ == "__main__":
    unittest.main()
