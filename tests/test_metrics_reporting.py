from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qalf.metrics import compute_metrics, select_threshold
from qalf.reporting import format_evaluation_report, save_evaluation_plots


class ComprehensiveMetricTests(unittest.TestCase):
    def test_eer_threshold_is_finite_and_in_score_range(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.3, 0.7, 0.9])
        threshold = select_threshold(labels, scores, strategy="eer")
        self.assertTrue(0.0 <= threshold <= 1.0)

    def test_unknown_threshold_strategy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_threshold(np.asarray([0, 1]), np.asarray([0.1, 0.9]), "bad")

    def test_confusion_counts_and_class_metrics_use_fake_as_positive(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.8, 0.2, 0.9])

        metrics = compute_metrics(labels, scores, threshold=0.5)

        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(float(metrics["precision_fake"]), 0.5)
        self.assertAlmostEqual(float(metrics["recall_fake"]), 0.5)
        self.assertAlmostEqual(float(metrics["recall_real"]), 0.5)
        self.assertAlmostEqual(float(metrics["f1_fake"]), 0.5)
        self.assertAlmostEqual(float(metrics["f1_real"]), 0.5)
        self.assertAlmostEqual(float(metrics["acer"]), 0.5)


class ReportingTests(unittest.TestCase):
    def test_report_has_stable_section_order(self) -> None:
        report = format_evaluation_report(
            {
                "threshold": 0.4,
                "auc": 0.8,
                "sample_count": 10,
                "true_negative": 4,
            },
            context={"Dataset": "celebdf"},
        )

        self.assertLess(report.index("RANKING METRICS"), report.index("OPERATING POINT"))
        self.assertLess(report.index("OPERATING POINT"), report.index("CONFUSION COUNTS"))
        self.assertIn("Dataset", report)
        self.assertIn("celebdf", report)

    def test_evaluation_plots_are_written(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.3, 0.7, 0.9])
        with tempfile.TemporaryDirectory() as temporary:
            filenames = save_evaluation_plots(labels, scores, 0.5, temporary)
            self.assertEqual(len(filenames), 5)
            for filename in filenames:
                self.assertTrue((Path(temporary) / filename).is_file())


if __name__ == "__main__":
    unittest.main()
