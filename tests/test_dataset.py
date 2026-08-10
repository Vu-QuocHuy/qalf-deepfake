import random
import unittest

import numpy as np

from qalf.data.dataset import _augment_geometry_landmarks
from qalf.data.geometry import build_geometry_features


class GeometryAugmentationTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.landmarks = rng.normal(size=(32, 468, 3)).astype(np.float32)
        self.detected = np.ones(32, dtype=bool)

    def test_corruption_is_recoverable_by_feature_builder(self) -> None:
        random.seed(11)
        np.random.seed(11)
        augmented, detected = _augment_geometry_landmarks(
            self.landmarks,
            self.detected,
            {
                "noise_probability": 1.0,
                "noise_std": 0.01,
                "drift_probability": 1.0,
                "drift_std": 0.001,
                "frame_dropout_probability": 1.0,
                "max_frame_dropout_ratio": 0.15,
                "point_dropout_probability": 0.01,
            },
        )
        self.assertFalse(np.array_equal(augmented, self.landmarks, equal_nan=True))
        self.assertGreater(int((~detected).sum()), 0)
        features, quality = build_geometry_features(
            augmented,
            detected,
            np.arange(32, dtype=np.float32) / 8.0,
            feature_mode="aligned_motion_3d",
        )
        self.assertTrue(np.isfinite(features).all())
        self.assertLess(float(quality[0]), 1.0)
        self.assertLess(float(quality[1]), 1.0)

    def test_unknown_setting_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _augment_geometry_landmarks(
                self.landmarks,
                self.detected,
                {"not_a_setting": 1.0},
            )

    def test_explicit_rng_makes_evaluation_corruption_deterministic(self) -> None:
        config = {
            "noise_probability": 1.0,
            "noise_std": 0.01,
            "drift_probability": 0.0,
            "frame_dropout_probability": 1.0,
            "max_frame_dropout_ratio": 0.1,
            "point_dropout_probability": 0.01,
        }
        first = _augment_geometry_landmarks(
            self.landmarks,
            self.detected,
            config,
            python_rng=random.Random(91),
            numpy_rng=np.random.default_rng(91),
        )
        second = _augment_geometry_landmarks(
            self.landmarks,
            self.detected,
            config,
            python_rng=random.Random(91),
            numpy_rng=np.random.default_rng(91),
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
