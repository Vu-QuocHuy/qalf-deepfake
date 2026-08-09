import unittest

import numpy as np

from qalf.data.geometry import DEFAULT_LANDMARK_INDICES, build_geometry_features, geometry_input_dim


class GeometryTest(unittest.TestCase):
    def test_rigid_translation_is_removed(self):
        rng = np.random.default_rng(42)
        base = rng.normal(size=(468, 3)).astype(np.float32)
        sequence = np.stack([base + np.asarray([step * 0.1, step * 0.2, 0]) for step in range(8)])
        features, quality = build_geometry_features(
            sequence,
            np.ones(8, dtype=bool),
            np.arange(8, dtype=np.float32) / 10.0,
        )
        self.assertEqual(features.shape, (8, geometry_input_dim()))
        self.assertEqual(quality.shape, (5,))
        point_count = len(DEFAULT_LANDMARK_INDICES)
        velocity = features[:, point_count * 2 : point_count * 4]
        self.assertLess(float(np.abs(velocity).mean()), 1e-4)

    def test_feature_ablation_dimensions(self):
        expected_multipliers = {
            "normalized": 2,
            "aligned": 2,
            "motion_only": 4,
            "aligned_motion": 6,
        }
        for mode, multiplier in expected_multipliers.items():
            self.assertEqual(
                geometry_input_dim(feature_mode=mode),
                len(DEFAULT_LANDMARK_INDICES) * multiplier + 1,
            )


if __name__ == "__main__":
    unittest.main()
