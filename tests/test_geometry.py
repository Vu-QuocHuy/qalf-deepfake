import unittest

import numpy as np

from qalf.data.geometry import (
    DEFAULT_LANDMARK_INDICES,
    _alignment_medoid,
    build_geometry_features,
    geometry_input_dim,
)


def _rotation(axis: str, angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=np.float32)
    if axis == "y":
        return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float32)
    if axis == "z":
        return np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float32)
    raise ValueError(axis)


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.base = np.empty((468, 3), dtype=np.float32)
        self.base[:, 0] = rng.normal(0.0, 0.45, 468)
        self.base[:, 1] = rng.normal(0.0, 0.60, 468)
        self.base[:, 2] = rng.normal(0.0, 0.20, 468)
        self.detected = np.ones(8, dtype=bool)
        self.timestamps = np.arange(8, dtype=np.float32) / 8.0

    def _rigid_sequence(self, axis: str) -> np.ndarray:
        frames = []
        for step, angle in enumerate(np.deg2rad(np.linspace(0, 25, 8))):
            rotated = self.base @ _rotation(axis, float(angle)).T
            scale = 1.0 + step * 0.025
            translation = np.asarray([step * 0.03, -step * 0.02, step * 0.01])
            frames.append(rotated * scale + translation)
        return np.stack(frames).astype(np.float32)

    def test_legacy_2d_alignment_removes_translation_scale_and_roll(self) -> None:
        sequence = self._rigid_sequence("z")
        features, quality = build_geometry_features(
            sequence,
            self.detected,
            self.timestamps,
            feature_mode="aligned_motion",
        )
        point_count = len(DEFAULT_LANDMARK_INDICES)
        velocity = features[:, point_count * 2 : point_count * 4]
        self.assertLess(float(np.abs(velocity).mean()), 1e-5)
        self.assertEqual(quality.shape, (5,))

    def test_legacy_2d_alignment_does_not_remove_yaw(self) -> None:
        sequence = self._rigid_sequence("y")
        features, _ = build_geometry_features(
            sequence,
            self.detected,
            self.timestamps,
            feature_mode="aligned_motion",
        )
        point_count = len(DEFAULT_LANDMARK_INDICES)
        velocity = features[:, point_count * 2 : point_count * 4]
        self.assertGreater(float(np.abs(velocity).mean()), 1e-4)

    def test_3d_alignment_removes_roll_yaw_and_pitch(self) -> None:
        point_count = len(DEFAULT_LANDMARK_INDICES)
        for axis in ("x", "y", "z"):
            with self.subTest(axis=axis):
                features, _ = build_geometry_features(
                    self._rigid_sequence(axis),
                    self.detected,
                    self.timestamps,
                    feature_mode="aligned_motion_3d",
                )
                velocity = features[:, point_count * 3 : point_count * 6]
                self.assertLess(float(np.abs(velocity).mean()), 1e-5)

    def test_3d_alignment_preserves_non_rigid_mouth_motion(self) -> None:
        sequence = self._rigid_sequence("y")
        mouth_index = DEFAULT_LANDMARK_INDICES.index(13)
        sequence[:, 13, 1] += np.linspace(0.0, 0.20, len(sequence), dtype=np.float32)
        features, _ = build_geometry_features(
            sequence,
            self.detected,
            self.timestamps,
            feature_mode="motion_3d",
        )
        mouth_y_velocity = features[:, mouth_index * 3 + 1]
        self.assertGreater(float(np.abs(mouth_y_velocity).mean()), 1e-3)

    def test_alignment_reference_is_not_forced_to_noisy_first_frame(self) -> None:
        clean = self.base - self.base.mean(axis=0, keepdims=True)
        clean /= np.sqrt(np.mean(np.sum(clean**2, axis=1)))
        frames = np.repeat(clean[None, ...], 5, axis=0)
        frames[0] = frames[0].copy()
        frames[0, :40] += 2.0
        reference = _alignment_medoid(frames)
        self.assertLess(float(np.sqrt(np.mean((reference - frames[1]) ** 2))), 1e-5)

    def test_individual_missing_point_is_interpolated(self) -> None:
        sequence = np.repeat(self.base[None, ...], 8, axis=0)
        sequence[3:5, 13, 1] = np.nan
        features, quality = build_geometry_features(
            sequence,
            self.detected,
            self.timestamps,
            feature_mode="aligned_motion_3d",
        )
        self.assertTrue(np.isfinite(features).all())
        self.assertLess(float(quality[1]), 1.0)

    def test_feature_ablation_dimensions(self) -> None:
        expected_multipliers = {
            "normalized": 2,
            "aligned": 2,
            "motion_only": 4,
            "aligned_motion": 6,
            "aligned_3d": 3,
            "motion_3d": 6,
            "aligned_motion_3d": 9,
        }
        for mode, multiplier in expected_multipliers.items():
            self.assertEqual(
                geometry_input_dim(feature_mode=mode),
                len(DEFAULT_LANDMARK_INDICES) * multiplier + 1,
            )


if __name__ == "__main__":
    unittest.main()
