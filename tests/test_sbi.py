from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qalf.data.dataset import QALFVideoDataset
from qalf.data.manifest import VideoRecord, write_manifest
from qalf.data.sbi import (
    SAMPLE_ORIGINAL_FAKE,
    SAMPLE_REAL,
    SAMPLE_SBI,
    face_mask_from_aligned_landmarks,
    generate_self_blended_clip,
    resolve_sbi_config,
    stratum_sampling_weights,
)


class SBIConfigTests(unittest.TestCase):
    def test_locked_mixture_is_complete(self) -> None:
        config = resolve_sbi_config({"enabled": True})

        self.assertTrue(config["enabled"])
        self.assertEqual(
            config["mixture"],
            {SAMPLE_REAL: 0.5, SAMPLE_ORIGINAL_FAKE: 0.25, SAMPLE_SBI: 0.25},
        )

    def test_invalid_mixture_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_sbi_config(
                {
                    "enabled": True,
                    "mixture": {
                        SAMPLE_REAL: 0.5,
                        SAMPLE_ORIGINAL_FAKE: 0.3,
                        SAMPLE_SBI: 0.3,
                    },
                }
            )

    def test_stratum_weights_have_the_locked_probability_mass(self) -> None:
        strata = [SAMPLE_REAL] * 4 + [SAMPLE_ORIGINAL_FAKE] * 8 + [SAMPLE_SBI] * 2
        mixture = resolve_sbi_config({"enabled": True})["mixture"]
        weights = stratum_sampling_weights(strata, mixture)

        mass = {
            name: sum(
                weight for weight, stratum in zip(weights, strata, strict=True) if stratum == name
            )
            for name in (SAMPLE_REAL, SAMPLE_ORIGINAL_FAKE, SAMPLE_SBI)
        }
        self.assertEqual(mass, mixture)


class SBIGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_and_clip_coherent(self) -> None:
        image_size = 64
        x = np.linspace(20, 230, image_size, dtype=np.uint8)
        frame = np.repeat(x[None, :, None], image_size, axis=0)
        frame = np.concatenate([frame, np.flip(frame, axis=1), frame], axis=2)
        frames = np.repeat(frame[None, :, :, :], 3, axis=0)
        face_mask = face_mask_from_aligned_landmarks(None, image_size)

        first, first_masks, first_parameters = generate_self_blended_clip(
            frames,
            face_mask,
            {"enabled": True},
            rng=np.random.default_rng(123),
        )
        second, second_masks, second_parameters = generate_self_blended_clip(
            frames,
            face_mask,
            {"enabled": True},
            rng=np.random.default_rng(123),
        )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_masks, second_masks)
        self.assertEqual(first_parameters, second_parameters)
        np.testing.assert_array_equal(first[0], first[1])
        np.testing.assert_array_equal(first[1], first[2])
        np.testing.assert_array_equal(first_masks[0], first_masks[1])
        self.assertGreater(float(first_masks.max()), 0.0)
        self.assertLessEqual(float(first_masks.max()), 1.0)
        self.assertGreater(int(np.count_nonzero(first != frames)), 0)

    def test_mask_is_non_empty_and_does_not_cover_the_image(self) -> None:
        mask = face_mask_from_aligned_landmarks(None, 64)

        self.assertGreater(float(mask.sum()), 0.0)
        self.assertLess(float(mask.sum()), float(mask.size))
        self.assertEqual(float(mask[0].sum()), 0.0)
        self.assertEqual(float(mask[-1].sum()), 0.0)


class SBIDatasetStrataTests(unittest.TestCase):
    def test_sbi_adds_explicit_real_companion_without_changing_epoch_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "train.jsonl"
            frames = [f"frame_{index:03d}.jpg" for index in range(2)]
            write_manifest(
                [
                    VideoRecord(
                        dataset="ffpp",
                        split="train",
                        video_id="real",
                        label=0,
                        method="Original",
                        source_video="real.mp4",
                        frames=frames,
                        landmark_path="real.npz",
                    ),
                    VideoRecord(
                        dataset="ffpp",
                        split="train",
                        video_id="fake",
                        label=1,
                        method="Deepfakes",
                        source_video="fake.mp4",
                        frames=frames,
                        landmark_path="fake.npz",
                    ),
                ],
                manifest,
            )
            dataset = QALFVideoDataset(
                manifest,
                temporary,
                temporary,
                num_frames=2,
                texture_frames=1,
                image_size=32,
                texture_mode="full_face",
                training=True,
                texture_augmentation={},
                sbi_config={"enabled": True},
            )
            baseline = QALFVideoDataset(
                manifest,
                temporary,
                temporary,
                num_frames=2,
                texture_frames=1,
                image_size=32,
                texture_mode="full_face",
                training=True,
                texture_augmentation={},
                sbi_config={"enabled": False},
            )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.samples_per_epoch, 2)
        self.assertEqual(dataset.labels, [0, 1, 1])
        self.assertEqual(
            dataset.sampling_strata,
            [SAMPLE_REAL, SAMPLE_ORIGINAL_FAKE, SAMPLE_SBI],
        )
        self.assertEqual(len(baseline), 2)
        self.assertEqual(baseline.labels, [0, 1])


if __name__ == "__main__":
    unittest.main()
