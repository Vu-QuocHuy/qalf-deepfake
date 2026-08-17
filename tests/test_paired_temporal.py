from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from qalf.data.dataset import (
    DEFAULT_TEXTURE_AUGMENTATION,
    _augment,
    _sample_augmentation_plan,
    _texture_positions,
)
from qalf.models.texture import PairedTemporalResidualPool


class PairedTemporalSamplingTests(unittest.TestCase):
    def test_eight_frames_form_four_adjacent_pairs_across_the_window(self) -> None:
        positions = _texture_positions(32, 8, "paired")

        np.testing.assert_array_equal(
            positions,
            np.asarray([0, 1, 10, 11, 20, 21, 30, 31]),
        )

    def test_uniform_sampling_retains_the_baseline_positions(self) -> None:
        positions = _texture_positions(32, 8, "uniform")

        np.testing.assert_array_equal(
            positions,
            np.rint(np.linspace(0, 31, 8)).astype(np.int64),
        )

    def test_paired_sampling_rejects_odd_frame_counts(self) -> None:
        with self.assertRaises(ValueError):
            _texture_positions(32, 7, "paired")


class CoherentAugmentationTests(unittest.TestCase):
    def test_one_plan_produces_identical_transforms_for_identical_frames(self) -> None:
        settings = dict(DEFAULT_TEXTURE_AUGMENTATION)
        settings.update(
            {
                "flip_probability": 1.0,
                "brightness_contrast_probability": 1.0,
                "gamma_probability": 1.0,
                "downsample_probability": 1.0,
                "blur_probability": 1.0,
                "noise_probability": 1.0,
                "jpeg_probability": 1.0,
            }
        )
        image = (
            np.arange(64 * 64 * 3, dtype=np.int32).reshape(64, 64, 3) % 256
        ).astype(np.uint8)
        random.seed(123)
        plan = _sample_augmentation_plan(settings)

        first = _augment(image, settings, plan)
        second = _augment(image, settings, plan)

        np.testing.assert_array_equal(first, second)


class PairedTemporalResidualPoolTests(unittest.TestCase):
    def test_zero_initialized_pool_starts_exactly_at_mean(self) -> None:
        torch.manual_seed(123)
        pooling = PairedTemporalResidualPool(embedding_dim=192).eval()
        sequence = torch.randn(3, 8, 192)

        output = pooling(sequence)

        torch.testing.assert_close(output, sequence.mean(dim=1), rtol=0.0, atol=0.0)

    def test_residual_is_bounded_and_head_is_small(self) -> None:
        pooling = PairedTemporalResidualPool(
            embedding_dim=192,
            bottleneck_dim=32,
            residual_scale=0.1,
        ).eval()
        with torch.no_grad():
            pooling.output_projection.bias.fill_(100.0)
        sequence = torch.randn(2, 8, 192)

        correction = pooling(sequence) - sequence.mean(dim=1)
        parameters = sum(parameter.numel() for parameter in pooling.parameters())

        self.assertLessEqual(float(correction.abs().max()), 0.100001)
        self.assertLess(parameters, 25_000)


if __name__ == "__main__":
    unittest.main()
