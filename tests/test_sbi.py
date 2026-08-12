from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
from qalf.engine import qalf_loss


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


class SBILossMaskTests(unittest.TestCase):
    def test_branch_warmup_disables_fused_gradient(self) -> None:
        fused_logit = torch.tensor([0.2, -0.1], requires_grad=True)
        auxiliary_logit = torch.tensor([0.3, -0.2], requires_grad=True)
        texture_logit = torch.tensor([0.4, -0.3], requires_grad=True)
        loss, _ = qalf_loss(
            {
                "logit": fused_logit,
                "auxiliary_logit": auxiliary_logit,
                "texture_logit": texture_logit,
            },
            torch.tensor([0.0, 1.0]),
            nn.BCEWithLogitsLoss(),
            auxiliary_weight=0.5,
            texture_weight=0.5,
            fused_weight=0.0,
        )
        loss.backward()

        torch.testing.assert_close(fused_logit.grad, torch.zeros_like(fused_logit))
        self.assertGreater(float(auxiliary_logit.grad.abs().sum()), 0.0)
        self.assertGreater(float(texture_logit.grad.abs().sum()), 0.0)

    def test_all_one_mask_matches_legacy_loss(self) -> None:
        outputs = {
            "logit": torch.tensor([-0.5, 0.4]),
            "texture_logit": torch.tensor([0.2, -0.3]),
            "auxiliary_logit": torch.tensor([0.7, -0.9]),
        }
        labels = torch.tensor([0.0, 1.0])
        legacy, legacy_parts = qalf_loss(
            outputs,
            labels,
            nn.BCEWithLogitsLoss(),
            auxiliary_weight=0.25,
            texture_weight=0.25,
        )
        masked, masked_parts = qalf_loss(
            outputs,
            labels,
            nn.BCEWithLogitsLoss(),
            auxiliary_weight=0.25,
            texture_weight=0.25,
            auxiliary_loss_mask=torch.ones(2),
        )

        torch.testing.assert_close(masked, legacy)
        self.assertEqual(masked_parts, legacy_parts)

    def test_sbi_sample_is_excluded_from_geometry_auxiliary_loss(self) -> None:
        outputs = {
            "logit": torch.zeros(2),
            "texture_logit": torch.zeros(2),
            "auxiliary_logit": torch.tensor([0.0, -100.0], requires_grad=True),
        }
        labels = torch.tensor([0.0, 1.0])
        loss, parts = qalf_loss(
            outputs,
            labels,
            nn.BCEWithLogitsLoss(),
            auxiliary_weight=0.25,
            texture_weight=0.25,
            auxiliary_loss_mask=torch.tensor([1.0, 0.0]),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(parts["auxiliary"], float(np.log(2.0)), places=5)

    def test_all_masked_auxiliary_loss_is_differentiable_zero(self) -> None:
        auxiliary_logit = torch.tensor([2.0, -3.0], requires_grad=True)
        outputs = {
            "logit": torch.zeros(2, requires_grad=True),
            "texture_logit": torch.zeros(2, requires_grad=True),
            "auxiliary_logit": auxiliary_logit,
        }
        loss, parts = qalf_loss(
            outputs,
            torch.tensor([0.0, 1.0]),
            nn.BCEWithLogitsLoss(),
            auxiliary_weight=0.25,
            texture_weight=0.25,
            auxiliary_loss_mask=torch.zeros(2),
        )
        loss.backward()

        self.assertEqual(parts["auxiliary"], 0.0)
        torch.testing.assert_close(auxiliary_logit.grad, torch.zeros_like(auxiliary_logit))


if __name__ == "__main__":
    unittest.main()
