from __future__ import annotations

import unittest

import torch
from torch import nn

from qalf.engine import EMAModel
from qalf.models.fusion import QualityAwareFusion
from qalf.models.texture import (
    DualViewFusion,
    TemporalDynamicsPool,
    VideoMixStyle,
)


class EMATests(unittest.TestCase):
    def test_update_averages_parameters_and_copies_batch_norm_buffers(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.BatchNorm1d(2))
        with torch.no_grad():
            model[0].weight.fill_(1.0)
        ema = EMAModel(model, decay=0.75)

        with torch.no_grad():
            model[0].weight.fill_(3.0)
            model[1].running_mean.fill_(4.0)
            model[1].running_var.fill_(5.0)
            model[1].num_batches_tracked.fill_(7)
        ema.update(model)

        torch.testing.assert_close(ema.shadow[0].weight, torch.full((2, 2), 1.5))
        torch.testing.assert_close(ema.shadow[1].running_mean, torch.full((2,), 4.0))
        torch.testing.assert_close(ema.shadow[1].running_var, torch.full((2,), 5.0))
        self.assertEqual(int(ema.shadow[1].num_batches_tracked), 7)

    def test_decay_must_be_in_open_interval(self) -> None:
        model = nn.Linear(1, 1)
        for decay in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                EMAModel(model, decay=decay)


class TextureGateBiasTests(unittest.TestCase):
    def test_bias_only_changes_initial_texture_gate_logit(self) -> None:
        torch.manual_seed(123)
        baseline = QualityAwareFusion(embedding_dim=8, texture_gate_bias=0.0)
        torch.manual_seed(123)
        texture_biased = QualityAwareFusion(embedding_dim=8, texture_gate_bias=1.0)

        torch.testing.assert_close(
            texture_biased.gate[-1].bias - baseline.gate[-1].bias,
            torch.tensor([0.0, 1.0]),
        )


class DualViewFusionTests(unittest.TestCase):
    def test_initial_fusion_preserves_full_face_prior(self) -> None:
        fusion = DualViewFusion(embedding_dim=8)
        view_embeddings = torch.randn(3, 12, 2, 8)

        torch.testing.assert_close(
            fusion(view_embeddings),
            0.8 * view_embeddings[:, :, 0] + 0.2 * view_embeddings[:, :, 1],
        )


class TemporalDynamicsPoolTests(unittest.TestCase):
    def test_initial_pooling_is_exact_frame_mean(self) -> None:
        pooling = TemporalDynamicsPool(embedding_dim=8, dropout=0.0)
        frame_embeddings = torch.randn(3, 12, 8)

        torch.testing.assert_close(
            pooling(frame_embeddings),
            frame_embeddings.mean(dim=1),
        )

    def test_short_sequences_are_finite(self) -> None:
        pooling = TemporalDynamicsPool(embedding_dim=8, dropout=0.0)
        for frame_count in (1, 2):
            with self.subTest(frame_count=frame_count):
                output = pooling(torch.randn(3, frame_count, 8))
                self.assertTrue(torch.isfinite(output).all())


class VideoMixStyleTests(unittest.TestCase):
    def test_evaluation_is_identity(self) -> None:
        module = VideoMixStyle(probability=1.0, alpha=0.1).eval()
        features = torch.randn(6, 4, 5, 5)
        torch.testing.assert_close(module(features, batch_size=2, frames=3), features)

    def test_training_uses_temporally_coherent_style(self) -> None:
        torch.manual_seed(7)
        module = VideoMixStyle(probability=1.0, alpha=0.1).train()
        video_a = torch.ones(3, 4, 5, 5)
        video_b = torch.full((3, 4, 5, 5), 4.0)
        output = module(torch.cat([video_a, video_b]), batch_size=2, frames=3)
        output = output.reshape(2, 3, 4, 5, 5)

        torch.testing.assert_close(output[:, 0], output[:, 1])
        torch.testing.assert_close(output[:, 1], output[:, 2])


if __name__ == "__main__":
    unittest.main()
