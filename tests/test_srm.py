from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from qalf.models.fusion import ResidualInteractionFusion
from qalf.models.qalf import QALFModel
from qalf.models.srm import (
    ConstrainedHighPassBank,
    LearnableSRMEncoder,
    SRMEncoder,
    _srm_kernels,
)


class _TinyTextureEncoder(nn.Module):
    def __init__(self, embedding_dim: int, *_args, **_kwargs) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = texture.shape[0]
        embedding = texture.new_zeros(batch, self.embedding_dim)
        return embedding, texture.new_zeros(batch)


class _SeededTinyTextureEncoder(nn.Module):
    def __init__(self, embedding_dim: int, *_args, **_kwargs) -> None:
        super().__init__()
        self.projection = nn.Linear(3, embedding_dim)

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = texture.mean(dim=(1, 3, 4))
        embedding = self.projection(pooled)
        return embedding, embedding.mean(dim=1)


class SRMEncoderTests(unittest.TestCase):
    def test_fixed_filters_are_zero_sum(self) -> None:
        kernels = _srm_kernels()
        torch.testing.assert_close(kernels.sum(dim=(1, 2)), torch.zeros(3), atol=1e-6, rtol=0)

    def test_encoder_outputs_clip_embedding_logit_and_quality(self) -> None:
        encoder = SRMEncoder(embedding_dim=16, dropout=0.0)
        texture = torch.randn(2, 4, 3, 32, 32)
        embedding, logit, quality = encoder(texture)
        self.assertEqual(embedding.shape, (2, 16))
        self.assertEqual(logit.shape, (2,))
        self.assertEqual(quality.shape, (2, SRMEncoder.quality_dim))
        self.assertFalse(encoder.residual.weight.requires_grad)

    def test_qalf_srm_mode_uses_residual_as_auxiliary_branch(self) -> None:
        with patch("qalf.models.qalf.TextureEncoder", _TinyTextureEncoder):
            model = QALFModel(
                geometry_input_dim=12,
                embedding_dim=16,
                dropout=0.0,
                texture_pretrained=False,
                auxiliary_branch="srm",
            )
        outputs = model(
            {
                "texture": torch.randn(2, 4, 3, 32, 32),
                "texture_quality": torch.ones(2, 5),
            }
        )
        self.assertEqual(outputs["auxiliary_name"], "srm")
        self.assertEqual(outputs["logit"].shape, (2,))
        self.assertEqual(outputs["auxiliary_logit"].shape, (2,))
        self.assertEqual(outputs["fusion_weights"].shape, (2, 2))
        self.assertIsNone(model.geometry_encoder)

    def test_learned_filters_remain_zero_sum_and_receive_gradient(self) -> None:
        filters = ConstrainedHighPassBank()
        images = torch.randn(2, 3, 16, 16)
        output = filters(images)
        output.square().mean().backward()

        effective = filters.effective_weight()
        torch.testing.assert_close(
            effective.sum(dim=(-2, -1)),
            torch.zeros(9),
            atol=1e-5,
            rtol=0,
        )
        self.assertIsNotNone(filters.delta.grad)
        self.assertGreater(float(filters.delta.grad.abs().sum()), 0.0)

    def test_learnable_encoder_outputs_expected_shapes(self) -> None:
        encoder = LearnableSRMEncoder(embedding_dim=16, dropout=0.0)
        embedding, logit, quality = encoder(torch.randn(2, 3, 3, 32, 32))
        self.assertEqual(embedding.shape, (2, 16))
        self.assertEqual(logit.shape, (2,))
        self.assertEqual(quality.shape, (2, LearnableSRMEncoder.quality_dim))

    def test_residual_interaction_starts_from_texture_decision(self) -> None:
        fusion = ResidualInteractionFusion(
            embedding_dim=16,
            auxiliary_quality_dim=6,
            texture_quality_dim=5,
            dropout=0.0,
        )
        texture_logit = torch.tensor([0.2, -0.4])
        fused, routing = fusion(
            torch.randn(2, 16),
            torch.randn(2, 16),
            torch.randn(2),
            texture_logit,
            torch.randn(2, 6),
            torch.randn(2, 5),
        )
        torch.testing.assert_close(fused, texture_logit)
        self.assertEqual(routing.shape, (2, 2))

    def test_component_seed_makes_texture_initialization_branch_invariant(self) -> None:
        with patch("qalf.models.qalf.TextureEncoder", _SeededTinyTextureEncoder):
            geometry_model = QALFModel(
                geometry_input_dim=12,
                embedding_dim=16,
                texture_pretrained=False,
                auxiliary_branch="geometry",
                component_initialization_seed=73,
            )
            learned_srm_model = QALFModel(
                geometry_input_dim=12,
                embedding_dim=16,
                texture_pretrained=False,
                auxiliary_branch="learned_srm",
                fusion_architecture="residual_interaction",
                component_initialization_seed=73,
            )
        for left, right in zip(
            geometry_model.texture_encoder.parameters(),
            learned_srm_model.texture_encoder.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
