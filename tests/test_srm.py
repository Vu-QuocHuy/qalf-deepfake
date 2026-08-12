from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from qalf.models.qalf import QALFModel
from qalf.models.srm import SRMEncoder, _srm_kernels


class _TinyTextureEncoder(nn.Module):
    def __init__(self, embedding_dim: int, *_args, **_kwargs) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = texture.shape[0]
        embedding = texture.new_zeros(batch, self.embedding_dim)
        return embedding, texture.new_zeros(batch)


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


if __name__ == "__main__":
    unittest.main()
