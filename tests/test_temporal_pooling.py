import unittest
from unittest.mock import patch

import torch
from torch import nn

from qalf.models.texture import TextureEncoder


class TestTemporalAttentionInitialization(unittest.TestCase):
    def test_attention_starts_as_uniform_mean_pool(self):
        backbone = (nn.Identity(), nn.AdaptiveAvgPool2d(1), 3)
        with patch("qalf.models.texture._build_backbone", return_value=backbone):
            encoder = TextureEncoder(
                embedding_dim=3,
                dropout=0.0,
                pretrained=False,
                temporal_pooling="attention",
            )

        self.assertTrue(
            torch.equal(
                encoder.temporal_attention.weight,
                torch.zeros_like(encoder.temporal_attention.weight),
            )
        )
        self.assertTrue(
            torch.equal(
                encoder.temporal_attention.bias,
                torch.zeros_like(encoder.temporal_attention.bias),
            )
        )

        frame_embeddings = torch.randn(2, 8, 3)
        logits = encoder.temporal_attention(frame_embeddings).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        expected = torch.full_like(weights, 1.0 / frame_embeddings.shape[1])
        self.assertTrue(torch.allclose(weights, expected))

    def test_shared_head_initialization_matches_mean_control(self):
        backbone = (nn.Identity(), nn.AdaptiveAvgPool2d(1), 3)
        with patch("qalf.models.texture._build_backbone", return_value=backbone):
            torch.manual_seed(123)
            mean_encoder = TextureEncoder(
                embedding_dim=3,
                dropout=0.0,
                pretrained=False,
                temporal_pooling="mean",
            )
            torch.manual_seed(123)
            attention_encoder = TextureEncoder(
                embedding_dim=3,
                dropout=0.0,
                pretrained=False,
                temporal_pooling="attention",
            )
        for name, value in mean_encoder.state_dict().items():
            if name in attention_encoder.state_dict():
                self.assertTrue(torch.equal(value, attention_encoder.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
