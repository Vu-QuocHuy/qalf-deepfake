from __future__ import annotations

import unittest

import torch
from torch import nn

from qalf.engine import predict


class _FlipSensitiveModel(nn.Module):
    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        texture = batch["texture"]
        left = texture[..., 0].mean(dim=(1, 2, 3))
        right = texture[..., -1].mean(dim=(1, 2, 3))
        logits = left - right
        batch_size = int(logits.shape[0])
        return {
            "logit": logits,
            "geometry_logit": torch.zeros_like(logits),
            "texture_logit": logits,
            "fusion_weights": torch.tensor(
                [0.25, 0.75], dtype=logits.dtype, device=logits.device
            ).repeat(batch_size, 1),
        }


class TextureFlipTTATests(unittest.TestCase):
    def test_predict_averages_original_and_flipped_probabilities(self) -> None:
        texture = torch.zeros(2, 1, 1, 2, 2)
        texture[:, :, :, :, 0] = 2.0
        batch = {
            "texture": texture,
            "label": torch.tensor([0.0, 1.0]),
            "clip_index": torch.tensor([0, 0]),
            "video_id": ["real", "fake"],
            "method": ["real", "fake"],
            "dataset": ["test", "test"],
        }

        predictions = predict(
            _FlipSensitiveModel(),
            [batch],
            torch.device("cpu"),
            texture_flip_tta=True,
        )

        torch.testing.assert_close(
            torch.tensor(predictions["score"]),
            torch.full((2,), 0.5),
        )
        torch.testing.assert_close(
            torch.tensor(predictions["texture_score"]),
            torch.full((2,), 0.5),
        )
        self.assertEqual(predictions["geometry_weight"], [0.25, 0.25])
        self.assertEqual(predictions["texture_weight"], [0.75, 0.75])


if __name__ == "__main__":
    unittest.main()
