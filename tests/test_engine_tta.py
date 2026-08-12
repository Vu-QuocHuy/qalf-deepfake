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
        return {"logit": logits}


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


if __name__ == "__main__":
    unittest.main()
