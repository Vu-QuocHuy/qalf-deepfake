from __future__ import annotations

import unittest

import torch
from torch import nn

from qalf.engine import aggregate_predictions, predict


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


class _CounterfactualModel(nn.Module):
    def fuse_precomputed(
        self,
        geometry_embedding: torch.Tensor,
        geometry_logit: torch.Tensor,
        texture_embedding: torch.Tensor,
        texture_logit: torch.Tensor,
        geometry_quality: torch.Tensor,
        texture_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del geometry_logit, geometry_quality, texture_quality
        logit = geometry_embedding[:, 0] + texture_logit
        weights = torch.full((logit.shape[0], 2), 0.5, dtype=logit.dtype, device=logit.device)
        return logit, weights

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        texture = batch["texture"]
        batch_size = int(texture.shape[0])
        geometry_embedding = torch.ones(batch_size, 2)
        texture_embedding = torch.zeros(batch_size, 2)
        texture_logit = torch.zeros(batch_size)
        logit, weights = self.fuse_precomputed(
            geometry_embedding,
            torch.ones(batch_size),
            texture_embedding,
            texture_logit,
            batch["geometry_quality"],
            batch["texture_quality"],
        )
        return {
            "logit": logit,
            "geometry_logit": torch.ones(batch_size),
            "texture_logit": texture_logit,
            "fusion_weights": weights,
            "geometry_embedding": geometry_embedding,
            "texture_embedding": texture_embedding,
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
        self.assertEqual(predictions["geometry_weight"], [0.25, 0.25])
        self.assertEqual(predictions["texture_weight"], [0.75, 0.75])

    def test_zero_geometry_counterfactual_is_reported_separately(self) -> None:
        batch = {
            "texture": torch.zeros(2, 1, 3, 2, 2),
            "geometry_quality": torch.ones(2, 5),
            "texture_quality": torch.ones(2, 5),
            "label": torch.tensor([0.0, 1.0]),
            "clip_index": torch.tensor([0, 0]),
            "video_id": ["real", "fake"],
            "method": ["real", "fake"],
            "dataset": ["test", "test"],
        }

        predictions = predict(
            _CounterfactualModel(),
            [batch],
            torch.device("cpu"),
            zero_geometry_counterfactual=True,
        )

        torch.testing.assert_close(
            torch.tensor(predictions["score"]), torch.full((2,), torch.sigmoid(torch.tensor(1.0)))
        )
        torch.testing.assert_close(
            torch.tensor(predictions["zero_geometry_score"]), torch.full((2,), 0.5)
        )
        aggregated = aggregate_predictions(predictions)
        self.assertEqual(aggregated["zero_geometry_score"], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
