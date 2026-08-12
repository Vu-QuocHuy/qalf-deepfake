from __future__ import annotations

import unittest

import torch

from qalf.models.fusion import QualityAwareFusion


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


if __name__ == "__main__":
    unittest.main()
