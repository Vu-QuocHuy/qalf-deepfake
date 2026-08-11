from __future__ import annotations

import unittest

import torch
from torch import nn

from qalf.engine import EMAModel
from qalf.models.fusion import QualityAwareFusion


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


if __name__ == "__main__":
    unittest.main()
