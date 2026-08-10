from __future__ import annotations

import unittest

import torch
from torch import nn

from qalf.engine import EMAModel
from qalf.models.srm import SRMChannelAdapter, SRMFilterLayer, _build_srm_kernels


class SRMTests(unittest.TestCase):
    def test_grouped_filter_layout_applies_every_kernel_to_every_channel(self) -> None:
        layer = SRMFilterLayer()
        expected = torch.from_numpy(_build_srm_kernels())
        actual = layer.weight[:, 0].reshape(3, 3, 5, 5)
        for channel in range(3):
            torch.testing.assert_close(actual[channel], expected)

    def test_adapter_is_identity_at_initialization(self) -> None:
        adapter = SRMChannelAdapter().eval()
        inputs = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            outputs = adapter(inputs)
        torch.testing.assert_close(outputs, inputs)


class EMATests(unittest.TestCase):
    def test_update_averages_floating_state_and_copies_integer_buffers(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.BatchNorm1d(2))
        with torch.no_grad():
            model[0].weight.fill_(1.0)
        ema = EMAModel(model, decay=0.75)

        with torch.no_grad():
            model[0].weight.fill_(3.0)
            model[1].running_mean.fill_(4.0)
            model[1].num_batches_tracked.fill_(7)
        ema.update(model)

        torch.testing.assert_close(ema.shadow[0].weight, torch.full((2, 2), 1.5))
        torch.testing.assert_close(ema.shadow[1].running_mean, torch.ones(2))
        self.assertEqual(int(ema.shadow[1].num_batches_tracked), 7)

    def test_decay_must_be_valid(self) -> None:
        model = nn.Linear(1, 1)
        for decay in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                EMAModel(model, decay=decay)


if __name__ == "__main__":
    unittest.main()
