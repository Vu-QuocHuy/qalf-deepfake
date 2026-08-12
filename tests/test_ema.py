import unittest

import torch
from torch import nn

from qalf.ema import ModelEMA


class TestModelEMA(unittest.TestCase):
    def test_shadow_tracks_optimizer_updates(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        ema = ModelEMA(model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema.update(model)
        self.assertEqual(float(ema.shadow["weight"].item()), 1.0)
        ema.copy_to(model)
        self.assertEqual(float(model.weight.item()), 1.0)

    def test_decay_range(self):
        model = nn.Linear(1, 1)
        with self.assertRaises(ValueError):
            ModelEMA(model, decay=0.0)
        with self.assertRaises(ValueError):
            ModelEMA(model, decay=1.0)


if __name__ == "__main__":
    unittest.main()
