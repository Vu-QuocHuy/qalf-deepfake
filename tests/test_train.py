import unittest
from unittest.mock import patch

import torch

from scripts.train import _make_grad_scaler


class GradScalerCompatibilityTest(unittest.TestCase):
    def test_uses_current_amp_api_when_available(self) -> None:
        sentinel = object()
        with patch.object(torch.amp, "GradScaler", create=True, return_value=sentinel) as factory:
            scaler = _make_grad_scaler(True)
        self.assertIs(scaler, sentinel)
        factory.assert_called_once_with("cuda", enabled=True)

    def test_falls_back_for_pytorch_22(self) -> None:
        with patch.object(torch.amp, "GradScaler", None, create=True):
            scaler = _make_grad_scaler(False)
        self.assertIsInstance(scaler, torch.cuda.amp.GradScaler)
        self.assertFalse(scaler.is_enabled())


if __name__ == "__main__":
    unittest.main()
