"""Training-only exponential moving average for stable lightweight checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            target = self.shadow[name]
            if target.is_floating_point():
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(value.detach())
        self.updates += 1

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": self.model_state_dict(),
        }

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        backup = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)
