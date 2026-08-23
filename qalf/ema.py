"""Exponential moving average (EMA) weights for model evaluation."""

from __future__ import annotations

import torch
from torch import nn


class ModelEMA:
    """Maintain a shadow copy of model weights with an exponential average.

    Floating-point tensors are averaged after every optimizer step. Integer and
    boolean buffers (for example BatchNorm counters) are copied directly so
    that the shadow state remains loadable with the model's exact state dict.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the shadow state from the current model state."""

        keep = self.decay
        update = 1.0 - keep
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if value.is_floating_point():
                shadow.mul_(keep).add_(value.detach(), alpha=update)
            else:
                shadow.copy_(value.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Replace ``model`` weights with the EMA shadow state."""

        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached snapshot suitable for checkpointing."""

        return {name: value.detach().clone() for name, value in self.shadow.items()}
