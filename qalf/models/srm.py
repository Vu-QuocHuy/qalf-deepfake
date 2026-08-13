"""Small fixed SRM-inspired residual preprocessor.

The preprocessor is deliberately not a second modality.  It applies three
fixed zero-sum 5x5 high-pass kernels to a luminance-like projection and adds
the bounded residual back to the RGB tensor before EfficientNet-B0.  This
keeps the ImageNet RGB input interface and avoids a second encoder or fusion
gate that could collapse to the stronger RGB branch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _kernel_bank() -> torch.Tensor:
    # One isotropic SRM-style residual and two directional second-derivative
    # filters.  All filters have zero DC response; denominators keep the
    # injected residual in a comparable range to normalized RGB.
    isotropic = torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    ) / 12.0
    horizontal = torch.zeros((5, 5), dtype=torch.float32)
    horizontal[2] = torch.tensor([-1, 2, -2, 2, -1], dtype=torch.float32) / 8.0
    vertical = horizontal.t().contiguous()
    return torch.stack((isotropic, horizontal, vertical), dim=0).unsqueeze(1)


class FixedSRMPreprocess(nn.Module):
    """Inject three bounded fixed high-pass residual maps into RGB input."""

    def __init__(self, initial_scale: float = 0.05) -> None:
        super().__init__()
        self.register_buffer("kernel_bank", _kernel_bank())
        # A single learnable scale lets training decide whether the residual
        # is useful while retaining a safe, near-RGB initialization.
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"SRM preprocessing expects [N, 3, H, W], received {tuple(rgb.shape)}"
            )
        # Channel averaging is sufficient here: the filters are zero-sum, so
        # ImageNet normalization offsets do not create a DC residual.
        luminance = rgb.mean(dim=1, keepdim=True)
        residual = F.conv2d(luminance, self.kernel_bank, padding=2)
        residual = torch.tanh(residual)
        # Three responses are intentionally mapped back to the three RGB
        # channels, preserving EfficientNet's pretrained input contract.
        # Keep the learned perturbation small enough that ImageNet-normalized
        # RGB remains in a familiar range even if optimization is noisy.
        scale = self.residual_scale.clamp(-0.25, 0.25)
        return rgb + scale * residual


__all__ = ["FixedSRMPreprocess"]
