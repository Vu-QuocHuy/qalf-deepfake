"""Fixed SRM (Spatial Rich Model) noise-residual filters for cross-dataset robustness.

SRM filters extract high-frequency noise patterns that reveal manipulation traces
independent of compression codec or quality settings.  The filters are non-learnable
so they do not overfit to training-domain compression artifacts.

References:
    - Zhou et al., "Two-Stream Neural Networks for Tampered Face Detection", CVPRW 2017
    - Masi et al., "Two-Branch Recurrent Network for Isolating Deepfakes", ECCV 2020
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _build_srm_kernels() -> np.ndarray:
    """Return three classic 5×5 SRM kernels (edge-1, edge-2, square-3)."""

    kernels = np.zeros((3, 5, 5), dtype=np.float32)

    # 1st-order horizontal edge
    kernels[0, 2, 1] = 1.0
    kernels[0, 2, 2] = -1.0

    # 2nd-order Laplacian
    kernels[1, 1, 2] = 1.0
    kernels[1, 2, 1] = 1.0
    kernels[1, 2, 2] = -4.0
    kernels[1, 2, 3] = 1.0
    kernels[1, 3, 2] = 1.0

    # 3rd-order SQUARE kernel
    kernels[2, 1, 1] = -1.0
    kernels[2, 1, 2] = 2.0
    kernels[2, 1, 3] = -1.0
    kernels[2, 2, 1] = 2.0
    kernels[2, 2, 2] = -4.0
    kernels[2, 2, 3] = 2.0
    kernels[2, 3, 1] = -1.0
    kernels[2, 3, 2] = 2.0
    kernels[2, 3, 3] = -1.0

    return kernels


class SRMFilterLayer(nn.Module):
    """Non-learnable SRM convolution that produces 9-channel noise residuals.

    Input: (B, 3, H, W) RGB image
    Output: (B, 9, H, W) noise residuals (3 filters × 3 channels, groups=3)
    """

    def __init__(self) -> None:
        super().__init__()
        srm = _build_srm_kernels()  # (3, 5, 5)
        # Repeat each kernel for 3 input channels using groups=3 depthwise conv
        # Weight shape for groups=3: (out_channels=9, in_channels/groups=1, 5, 5)
        weight = np.zeros((9, 1, 5, 5), dtype=np.float32)
        for channel_index in range(3):
            for kernel_index in range(3):
                # Grouped convolution assigns three consecutive outputs to each
                # input channel, so the channel index must be the outer index.
                weight[channel_index * 3 + kernel_index, 0] = srm[kernel_index]
        self.register_buffer("weight", torch.from_numpy(weight))
        self.bn = nn.BatchNorm2d(9)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residuals = F.conv2d(inputs, self.weight, padding=2, groups=3)
        return self.bn(residuals)


class SRMChannelAdapter(nn.Module):
    """Add a learnable SRM correction while preserving pretrained RGB at initialization."""

    def __init__(self) -> None:
        super().__init__()
        self.srm = SRMFilterLayer()
        self.adapter = nn.Conv2d(9, 3, kernel_size=1, bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residuals = self.srm(inputs)
        return inputs + self.adapter(residuals)
