"""Fixed SRM high-pass filter preprocessing for frequency-aware deepfake detection.

Three classical Spatial Rich Model kernels extract high-frequency residuals
that reveal compression artefacts and blending boundaries invisible to
standard ImageNet features.  The filters are **non-learnable**; only a thin
1×1 adapter projects the concatenated (RGB + SRM) channels back to 3 so
the pretrained EfficientNet-B0 backbone can consume them unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


def _srm_kernels() -> torch.Tensor:
    """Return three 5×5 SRM filter kernels stacked as ``(3, 1, 5, 5)``."""

    # Type-1: edge residual (centre = −4, four neighbours = +1)
    k1 = torch.zeros(5, 5)
    k1[1, 2] = 1.0
    k1[2, 1] = 1.0
    k1[2, 3] = 1.0
    k1[3, 2] = 1.0
    k1[2, 2] = -4.0

    # Type-2: second-order edge residual
    k2 = torch.zeros(5, 5)
    k2[0, 2] = -1.0
    k2[1, 2] = 2.0
    k2[2, 2] = -2.0
    k2[3, 2] = 2.0
    k2[4, 2] = -1.0

    # Type-3: square 3×3 high-pass
    k3 = torch.zeros(5, 5)
    k3[1, 1] = -1.0
    k3[1, 2] = 2.0
    k3[1, 3] = -1.0
    k3[2, 1] = 2.0
    k3[2, 2] = -4.0
    k3[2, 3] = 2.0
    k3[3, 1] = -1.0
    k3[3, 2] = 2.0
    k3[3, 3] = -1.0

    return torch.stack([k1, k2, k3]).unsqueeze(1)  # (3, 1, 5, 5)


class FrequencyPreprocess(nn.Module):
    """Concatenate RGB with fixed SRM residuals, then project back to 3 ch.

    The SRM convolution uses ``groups=3`` so each colour channel is filtered
    independently by each kernel, producing ``3 kernels × 3 channels = 9``
    feature maps.  A learnable ``1×1`` convolution then projects the combined
    ``3 + 9 = 12`` channels down to 3 so the downstream backbone receives a
    tensor with the original channel count.
    """

    def __init__(self) -> None:
        super().__init__()
        kernels = _srm_kernels()  # (3, 1, 5, 5)
        # Repeat for 3 input channels → groups=3 depthwise-style filtering.
        # Shape: (9, 1, 5, 5) with groups=3 → each of 3 channels filtered by
        # 3 kernels independently.
        self.register_buffer(
            "srm_weight",
            kernels.repeat(3, 1, 1, 1),  # (9, 1, 5, 5)
        )
        # Adapter: 12 input channels (3 RGB + 9 SRM) → 3 output channels.
        self.adapter = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        srm = torch.nn.functional.conv2d(
            x,
            self.srm_weight,
            padding=2,
            groups=3,
        )
        combined = torch.cat([x, srm], dim=1)  # (B, 12, H, W)
        return self.adapter(combined)
