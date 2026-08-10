"""Lightweight RGB, forensic-residual, and temporal texture encoders."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_TEXTURE_BACKBONES = {
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
}
TEXTURE_TEMPORAL_MODES = {"mean", "difference"}

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _initial_srm_kernels(count: int) -> torch.Tensor:
    """Build deterministic zero-sum high-pass kernels without a second backbone."""

    filters: list[torch.Tensor] = []
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        kernel = torch.zeros(5, 5)
        kernel[2, 2] = -1.0
        kernel[2 + dx, 2 + dy] = 1.0
        filters.append(kernel)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        kernel = torch.zeros(5, 5)
        kernel[2 - dx, 2 - dy] = 1.0
        kernel[2, 2] = -2.0
        kernel[2 + dx, 2 + dy] = 1.0
        filters.append(kernel)
    laplacian = torch.zeros(5, 5)
    laplacian[1:4, 1:4] = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    )
    filters.append(laplacian)
    filters.append(
        torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -2.0, 0.0, 0.0],
                [-1.0, -2.0, 16.0, -2.0, -1.0],
                [0.0, 0.0, -2.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
            ]
        )
    )
    base = tuple(filters)
    while len(filters) < count:
        source = base[len(filters) % len(base)]
        filters.append(torch.rot90(source, len(filters) % 4, dims=(0, 1)).clone())
    kernels = torch.stack(filters[:count]).unsqueeze(1)
    kernels -= kernels.mean(dim=(-2, -1), keepdim=True)
    return kernels / kernels.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


class LightweightSRMExpert(nn.Module):
    """Small learnable high-pass expert operating on ImageNet-normalized RGB frames."""

    def __init__(self, embedding_dim: int, filters: int = 12, channels: int = 48) -> None:
        super().__init__()
        if filters < 1 or channels < 8:
            raise ValueError("SRM filters must be positive and channels must be >= 8")
        self.filters = int(filters)
        self.highpass = nn.Parameter(_initial_srm_kernels(self.filters))
        hidden = max(16, channels // 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.filters, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.register_buffer(
            "image_mean", torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def _constrained_kernels(self) -> torch.Tensor:
        kernels = self.highpass - self.highpass.mean(dim=(-2, -1), keepdim=True)
        return kernels / kernels.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _skin_interior_mask(texture: torch.Tensor) -> torch.Tensor:
        """Exclude artificial canonical-mask borders from forensic residuals."""

        height, width = texture.shape[-2:]
        y = (torch.arange(height, device=texture.device) + 0.5) / float(height)
        x = (torch.arange(width, device=texture.device) + 0.5) / float(width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        mask = torch.zeros((height, width), dtype=torch.bool, device=texture.device)
        for x1, y1, x2, y2 in (
            (0.23, 0.08, 0.77, 0.33),
            (0.08, 0.43, 0.43, 0.72),
            (0.57, 0.43, 0.92, 0.72),
            (0.42, 0.35, 0.58, 0.67),
        ):
            mask |= (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
        mask_float = mask.to(dtype=texture.dtype).view(1, 1, height, width)
        return 1.0 - F.max_pool2d(1.0 - mask_float, kernel_size=5, stride=1, padding=2)

    def forward(self, texture: torch.Tensor) -> torch.Tensor:
        rgb = (texture * self.image_std + self.image_mean).clamp(0.0, 1.0)
        gray = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        residual = torch.abs(F.conv2d(gray, self._constrained_kernels(), padding=2))
        residual = residual * self._skin_interior_mask(texture)
        return self.projection(self.encoder(residual))


class TemporalDifferenceAdapter(nn.Module):
    """Order-aware residual temporal model with fewer than 0.1M parameters at D=192."""

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.diff_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.diff_gate = nn.Linear(embedding_dim * 2, 1)
        self.depthwise = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size=3,
            padding=1,
            groups=embedding_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.frame_score = nn.Linear(embedding_dim, 1)
        nn.init.zeros_(self.diff_projection.weight)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.frame_score.weight)
        nn.init.zeros_(self.frame_score.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        difference = torch.zeros_like(frames)
        difference[:, 1:] = frames[:, 1:] - frames[:, :-1]
        projected = F.silu(self.diff_projection(difference))
        gate = torch.sigmoid(self.diff_gate(torch.cat([frames, projected], dim=-1)))
        output = frames + gate * projected
        temporal = self.pointwise(self.depthwise(output.transpose(1, 2))).transpose(1, 2)
        output = output + self.dropout(F.silu(temporal))
        weights = torch.softmax(self.frame_score(output), dim=1)
        return torch.sum(weights * output, dim=1)


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    if name == "mobilenet_v3_small":
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        feature_dim = int(model.classifier[0].in_features)
    elif name == "mobilenet_v3_large":
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        feature_dim = int(model.classifier[0].in_features)
    elif name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        feature_dim = int(model.classifier[-1].in_features)
    else:
        raise ValueError(f"Unsupported texture backbone: {name}")
    return model.features, model.avgpool, feature_dim


class TextureEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
        backbone: str = "mobilenet_v3_small",
        temporal_mode: str = "mean",
        srm_enabled: bool = False,
        srm_filters: int = 12,
        srm_channels: int = 48,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_TEXTURE_BACKBONES:
            raise ValueError(f"Unsupported texture backbone: {backbone}")
        self.backbone_name = backbone
        if temporal_mode not in TEXTURE_TEMPORAL_MODES:
            raise ValueError(f"Unsupported texture temporal mode: {temporal_mode}")
        self.temporal_mode = temporal_mode
        self.srm_enabled = bool(srm_enabled)
        self.features, self.pool, feature_dim = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.srm_expert = (
            LightweightSRMExpert(embedding_dim, srm_filters, srm_channels)
            if self.srm_enabled
            else None
        )
        self.srm_gate = (
            nn.Sequential(
                nn.Linear(embedding_dim * 2, max(16, embedding_dim // 4)),
                nn.SiLU(),
                nn.Linear(max(16, embedding_dim // 4), 1),
                nn.Sigmoid(),
            )
            if self.srm_enabled
            else None
        )
        if self.srm_gate is not None:
            nn.init.zeros_(self.srm_gate[-2].weight)
            nn.init.constant_(self.srm_gate[-2].bias, -2.0)
        self.temporal = (
            TemporalDifferenceAdapter(embedding_dim, dropout)
            if temporal_mode == "difference"
            else None
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.last_srm_weight: torch.Tensor | None = None

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        flattened = texture.reshape(batch * frames, channels, height, width)
        output = self.pool(self.features(flattened)).flatten(1)
        output = self.projection(output)
        if self.srm_expert is not None:
            assert self.srm_gate is not None
            srm = self.srm_expert(flattened)
            gate = self.srm_gate(torch.cat([output, srm], dim=1))
            output = output + gate * srm
            self.last_srm_weight = gate.reshape(batch, frames).mean(dim=1)
        else:
            self.last_srm_weight = output.new_zeros(batch)
        output = output.reshape(batch, frames, -1)
        embedding = self.temporal(output) if self.temporal is not None else output.mean(dim=1)
        return embedding, self.classifier(embedding).squeeze(-1)
