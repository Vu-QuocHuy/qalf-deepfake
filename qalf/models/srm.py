"""Fixed and learnable high-pass residual encoders."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def _srm_kernels() -> torch.Tensor:
    """Return three zero-sum high-pass kernels adapted from SRM residual filters."""

    first = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, -1, 4, -1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 4.0
    second = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 4.0
    third = torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    ) / 12.0
    return torch.stack((first, second, third))


def _diverse_srm_kernels(count: int = 30) -> torch.Tensor:
    """Build a deterministic, diverse bank of normalized zero-DC filters."""

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
    laplacian3 = torch.zeros(5, 5)
    laplacian3[1:4, 1:4] = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    )
    filters.append(laplacian3)
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
    filters.append(
        torch.tensor(
            [[-1.0, 2.0, -1.0, 0.0, 0.0], [2.0, -4.0, 2.0, 0.0, 0.0],
             [-1.0, 2.0, -1.0, 0.0, 0.0], [0.0] * 5, [0.0] * 5]
        )
    )
    diagonal = torch.zeros(5, 5)
    diagonal[0, 0], diagonal[1, 1], diagonal[2, 2] = 1.0, -2.0, 1.0
    filters.append(diagonal)

    generator = torch.Generator()
    generator.manual_seed(42)
    base_count = len(filters)
    while len(filters) < count:
        source = filters[len(filters) % base_count]
        scale = 0.5 + (len(filters) % 5) * 0.3
        filters.append(source * scale + 0.01 * torch.randn(5, 5, generator=generator))
    bank = torch.stack(filters[:count]).float()
    bank = bank - bank.mean(dim=(-2, -1), keepdim=True)
    return bank / (bank.abs().sum(dim=(-2, -1), keepdim=True) + 1e-6)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SRMEncoder(nn.Module):
    """Encode per-frame RGB residuals and mean-pool them at video-clip level."""

    quality_dim = 3

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        kernels = _srm_kernels()[:, None].repeat(3, 1, 1, 1)
        self.residual = nn.Conv2d(3, 9, kernel_size=5, padding=2, groups=3, bias=False)
        self.residual.weight = nn.Parameter(kernels, requires_grad=False)
        self.features = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            DepthwiseSeparableBlock(24, 48),
            DepthwiseSeparableBlock(48, 96),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(96, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
        )

    def forward(
        self, texture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        if channels != 3:
            raise ValueError(f"SRM encoder expects 3 channels, received {channels}")
        images = texture.reshape(batch * frames, channels, height, width)
        images = images * self.image_std + self.image_mean
        residual = self.residual(images)
        frame_energy = residual.square().mean(dim=(1, 2, 3)).sqrt().reshape(batch, frames)
        quality = torch.stack(
            (
                frame_energy.mean(dim=1),
                frame_energy.std(dim=1, unbiased=False),
                frame_energy.amax(dim=1),
            ),
            dim=1,
        )
        output = self.features(torch.tanh(residual)).flatten(1)
        output = self.projection(output).reshape(batch, frames, -1)
        embedding = output.mean(dim=1)
        return embedding, self.classifier(embedding).squeeze(-1), quality


class ResidualBlock(nn.Module):
    """A compact residual block used by the learned forensic stream."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        )
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class ConstrainedHighPassBank(nn.Module):
    """Learn an SRM-initialized filter bank while preserving zero DC response."""

    def __init__(self) -> None:
        super().__init__()
        kernels = _srm_kernels()[:, None].repeat(3, 1, 1, 1)
        self.register_buffer("base_weight", kernels)
        self.delta = nn.Parameter(torch.zeros_like(kernels))
        self.log_gain = nn.Parameter(torch.zeros(9, 1, 1, 1))

    def effective_weight(self) -> torch.Tensor:
        centered_delta = self.delta - self.delta.mean(dim=(-2, -1), keepdim=True)
        gain = torch.exp(self.log_gain.clamp(min=-2.0, max=2.0))
        return (self.base_weight + centered_delta) * gain

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(
            images,
            self.effective_weight(),
            padding=2,
            groups=3,
        )


class LearnableSRMEncoder(nn.Module):
    """Higher-capacity residual stream with constrained learnable high-pass filters."""

    quality_dim = 6

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.residual = ConstrainedHighPassBank()
        self.features = nn.Sequential(
            nn.Conv2d(9, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            ResidualBlock(48, 64, stride=2),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
        )

    def forward(
        self, texture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        if channels != 3:
            raise ValueError(f"Learnable SRM encoder expects 3 channels, received {channels}")
        images = texture.reshape(batch * frames, channels, height, width)
        images = images * self.image_std + self.image_mean
        residual = self.residual(images)

        per_filter_energy = residual.square().mean(dim=(-2, -1)).sqrt()
        per_filter_energy = per_filter_energy.reshape(batch, frames, 3, 3).mean(dim=2)
        quality = torch.cat(
            (
                per_filter_energy.mean(dim=1),
                per_filter_energy.std(dim=1, unbiased=False),
            ),
            dim=1,
        )

        frame_features = self.features(torch.tanh(2.0 * residual)).flatten(1)
        frame_features = self.projection(frame_features).reshape(batch, frames, -1)
        embedding = frame_features.mean(dim=1)
        return embedding, self.classifier(embedding).squeeze(-1), quality


class ResidualSpatialAttention(nn.Module):
    """Very small CBAM-style spatial attention for residual maps."""

    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        summary = torch.cat(
            (inputs.mean(dim=1, keepdim=True), inputs.amax(dim=1, keepdim=True)),
            dim=1,
        )
        return inputs * self.project(summary)


class ConstrainedDiverseHighPassBank(nn.Module):
    """Learnable 5x5 bank initialized from diverse SRM filters."""

    def __init__(self, filters: int = 30) -> None:
        super().__init__()
        initial = _diverse_srm_kernels(filters)[:, None]
        self.register_buffer("base_weight", initial)
        self.delta = nn.Parameter(torch.zeros_like(initial))
        self.log_gain = nn.Parameter(torch.zeros(filters, 1, 1, 1))

    def effective_weight(self) -> torch.Tensor:
        weight = self.base_weight + self.delta
        weight = weight - weight.mean(dim=(-2, -1), keepdim=True)
        weight = weight / (weight.abs().sum(dim=(-2, -1), keepdim=True) + 1e-6)
        return weight * torch.exp(self.log_gain.clamp(-1.5, 1.5))

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(gray, self.effective_weight(), padding=2)


class LearnableSRMEncoderV2(nn.Module):
    """Lightweight residual stream using diverse SRM filters and temporal stats."""

    quality_dim = 6

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        filters: int = 30,
    ) -> None:
        super().__init__()
        self.filters = int(filters)
        self.residual = ConstrainedDiverseHighPassBank(self.filters)
        self.adapter = nn.Sequential(
            nn.Conv2d(self.filters, 3, kernel_size=1, bias=False),
            nn.GroupNorm(1, 3),
        )
        self.spatial_attention = ResidualSpatialAttention()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            DepthwiseSeparableBlock(32, 64),
            DepthwiseSeparableBlock(64, 96),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(96, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
        )

    def forward(
        self, texture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        if channels != 3:
            raise ValueError(f"Learnable SRM v2 expects 3 channels, received {channels}")
        images = texture.reshape(batch * frames, channels, height, width)
        images = images * self.image_std + self.image_mean
        gray = (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )
        residual = self.residual(gray).abs()
        energy = residual.square().mean(dim=(-2, -1)).sqrt().reshape(batch, frames, -1)
        energy_mean = energy.mean(dim=(1, 2))
        energy_std = energy.std(dim=(1, 2), unbiased=False)
        energy_max = energy.amax(dim=(1, 2))
        energy_delta = (energy[:, 1:] - energy[:, :-1]).abs()
        delta_mean = energy_delta.mean(dim=(1, 2)) if frames > 1 else energy_mean * 0.0
        delta_max = energy_delta.amax(dim=(1, 2)) if frames > 1 else energy_mean * 0.0
        quality = torch.stack(
            (energy_mean, energy_std, energy_max, delta_mean, delta_max, energy.mean(dim=2).std(dim=1, unbiased=False)),
            dim=1,
        )
        residual_rgb = torch.sigmoid(self.adapter(residual))
        residual_rgb = self.spatial_attention(residual_rgb)
        frame_features = self.features(residual_rgb).flatten(1)
        frame_features = self.projection(frame_features).reshape(batch, frames, -1)
        mean = frame_features.mean(dim=1)
        std = frame_features.std(dim=1, unbiased=False)
        delta = (
            (frame_features[:, 1:] - frame_features[:, :-1]).abs().mean(dim=1)
            if frames > 1
            else mean * 0.0
        )
        embedding = self.temporal_projection(torch.cat((mean, std, delta), dim=1))
        return embedding, self.classifier(embedding).squeeze(-1), quality
