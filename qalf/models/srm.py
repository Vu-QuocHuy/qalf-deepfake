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
