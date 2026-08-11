"""Pretrained EfficientNet encoders for landmark-aligned face sequences."""

from __future__ import annotations

import torch
from torch import nn

SUPPORTED_TEXTURE_BACKBONES = {"efficientnet_b0", "efficientnet_b1"}
SUPPORTED_TEXTURE_POOLING = {"mean", "dynamics"}


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, int]:
    if name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        feature_dim = int(model.classifier[-1].in_features)
    elif name == "efficientnet_b1":
        from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = efficientnet_b1(weights=weights)
        feature_dim = int(model.classifier[-1].in_features)
    else:
        raise ValueError(f"Unsupported texture backbone: {name}")
    return model.features, model.avgpool, feature_dim


class TemporalDynamicsPool(nn.Module):
    """Pool appearance and short-term embedding dynamics into one representation.

    Mean pooling is retained as a residual path.  The learned path sees temporal
    standard deviation plus first- and second-order absolute differences.  This
    makes adjacency observable while keeping initialization identical to mean
    pooling, which is important for stable fine-tuning of an ImageNet encoder.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        statistics_dim = embedding_dim * 3
        self.projection = nn.Sequential(
            nn.LayerNorm(statistics_dim),
            nn.Linear(statistics_dim, embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        mean = frame_embeddings.mean(dim=1)
        centered = frame_embeddings - mean.unsqueeze(1)
        standard_deviation = torch.sqrt(centered.square().mean(dim=1) + 1e-6)
        if frame_embeddings.shape[1] > 1:
            first_difference = (
                frame_embeddings[:, 1:] - frame_embeddings[:, :-1]
            ).abs().mean(dim=1)
        else:
            first_difference = torch.zeros_like(mean)
        if frame_embeddings.shape[1] > 2:
            second_difference = (
                frame_embeddings[:, 2:]
                - 2.0 * frame_embeddings[:, 1:-1]
                + frame_embeddings[:, :-2]
            ).abs().mean(dim=1)
        else:
            second_difference = torch.zeros_like(mean)
        dynamics = torch.cat(
            [standard_deviation, first_difference, second_difference], dim=1
        )
        return mean + self.projection(dynamics)


class DualViewFusion(nn.Module):
    """Fuse full-face and canonical-skin embeddings for every frame.

    The initial 80/20 weighting preserves the empirically stronger full-face
    representation while allowing the gate to learn when skin-only evidence is
    useful.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([1.3862944, 0.0]))

    def forward(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        if view_embeddings.ndim != 4 or view_embeddings.shape[2] != 2:
            raise ValueError("DualViewFusion expects [batch, frames, 2, embedding]")
        gate_input = view_embeddings.flatten(start_dim=2)
        weights = torch.softmax(self.gate(gate_input), dim=2).unsqueeze(-1)
        return torch.sum(weights * view_embeddings, dim=2)


class VideoMixStyle(nn.Module):
    """Mix shallow feature statistics between videos during training only.

    One partner and one interpolation coefficient are shared across every frame
    of a video.  Unlike frame-wise MixStyle, this does not manufacture artificial
    flicker that could become a label shortcut for the temporal pooling module.
    """

    def __init__(self, probability: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("MixStyle probability must be in [0, 1]")
        if alpha <= 0.0:
            raise ValueError("MixStyle alpha must be positive")
        self.probability = float(probability)
        self.alpha = float(alpha)

    def forward(
        self,
        features: torch.Tensor,
        batch_size: int,
        frames: int,
    ) -> torch.Tensor:
        if (
            not self.training
            or self.probability == 0.0
            or batch_size < 2
            or bool(torch.rand((), device=features.device) > self.probability)
        ):
            return features
        if features.shape[0] != batch_size * frames:
            raise ValueError("VideoMixStyle received an inconsistent batch shape")

        original_dtype = features.dtype
        video_features = features.reshape(
            batch_size, frames, *features.shape[1:]
        ).float()
        mean = video_features.mean(dim=(-2, -1), keepdim=True)
        standard_deviation = torch.sqrt(
            video_features.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-6
        )
        normalized = (video_features - mean) / standard_deviation

        permutation = torch.randperm(batch_size, device=features.device)
        concentration = torch.full(
            (batch_size,), self.alpha, device=features.device, dtype=torch.float32
        )
        interpolation = torch.distributions.Beta(concentration, concentration).sample()
        interpolation = interpolation.reshape(batch_size, 1, 1, 1, 1)
        mixed_mean = interpolation * mean + (1.0 - interpolation) * mean[permutation]
        mixed_standard_deviation = (
            interpolation * standard_deviation
            + (1.0 - interpolation) * standard_deviation[permutation]
        )
        mixed = normalized * mixed_standard_deviation + mixed_mean
        return mixed.to(dtype=original_dtype).reshape_as(features)


class TextureEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
        backbone: str = "efficientnet_b0",
        temporal_pooling: str = "mean",
        mixstyle_probability: float = 0.0,
        mixstyle_alpha: float = 0.1,
        mixstyle_layers: tuple[int, ...] = (),
        texture_views: int = 1,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_TEXTURE_BACKBONES:
            raise ValueError(f"Unsupported texture backbone: {backbone}")
        if temporal_pooling not in SUPPORTED_TEXTURE_POOLING:
            raise ValueError(f"Unsupported texture temporal pooling: {temporal_pooling}")
        self.backbone_name = backbone
        self.temporal_pooling = temporal_pooling
        if texture_views not in {1, 2}:
            raise ValueError("Texture views must be one or two")
        self.texture_views = int(texture_views)
        self.features, self.pool, feature_dim = _build_backbone(backbone, pretrained)
        self.mixstyle_layers = tuple(sorted(set(int(index) for index in mixstyle_layers)))
        invalid_layers = [
            index for index in self.mixstyle_layers if not 0 <= index < len(self.features)
        ]
        if invalid_layers:
            raise ValueError(
                f"MixStyle layers outside EfficientNet features: {invalid_layers}"
            )
        self.mixstyle = (
            VideoMixStyle(mixstyle_probability, mixstyle_alpha)
            if mixstyle_probability > 0.0 and self.mixstyle_layers
            else None
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)
        self.view_fusion = DualViewFusion(embedding_dim) if texture_views == 2 else None
        self.temporal_dynamics = (
            TemporalDynamicsPool(embedding_dim, dropout)
            if temporal_pooling == "dynamics"
            else None
        )

    def forward(self, texture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = texture.shape
        expected_channels = self.texture_views * 3
        if channels != expected_channels:
            raise ValueError(
                f"Texture encoder expects {expected_channels} channels, received {channels}"
            )
        output = texture.reshape(
            batch, frames, self.texture_views, 3, height, width
        ).reshape(batch * frames * self.texture_views, 3, height, width)
        for index, layer in enumerate(self.features):
            output = layer(output)
            if self.mixstyle is not None and index in self.mixstyle_layers:
                output = self.mixstyle(output, batch, frames * self.texture_views)
        output = self.pool(output).flatten(1)
        output = self.projection(output).reshape(
            batch, frames, self.texture_views, -1
        )
        if self.view_fusion is not None:
            output = self.view_fusion(output)
        else:
            output = output.squeeze(2)
        if self.temporal_dynamics is not None:
            embedding = self.temporal_dynamics(output)
        else:
            embedding = output.mean(dim=1)
        return embedding, self.classifier(embedding).squeeze(-1)
