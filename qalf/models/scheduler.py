"""Interpretable rule-based texture refresh policy for streaming experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TextureRefreshPolicy:
    max_cache_age: int = 4
    geometry_uncertainty_threshold: float = 0.65
    min_landmark_detection_ratio: float = 0.80

    def should_refresh(
        self,
        geometry_logit: torch.Tensor,
        geometry_quality: torch.Tensor,
        cache_age: int,
    ) -> torch.Tensor:
        probability = torch.sigmoid(geometry_logit)
        uncertainty = 1.0 - torch.abs(probability - 0.5) * 2.0
        low_landmark_quality = geometry_quality[:, 0] < self.min_landmark_detection_ratio
        stale = torch.full_like(low_landmark_quality, cache_age >= self.max_cache_age)
        return (uncertainty > self.geometry_uncertainty_threshold) | low_landmark_quality | stale
