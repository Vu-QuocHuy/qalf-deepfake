"""Training and inference loops."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Iterable, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


class EMAModel:
    """Exponential Moving Average of model parameters for stable generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        shadow_state = self.shadow.state_dict()
        model_state = model.state_dict()
        if shadow_state.keys() != model_state.keys():
            raise ValueError("EMA model structure differs from the training model")
        for name, shadow_value in shadow_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(model_value)

    def state_dict(self) -> dict[str, object]:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.shadow.load_state_dict(state_dict)


class GradScalerProtocol(Protocol):
    """Common interface of the legacy and current PyTorch gradient scalers."""

    def is_enabled(self) -> bool: ...

    def get_scale(self) -> float: ...

    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...

    def step(self, optimizer: torch.optim.Optimizer) -> object: ...

    def update(self) -> None: ...


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def qalf_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    geometry_weight: float,
    texture_weight: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if label_smoothing > 0:
        smooth = labels * (1.0 - label_smoothing) + label_smoothing * 0.5
    else:
        smooth = labels
    fused = criterion(outputs["logit"], smooth)
    geometry = criterion(outputs["geometry_logit"], smooth)
    texture = criterion(outputs["texture_logit"], smooth)
    total = fused + geometry_weight * geometry + texture_weight * texture
    return total, {
        "fused": float(fused.detach()),
        "geometry": float(geometry.detach()),
        "texture": float(texture.detach()),
    }


def flip_consistency_loss(
    outputs: dict[str, torch.Tensor],
    flipped_outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Penalize orientation-sensitive fused and texture logits."""

    fused = F.smooth_l1_loss(outputs["logit"], flipped_outputs["logit"])
    texture = F.smooth_l1_loss(
        outputs["texture_logit"],
        flipped_outputs["texture_logit"],
    )
    return 0.5 * (fused + texture)


def train_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScalerProtocol,
    geometry_weight: float,
    texture_weight: float,
    label_smoothing: float = 0.0,
    ema: EMAModel | None = None,
    flip_consistency_weight: float = 0.0,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    samples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(batch)
            loss, parts = qalf_loss(
                outputs, labels, criterion, geometry_weight, texture_weight, label_smoothing
            )
            if flip_consistency_weight > 0.0:
                flipped_batch = {
                    **batch,
                    "texture": torch.flip(batch["texture"], dims=(-1,)),
                }
                flipped_outputs = model(flipped_batch)
                flipped_loss, flipped_parts = qalf_loss(
                    flipped_outputs,
                    labels,
                    criterion,
                    geometry_weight,
                    texture_weight,
                    label_smoothing,
                )
                consistency = flip_consistency_loss(outputs, flipped_outputs)
                loss = 0.5 * (loss + flipped_loss) + flip_consistency_weight * consistency
                parts = {
                    name: 0.5 * (value + flipped_parts[name])
                    for name, value in parts.items()
                }
                parts["flip_consistency"] = float(consistency.detach())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler skips optimizer.step when it detects non-finite gradients.
        # Keep EMA synchronized only with optimizer updates that actually happened.
        optimizer_updated = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
        if ema is not None and optimizer_updated:
            ema.update(model)
        batch_size = int(labels.shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in parts.items():
            totals[name] += value * batch_size
        samples += batch_size
    if samples == 0:
        raise RuntimeError("Training loader produced no samples")
    return {name: value / samples for name, value in totals.items()}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    texture_flip_tta: bool = False,
) -> dict[str, object]:
    model.eval()
    result: defaultdict[str, list] = defaultdict(list)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch(batch, device)
        outputs = model(batch)
        scores = torch.sigmoid(outputs["logit"])
        geometry_scores = torch.sigmoid(outputs["geometry_logit"])
        texture_scores = torch.sigmoid(outputs["texture_logit"])
        fusion_weights = outputs["fusion_weights"]
        if texture_flip_tta:
            flipped_batch = {
                **batch,
                "texture": torch.flip(batch["texture"], dims=(-1,)),
            }
            flipped_outputs = model(flipped_batch)
            scores = 0.5 * (scores + torch.sigmoid(flipped_outputs["logit"]))
            geometry_scores = 0.5 * (
                geometry_scores + torch.sigmoid(flipped_outputs["geometry_logit"])
            )
            texture_scores = 0.5 * (
                texture_scores + torch.sigmoid(flipped_outputs["texture_logit"])
            )
            fusion_weights = 0.5 * (fusion_weights + flipped_outputs["fusion_weights"])
        result["label"].extend(batch["label"].detach().cpu().numpy().tolist())
        result["score"].extend(scores.cpu().numpy().tolist())
        result["geometry_score"].extend(geometry_scores.cpu().numpy().tolist())
        result["texture_score"].extend(texture_scores.cpu().numpy().tolist())
        result["geometry_weight"].extend(fusion_weights[:, 0].cpu().numpy().tolist())
        result["texture_weight"].extend(fusion_weights[:, 1].cpu().numpy().tolist())
        result["clip_index"].extend(batch["clip_index"].detach().cpu().numpy().tolist())
        for key in ("video_id", "method", "dataset"):
            result[key].extend(batch[key])
    return dict(result)


def aggregate_predictions(
    predictions: dict[str, object],
    method: str = "mean",
    top_k: int = 1,
) -> dict[str, object]:
    """Aggregate ordered clip predictions into one row per dataset/method/video."""

    if method not in {"mean", "median", "topk"}:
        raise ValueError(f"Unsupported video aggregation: {method}")
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    required = {
        "label",
        "score",
        "geometry_score",
        "texture_score",
        "geometry_weight",
        "texture_weight",
        "video_id",
        "method",
        "dataset",
    }
    missing = required - predictions.keys()
    if missing:
        raise ValueError(f"Predictions are missing fields: {sorted(missing)}")

    groups: dict[tuple[str, str, str], list[int]] = {}
    total = len(predictions["score"])
    for index in range(total):
        key = (
            str(predictions["dataset"][index]),
            str(predictions["method"][index]),
            str(predictions["video_id"][index]),
        )
        groups.setdefault(key, []).append(index)

    result: defaultdict[str, list] = defaultdict(list)
    numeric_fields = (
        "score",
        "geometry_score",
        "texture_score",
        "geometry_weight",
        "texture_weight",
    )
    for (dataset, manipulation, video_id), indices in groups.items():
        labels = {int(round(float(predictions["label"][index]))) for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Inconsistent clip labels for {dataset}/{manipulation}/{video_id}")
        selected = indices
        if method == "topk":
            selected = sorted(
                indices,
                key=lambda index: float(predictions["score"][index]),
                reverse=True,
            )[: min(top_k, len(indices))]
        for field in numeric_fields:
            values = np.asarray(
                [float(predictions[field][index]) for index in selected], dtype=np.float64
            )
            value = float(np.median(values)) if method == "median" else float(np.mean(values))
            result[field].append(value)
        result["label"].append(labels.pop())
        result["video_id"].append(video_id)
        result["method"].append(manipulation)
        result["dataset"].append(dataset)
        result["clip_count"].append(len(indices))
    return dict(result)
