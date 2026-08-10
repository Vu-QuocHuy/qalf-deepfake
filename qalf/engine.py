"""Training and inference loops."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm


class GradScalerProtocol(Protocol):
    """Common interface of the legacy and current PyTorch gradient scalers."""

    def is_enabled(self) -> bool: ...

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
) -> tuple[torch.Tensor, dict[str, float]]:
    fused = criterion(outputs["logit"], labels)
    geometry = criterion(outputs["geometry_logit"], labels)
    texture = criterion(outputs["texture_logit"], labels)
    total = fused + geometry_weight * geometry + texture_weight * texture
    return total, {
        "fused": float(fused.detach()),
        "geometry": float(geometry.detach()),
        "texture": float(texture.detach()),
    }


def train_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScalerProtocol,
    geometry_weight: float,
    texture_weight: float,
    self_blend_weight: float = 0.0,
    method_discriminator: nn.Module | None = None,
    method_to_index: dict[str, int] | None = None,
    method_adversarial_weight: float = 0.0,
    method_grl_strength: float = 1.0,
) -> dict[str, float]:
    model.train()
    if method_discriminator is not None:
        method_discriminator.train()
    totals: defaultdict[str, float] = defaultdict(float)
    samples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(batch)
            loss, parts = qalf_loss(outputs, labels, criterion, geometry_weight, texture_weight)
            if self_blend_weight > 0.0 and "self_blend_valid" in batch:
                valid = batch["self_blend_valid"].bool()
                if bool(valid.any()):
                    _, self_blend_logits = model.forward_texture(batch["self_blend_texture"][valid])
                    self_blend_loss = criterion(
                        self_blend_logits,
                        torch.ones_like(self_blend_logits),
                    )
                    loss = loss + self_blend_weight * self_blend_loss
                    parts["self_blend"] = float(self_blend_loss.detach())
                else:
                    parts["self_blend"] = 0.0
            if (
                method_discriminator is not None
                and method_to_index
                and method_adversarial_weight > 0.0
            ):
                method_pairs = [
                    (index, method_to_index[method])
                    for index, method in enumerate(batch["method"])
                    if method in method_to_index
                ]
                if method_pairs:
                    method_indices = torch.tensor(
                        [pair[0] for pair in method_pairs],
                        dtype=torch.long,
                        device=device,
                    )
                    method_targets = torch.tensor(
                        [pair[1] for pair in method_pairs],
                        dtype=torch.long,
                        device=device,
                    )
                    method_logits = method_discriminator(
                        outputs["texture_embedding"].index_select(0, method_indices),
                        method_grl_strength,
                    )
                    method_loss = F.cross_entropy(method_logits, method_targets)
                    loss = loss + method_adversarial_weight * method_loss
                    parts["method_adversarial"] = float(method_loss.detach())
                else:
                    parts["method_adversarial"] = 0.0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = list(model.parameters())
        if method_discriminator is not None:
            parameters.extend(method_discriminator.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
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
) -> dict[str, object]:
    model.eval()
    result: defaultdict[str, list] = defaultdict(list)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch(batch, device)
        outputs = model(batch)
        result["label"].extend(batch["label"].detach().cpu().numpy().tolist())
        result["score"].extend(torch.sigmoid(outputs["logit"]).cpu().numpy().tolist())
        result["geometry_score"].extend(
            torch.sigmoid(outputs["geometry_logit"]).cpu().numpy().tolist()
        )
        result["texture_score"].extend(
            torch.sigmoid(outputs["texture_logit"]).cpu().numpy().tolist()
        )
        result["geometry_weight"].extend(outputs["fusion_weights"][:, 0].cpu().numpy().tolist())
        result["texture_weight"].extend(outputs["fusion_weights"][:, 1].cpu().numpy().tolist())
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
