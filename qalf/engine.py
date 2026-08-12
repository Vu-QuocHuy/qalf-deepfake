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
    geometry_loss_mask: torch.Tensor | None = None,
    reliability_gate_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    def mean_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        value = criterion(logits, targets)
        return value.mean() if value.ndim > 0 else value

    fused = mean_loss(outputs["logit"], labels)
    texture = mean_loss(outputs["texture_logit"], labels)
    if geometry_loss_mask is None:
        geometry = mean_loss(outputs["geometry_logit"], labels)
    else:
        valid = geometry_loss_mask.reshape(-1) > 0.5
        if bool(valid.any()):
            valid_logits = outputs["geometry_logit"][valid]
            valid_labels = labels[valid]
            geometry = mean_loss(valid_logits, valid_labels)
        else:
            # Preserve a differentiable graph for a hypothetical all-SBI batch.
            geometry = outputs["geometry_logit"].sum() * 0.0
    reliability = outputs["logit"].sum() * 0.0
    reliability_target = outputs.get("reliability_target")
    if reliability_gate_weight > 0.0 and reliability_target is not None:
        supervised = reliability_target >= 0
        if bool(supervised.any()):
            reliability = F.nll_loss(
                outputs["fusion_weights"][supervised].clamp_min(1e-6).log(),
                reliability_target[supervised],
            )
    total = (
        fused
        + geometry_weight * geometry
        + texture_weight * texture
        + reliability_gate_weight * reliability
    )
    return total, {
        "fused": float(fused.detach()),
        "geometry": float(geometry.detach()),
        "texture": float(texture.detach()),
        "reliability": float(reliability.detach()),
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
    reliability_gate_weight: float = 0.0,
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
            geometry_loss_mask = batch.get("geometry_loss_mask")
            loss, parts = qalf_loss(
                outputs,
                labels,
                criterion,
                geometry_weight,
                texture_weight,
                geometry_loss_mask if torch.is_tensor(geometry_loss_mask) else None,
                reliability_gate_weight=reliability_gate_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in parts.items():
            totals[name] += value * batch_size
        if torch.is_tensor(geometry_loss_mask):
            totals["geometry_supervised"] += float(geometry_loss_mask.sum().detach())
        else:
            totals["geometry_supervised"] += batch_size
        sample_types = batch.get("sample_type")
        if isinstance(sample_types, (list, tuple)):
            totals["sbi_samples"] += sum(value == "sbi" for value in sample_types)
        reliability_target = outputs.get("reliability_target")
        if torch.is_tensor(reliability_target):
            totals["reliability_supervised"] += float((reliability_target >= 0).sum().detach())
        samples += batch_size
    if samples == 0:
        raise RuntimeError("Training loader produced no samples")
    result = {
        name: value / samples
        for name, value in totals.items()
        if name not in {"geometry_supervised", "sbi_samples", "reliability_supervised"}
    }
    result["geometry_supervision_fraction"] = totals["geometry_supervised"] / samples
    result["sbi_fraction"] = totals["sbi_samples"] / samples
    result["reliability_supervision_fraction"] = totals["reliability_supervised"] / samples
    return result


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    texture_flip_tta: bool = False,
    zero_geometry_counterfactual: bool = False,
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
        zero_geometry_scores = None
        if zero_geometry_counterfactual:
            if not hasattr(model, "fuse_precomputed"):
                raise TypeError(
                    "zero-geometry counterfactual requires a model with fuse_precomputed"
                )
            zero_geometry_logit, _ = model.fuse_precomputed(
                torch.zeros_like(outputs["geometry_embedding"]),
                torch.zeros_like(outputs["geometry_logit"]),
                outputs["texture_embedding"],
                outputs["texture_logit"],
                torch.zeros_like(batch["geometry_quality"]),
                batch["texture_quality"],
            )
            zero_geometry_scores = torch.sigmoid(zero_geometry_logit)
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
            if zero_geometry_counterfactual:
                flipped_zero_geometry_logit, _ = model.fuse_precomputed(
                    torch.zeros_like(flipped_outputs["geometry_embedding"]),
                    torch.zeros_like(flipped_outputs["geometry_logit"]),
                    flipped_outputs["texture_embedding"],
                    flipped_outputs["texture_logit"],
                    torch.zeros_like(batch["geometry_quality"]),
                    batch["texture_quality"],
                )
                assert zero_geometry_scores is not None
                zero_geometry_scores = 0.5 * (
                    zero_geometry_scores + torch.sigmoid(flipped_zero_geometry_logit)
                )
        result["label"].extend(batch["label"].detach().cpu().numpy().tolist())
        result["score"].extend(scores.cpu().numpy().tolist())
        result["geometry_score"].extend(geometry_scores.cpu().numpy().tolist())
        result["texture_score"].extend(texture_scores.cpu().numpy().tolist())
        result["geometry_weight"].extend(fusion_weights[:, 0].cpu().numpy().tolist())
        result["texture_weight"].extend(fusion_weights[:, 1].cpu().numpy().tolist())
        if zero_geometry_scores is not None:
            result["zero_geometry_score"].extend(zero_geometry_scores.cpu().numpy().tolist())
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
    numeric_fields = [
        "score",
        "geometry_score",
        "texture_score",
        "geometry_weight",
        "texture_weight",
    ]
    if "zero_geometry_score" in predictions:
        numeric_fields.append("zero_geometry_score")
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
