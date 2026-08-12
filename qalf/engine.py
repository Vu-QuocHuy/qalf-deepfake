"""Texture-only training and video inference loops."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Protocol

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


class GradScalerProtocol(Protocol):
    def is_enabled(self) -> bool: ...

    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...

    def step(self, optimizer: torch.optim.Optimizer) -> object: ...

    def update(self) -> None: ...


class EMAProtocol(Protocol):
    def update(self, model: nn.Module) -> None: ...


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScalerProtocol,
    ema: EMAProtocol | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    samples = 0
    sbi_samples = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(batch)
            loss = criterion(outputs["logit"], labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        sample_types = batch.get("sample_type")
        if isinstance(sample_types, (list, tuple)):
            sbi_samples += sum(value == "sbi" for value in sample_types)
        samples += batch_size
    if samples == 0:
        raise RuntimeError("Training loader produced no samples")
    return {
        "loss": total_loss / samples,
        "sbi_fraction": sbi_samples / samples,
    }


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
        scores = torch.sigmoid(model(batch)["logit"])
        if texture_flip_tta:
            flipped = {**batch, "texture": torch.flip(batch["texture"], dims=(-1,))}
            scores = 0.5 * (scores + torch.sigmoid(model(flipped)["logit"]))
        result["label"].extend(batch["label"].detach().cpu().numpy().tolist())
        result["score"].extend(scores.cpu().numpy().tolist())
        result["clip_index"].extend(batch["clip_index"].detach().cpu().numpy().tolist())
        for key in ("video_id", "method", "dataset"):
            result[key].extend(batch[key])
    return dict(result)


def aggregate_predictions(
    predictions: dict[str, object],
    method: str = "mean",
    top_k: int = 1,
) -> dict[str, object]:
    """Aggregate clip scores into one score per dataset/method/video."""

    if method not in {"mean", "median", "topk"}:
        raise ValueError(f"Unsupported video aggregation: {method}")
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    required = {"label", "score", "video_id", "method", "dataset"}
    missing = required - predictions.keys()
    if missing:
        raise ValueError(f"Predictions are missing fields: {sorted(missing)}")

    groups: dict[tuple[str, str, str], list[int]] = {}
    for index in range(len(predictions["score"])):
        key = (
            str(predictions["dataset"][index]),
            str(predictions["method"][index]),
            str(predictions["video_id"][index]),
        )
        groups.setdefault(key, []).append(index)

    result: defaultdict[str, list] = defaultdict(list)
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
        scores = np.asarray(
            [float(predictions["score"][index]) for index in selected],
            dtype=np.float64,
        )
        score = float(np.median(scores)) if method == "median" else float(np.mean(scores))
        result["score"].append(score)
        result["label"].append(labels.pop())
        result["video_id"].append(video_id)
        result["method"].append(manipulation)
        result["dataset"].append(dataset)
        result["clip_count"].append(len(indices))
    return dict(result)
