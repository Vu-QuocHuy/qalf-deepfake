#!/usr/bin/env python3
"""Evaluate a texture-only checkpoint under realistic video corruptions.

This tool is evaluation-only. It denormalizes the dataset's ImageNet-normalized
RGB tensors, applies a corruption, normalizes them again, and then computes
video-level metrics. A threshold is fitted only on clean FF++ validation when
the optional threshold manifest is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import IMAGE_MEAN, IMAGE_STD, QALFVideoDataset
from qalf.engine import aggregate_predictions, move_batch
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import build_model_from_checkpoint

Corruption = Callable[[np.ndarray], np.ndarray]


def _jpeg(quality: int) -> Corruption:
    def apply(image: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return image if decoded is None else cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    return apply


def _blur(sigma: float) -> Corruption:
    return lambda image: cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def _downscale(scale: float) -> Corruption:
    def apply(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        small = cv2.resize(
            image,
            (max(8, int(width * scale)), max(8, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)

    return apply


def _noise(std: float, seed: int) -> Corruption:
    rng = np.random.default_rng(seed)

    def apply(image: np.ndarray) -> np.ndarray:
        noisy = image.astype(np.float32) + rng.normal(0.0, std, image.shape)
        return np.clip(noisy, 0.0, 255.0).astype(np.uint8)

    return apply


def corruption_suite(seed: int) -> dict[str, Corruption | None]:
    return {
        "clean": None,
        "jpeg_q80": _jpeg(80),
        "jpeg_q60": _jpeg(60),
        "jpeg_q40": _jpeg(40),
        "jpeg_q20": _jpeg(20),
        "blur_sigma1": _blur(1.0),
        "blur_sigma2": _blur(2.0),
        "blur_sigma3": _blur(3.0),
        "downscale_050": _downscale(0.50),
        "downscale_025": _downscale(0.25),
        "noise_std5": _noise(5.0, seed + 5),
        "noise_std10": _noise(10.0, seed + 10),
        "noise_std20": _noise(20.0, seed + 20),
    }


def _denormalize(texture: torch.Tensor) -> np.ndarray:
    mean = torch.as_tensor(IMAGE_MEAN, dtype=texture.dtype).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(IMAGE_STD, dtype=texture.dtype).view(1, 1, 3, 1, 1)
    raw = ((texture * std + mean) * 255.0).round().clamp(0.0, 255.0)
    return raw.to(torch.uint8).permute(0, 1, 3, 4, 2).numpy()


def _normalize(raw: np.ndarray) -> torch.Tensor:
    texture = torch.from_numpy(raw).permute(0, 1, 4, 2, 3).float() / 255.0
    mean = torch.as_tensor(IMAGE_MEAN, dtype=texture.dtype).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(IMAGE_STD, dtype=texture.dtype).view(1, 1, 3, 1, 1)
    return (texture - mean) / std


def corrupt_texture(texture: torch.Tensor, corruption: Corruption | None) -> torch.Tensor:
    """Corrupt normalized ``(B,T,C,H,W)`` RGB tensors without uint8 loss."""

    if corruption is None:
        return texture
    raw = _denormalize(texture)
    for batch_index in range(raw.shape[0]):
        for frame_index in range(raw.shape[1]):
            raw[batch_index, frame_index] = corruption(
                np.ascontiguousarray(raw[batch_index, frame_index])
            )
    return _normalize(raw)


@torch.inference_mode()
def predict_condition(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corruption: Corruption | None,
    texture_flip_tta: bool,
) -> dict[str, object]:
    model.eval()
    result: dict[str, list] = {
        "label": [], "score": [], "clip_index": [],
        "video_id": [], "method": [], "dataset": [],
    }
    for batch in loader:
        batch = dict(batch)
        batch["texture"] = corrupt_texture(batch["texture"], corruption)
        model_batch = move_batch(batch, device)
        scores = torch.sigmoid(model(model_batch)["logit"])
        if texture_flip_tta:
            flipped = {**model_batch, "texture": torch.flip(model_batch["texture"], dims=(-1,))}
            scores = 0.5 * (scores + torch.sigmoid(model(flipped)["logit"]))
        result["label"].extend(model_batch["label"].detach().cpu().numpy().tolist())
        result["score"].extend(scores.detach().cpu().numpy().tolist())
        result["clip_index"].extend(model_batch["clip_index"].detach().cpu().numpy().tolist())
        for key in ("video_id", "method", "dataset"):
            result[key].extend(model_batch[key])
    return result


def _dataset(
    manifest: str,
    frame_root: str,
    landmark_root: str,
    data: dict[str, object],
    frames: int,
    clips: int,
    fake_methods: object | None = None,
) -> QALFVideoDataset:
    return QALFVideoDataset(
        manifest, frame_root, landmark_root,
        num_frames=int(data["num_frames"]), texture_frames=frames,
        image_size=int(data["image_size"]), texture_mode=str(data.get("texture_mode", "full_face")),
        temporal_sampling=str(data.get("temporal_sampling", "uniform")),
        training=False, clips_per_video=clips, fake_methods=fake_methods,
    )


def _loader(dataset: QALFVideoDataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def _write_markdown(path: Path, rows: list[dict[str, object]], protocol: dict[str, object]) -> None:
    clean_auc = float(rows[0]["auc"])
    lines = [
        "# QALF robustness evaluation", "",
        "Corruptions are applied after RGB denormalization. The threshold is "
        "selected on clean FF++ validation only.", "",
        f"- checkpoint: `{protocol['checkpoint']}`",
        f"- texture frames: `{protocol['texture_frames']}`",
        f"- clips/video: `{protocol['clips_per_video']}`",
        f"- aggregation: `{protocol['aggregation']}`", "",
        "| Condition | AUC | AP | EER | Balanced accuracy | ACER | AUC drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        auc = float(row["auc"])
        lines.append(
            f"| {row['condition']} | {auc:.4f} | {float(row['average_precision']):.4f} | "
            f"{float(row['eer']):.4f} | {float(row['balanced_accuracy']):.4f} | "
            f"{float(row['acer']):.4f} | {auc - clean_auc:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-manifest")
    parser.add_argument("--threshold-frame-root")
    parser.add_argument("--threshold-landmark-root")
    parser.add_argument("--threshold-clips-per-video", type=int, default=3)
    parser.add_argument(
        "--threshold-selection",
        choices=("youden_j", "eer"),
        default="youden_j",
        help="Validation threshold rule; EER means closest finite ROC point.",
    )
    parser.add_argument("--texture-frames", type=int, default=12)
    parser.add_argument("--clips-per-video", type=int, default=3)
    parser.add_argument("--aggregation", choices=("mean", "median", "topk"), default="mean")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texture-flip-tta", action="store_true")
    args = parser.parse_args()

    threshold_paths = (args.threshold_manifest, args.threshold_frame_root, args.threshold_landmark_root)
    if any(threshold_paths) and not all(threshold_paths):
        parser.error("threshold manifest, frame root, and landmark root must be provided together")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(
        f"Robustness evaluation started: device={device} "
        f"checkpoint={args.checkpoint}",
        flush=True,
    )
    data = checkpoint["config"]["data"]
    model_config = checkpoint["config"]["model"]
    dataset = _dataset(args.manifest, args.frame_root, args.landmark_root, data, args.texture_frames, args.clips_per_video)
    loader = _loader(dataset, args.batch_size, args.num_workers)
    threshold = float(checkpoint.get("threshold", 0.5))
    threshold_source = "checkpoint"
    if args.threshold_manifest:
        print("Calibrating threshold on clean FF++ validation...", flush=True)
        threshold_dataset = _dataset(
            args.threshold_manifest,
            args.threshold_frame_root,
            args.threshold_landmark_root,
            data,
            args.texture_frames,
            args.threshold_clips_per_video,
            fake_methods=data.get("fake_methods"),
        )
        threshold_predictions = aggregate_predictions(
            predict_condition(model, _loader(threshold_dataset, args.batch_size, args.num_workers), device, None, args.texture_flip_tta),
            method=args.aggregation, top_k=args.top_k)
        threshold = select_threshold(
            np.asarray(threshold_predictions["label"], dtype=np.int64),
            np.asarray(threshold_predictions["score"], dtype=np.float64),
            strategy=args.threshold_selection,
        )
        threshold_source = args.threshold_manifest
        print(f"Threshold calibrated: {threshold:.4f}", flush=True)

    rows: list[dict[str, object]] = []
    suite = corruption_suite(args.seed)
    for index, (condition, corruption) in enumerate(suite.items(), 1):
        print(f"[{index}/{len(suite)}] evaluating {condition}...", flush=True)
        predictions = aggregate_predictions(
            predict_condition(model, loader, device, corruption, args.texture_flip_tta),
            method=args.aggregation, top_k=args.top_k)
        labels = np.asarray(predictions["label"], dtype=np.int64)
        scores = np.asarray(predictions["score"], dtype=np.float64)
        metrics = compute_metrics(labels, scores, threshold)
        rows.append({"condition": condition, **metrics})
        print(
            f"[{index}/{len(suite)}] {condition}: "
            f"auc={float(metrics['auc']):.4f} acer={float(metrics['acer']):.4f}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "checkpoint": str(args.checkpoint), "manifest": str(args.manifest),
        "texture_frames": args.texture_frames, "clips_per_video": args.clips_per_video,
        "aggregation": args.aggregation, "top_k": args.top_k,
        "texture_flip_tta": args.texture_flip_tta, "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_selection": args.threshold_selection,
        "corruption_seed": args.seed,
        "temporal_sampling": data.get("temporal_sampling", "uniform"),
        "temporal_pooling": model_config.get("temporal_pooling", "mean"),
    }
    output.write_text(json.dumps({"protocol": protocol, "results": rows}, indent=2), encoding="utf-8")
    _write_markdown(output.with_suffix(".md"), rows, protocol)
    print(f"Robustness results: {output}")
    print(f"Markdown summary: {output.with_suffix('.md')}")
    for row in rows:
        print(f"{row['condition']}: auc={float(row['auc']):.4f} acer={float(row['acer']):.4f}")


if __name__ == "__main__":
    main()
