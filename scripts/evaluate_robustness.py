#!/usr/bin/env python3
"""Evaluate robustness against common image corruptions.

Applies parameterized corruptions (JPEG compression, Gaussian blur,
Resolution downscaling, Gaussian noise) to the input sequence before
forwarding to the model. Reports AUC at different corruption severities.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import QALFVideoDataset
from qalf.engine import move_batch
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import QALFModel, build_model_from_checkpoint


def get_corruptions() -> dict[str, list[tuple[str, Callable[[np.ndarray], np.ndarray]]]]:
    def _jpeg(quality: int) -> Callable[[np.ndarray], np.ndarray]:
        def apply(img: np.ndarray) -> np.ndarray:
            success, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not success:
                return img
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR)  # type: ignore[no-any-return]
        return apply

    def _blur(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
        def apply(img: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        return apply

    def _downscale(scale: float) -> Callable[[np.ndarray], np.ndarray]:
        def apply(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return apply

    def _noise(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
        def apply(img: np.ndarray) -> np.ndarray:
            noise = np.random.normal(0, sigma, img.shape)
            noisy = img.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        return apply

    return {
        "jpeg": [(f"q={q}", _jpeg(q)) for q in (80, 60, 40, 20)],
        "blur": [(f"sigma={s}", _blur(s)) for s in (1.0, 2.0, 3.0)],
        "downscale": [(f"scale={s}", _downscale(s)) for s in (0.5, 0.25)],
        "noise": [(f"std={s}", _noise(s)) for s in (5.0, 10.0, 20.0)],
    }


def evaluate_corruption(
    model: QALFModel,
    loader: DataLoader,
    device: torch.device,
    corruption_fn: Callable[[np.ndarray], np.ndarray] | None,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            # Apply corruption
            if corruption_fn is not None:
                textures = batch["texture"].numpy()  # (B, T, C, H, W)
                # Convert to (B, T, H, W, C) for cv2
                textures = np.transpose(textures, (0, 1, 3, 4, 2)).astype(np.uint8)
                for b in range(textures.shape[0]):
                    for t in range(textures.shape[1]):
                        textures[b, t] = corruption_fn(textures[b, t])
                # Convert back to (B, T, C, H, W) float tensor
                textures = np.transpose(textures, (0, 1, 4, 2, 3))
                batch["texture"] = torch.from_numpy(textures).float()

            batch = move_batch(batch, device)
            output = model(batch)
            scores = torch.sigmoid(output["logit"]).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    return compute_metrics(np.array(all_labels), np.array(all_scores), threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    data = config["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(checkpoint).to(device)

    dataset = QALFVideoDataset(
        args.manifest,
        args.frame_root,
        args.landmark_root,
        num_frames=int(data["num_frames"]),
        texture_frames=int(data["texture_frames"]),
        image_size=int(data["image_size"]),
        training=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    print("Evaluating baseline (no corruption)...")
    baseline_metrics = evaluate_corruption(model, loader, device, None)
    
    results = {
        "baseline": {"auc": baseline_metrics["auc"], "eer": baseline_metrics["eer"]},
        "corruptions": {}
    }
    
    for c_type, levels in get_corruptions().items():
        results["corruptions"][c_type] = {}
        for c_name, c_fn in levels:
            print(f"Evaluating {c_type} {c_name}...")
            metrics = evaluate_corruption(model, loader, device, c_fn)
            results["corruptions"][c_type][c_name] = {
                "auc": metrics["auc"],
                "eer": metrics["eer"]
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\nRobustness Summary (AUC):")
    print(f"Baseline: {results['baseline']['auc']:.4f}")
    for c_type, levels in results["corruptions"].items():
        print(f"\n{c_type.upper()}:")
        for c_name, metrics in levels.items():
            print(f"  {c_name}: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
