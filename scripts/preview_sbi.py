#!/usr/bin/env python3
"""Save a small pristine/SBI preview grid without starting training."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.config import load_config
from qalf.data.dataset import IMAGE_MEAN, IMAGE_STD, QALFVideoDataset
from qalf.data.sbi import SAMPLE_REAL, SAMPLE_SBI


def _rgb_from_tensor(texture: torch.Tensor, frame_index: int) -> np.ndarray:
    frame = texture[frame_index].detach().cpu().numpy().transpose(1, 2, 0)
    frame = frame * IMAGE_STD + IMAGE_MEAN
    return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ffpp_to_celebdf.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least one")

    config = load_config(args.config)
    data = config["data"]
    sbi_config = dict(data["sbi"])
    sbi_config["enabled"] = True
    dataset = QALFVideoDataset(
        args.manifest,
        args.frame_root,
        args.landmark_root,
        num_frames=int(data["num_frames"]),
        texture_frames=int(data["texture_frames"]),
        image_size=int(data["image_size"]),
        texture_mode="full_face",
        temporal_sampling=str(data.get("temporal_sampling", "uniform")),
        training=True,
        clips_per_video=1,
        texture_augmentation={},
        fake_methods=data.get("fake_methods"),
        sbi_config=sbi_config,
    )

    pristine_indices = {
        record_index: sample_index
        for sample_index, (record_index, sample_type) in enumerate(dataset.sample_specs)
        if sample_type == SAMPLE_REAL
    }
    sbi_indices = {
        record_index: sample_index
        for sample_index, (record_index, sample_type) in enumerate(dataset.sample_specs)
        if sample_type == SAMPLE_SBI
    }
    record_indices = sorted(set(pristine_indices) & set(sbi_indices))[: args.samples]
    if not record_indices:
        raise RuntimeError("No real/SBI companion samples are available")

    rows: list[np.ndarray] = []
    for offset, record_index in enumerate(record_indices):
        sample_seed = args.seed + offset
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        pristine = dataset[pristine_indices[record_index]]
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        synthetic = dataset[sbi_indices[record_index]]
        frame_index = int(data["texture_frames"]) // 2
        pristine_rgb = _rgb_from_tensor(pristine["texture"], frame_index)
        synthetic_rgb = _rgb_from_tensor(synthetic["texture"], frame_index)
        difference = cv2.absdiff(pristine_rgb, synthetic_rgb)
        video_id = str(pristine["video_id"])
        rows.append(
            np.concatenate(
                [
                    _label(pristine_rgb, f"real: {video_id}"),
                    _label(synthetic_rgb, "SBI"),
                    _label(difference, "absolute difference"),
                ],
                axis=1,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.concatenate(rows, axis=0)):
        raise RuntimeError(f"Could not write preview image: {output}")
    print(f"SBI preview saved: {output}")


if __name__ == "__main__":
    main()
