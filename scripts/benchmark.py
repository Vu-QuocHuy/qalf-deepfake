#!/usr/bin/env python3
"""Measure incremental model latency and memory with batch size one."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import QALFVideoDataset
from qalf.engine import move_batch
from qalf.models import QALFModel


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    data, model_config = config["data"], config["model"]
    dataset = QALFVideoDataset(
        args.manifest,
        args.frame_root,
        args.landmark_root,
        num_frames=int(data["num_frames"]),
        texture_frames=int(data["texture_frames"]),
        image_size=int(data["image_size"]),
        geometry_mode=str(data.get("geometry_mode", "aligned_motion")),
        texture_mode=str(data.get("texture_mode", "canonical_skin")),
        training=False,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    model = QALFModel(
        geometry_input_dim=int(checkpoint["geometry_input_dim"]),
        geometry_hidden=int(model_config.get("geometry_hidden", 96)),
        geometry_layers=int(model_config.get("geometry_layers", 3)),
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=False,
        geometry_quality_dim=int(checkpoint.get("geometry_quality_dim", 5)),
        texture_quality_dim=int(checkpoint.get("texture_quality_dim", 5)),
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batch = move_batch(batch, device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(batch)
        synchronize()
        timings: list[float] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(args.iterations):
            synchronize()
            started = time.perf_counter()
            model(batch)
            synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)

    report = {
        "device": str(device),
        "fusion_mode": str(model_config.get("fusion_mode", "quality")),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "latency_ms_mean": statistics.mean(timings),
        "latency_ms_p50": percentile(timings, 0.50),
        "latency_ms_p95": percentile(timings, 0.95),
        "throughput_fps_from_mean": 1000.0 / statistics.mean(timings),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "note": "Incremental model cost only; video decode, MTCNN, and Face Mesh are excluded.",
    }
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
