#!/usr/bin/env python3
"""Comprehensive efficiency benchmark for QALF v2 models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.models import build_model_from_checkpoint


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    counts = {}
    counts["total"] = sum(p.numel() for p in model.parameters())
    counts["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Backbone
    counts["backbone"] = sum(p.numel() for p in model.texture_encoder.features.parameters())
    
    # Frequency
    counts["frequency"] = 0
    if model.texture_encoder.frequency is not None:
        counts["frequency"] = sum(p.numel() for p in model.texture_encoder.frequency.parameters())
        
    # Multiscale
    counts["multiscale"] = 0
    if model.texture_encoder.multiscale_agg is not None:
        counts["multiscale"] = sum(p.numel() for p in model.texture_encoder.multiscale_agg.parameters())
        
    # Temporal
    counts["temporal"] = 0
    if model.texture_encoder.temporal_pool is not None:
        counts["temporal"] = sum(p.numel() for p in model.texture_encoder.temporal_pool.parameters())
        
    # Projection (v1)
    counts["projection"] = 0
    if model.texture_encoder.projection is not None:
        counts["projection"] = sum(p.numel() for p in model.texture_encoder.projection.parameters())
        
    counts["classifier"] = sum(p.numel() for p in model.texture_encoder.classifier.parameters())
    
    return counts


def measure_latency(model: torch.nn.Module, device: torch.device, input_shape: tuple[int, ...], iterations: int = 100) -> dict[str, float]:
    model.eval()
    dummy_input = {"texture": torch.randn(input_shape).to(device)}
    
    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
                
    end_time = time.perf_counter()
    avg_latency_ms = (end_time - start_time) * 1000 / iterations
    return {"latency_ms": avg_latency_ms, "fps": 1000 / avg_latency_ms if avg_latency_ms > 0 else 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    data = config["data"]
    
    model = build_model_from_checkpoint(checkpoint)
    params = count_parameters(model)
    
    print("Parameter Breakdown:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # CPU Latency (Single Thread)
    torch.set_num_threads(1)
    cpu_model = build_model_from_checkpoint(checkpoint).to(torch.device("cpu"))
    input_shape = (1, int(data["texture_frames"]), 3, int(data["image_size"]), int(data["image_size"]))
    print(f"\nMeasuring CPU Latency (1 thread) with shape {input_shape}...")
    cpu_latency = measure_latency(cpu_model, torch.device("cpu"), input_shape)
    
    # CPU Latency (Multi Thread)
    torch.set_num_threads(4)
    print(f"Measuring CPU Latency (4 threads) with shape {input_shape}...")
    cpu_latency_multi = measure_latency(cpu_model, torch.device("cpu"), input_shape)
    
    results = {
        "parameters": params,
        "cpu_latency_1_thread_ms": cpu_latency["latency_ms"],
        "cpu_fps_1_thread": cpu_latency["fps"],
        "cpu_latency_4_threads_ms": cpu_latency_multi["latency_ms"],
        "cpu_fps_4_threads": cpu_latency_multi["fps"],
    }
    
    if torch.cuda.is_available():
        gpu_model = build_model_from_checkpoint(checkpoint).to(torch.device("cuda"))
        print(f"Measuring GPU Latency with shape {input_shape}...")
        gpu_latency = measure_latency(gpu_model, torch.device("cuda"), input_shape)
        results["gpu_latency_ms"] = gpu_latency["latency_ms"]
        results["gpu_fps"] = gpu_latency["fps"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\nLatency Summary:")
    print(f"  CPU (1 thread): {results['cpu_latency_1_thread_ms']:.2f} ms ({results['cpu_fps_1_thread']:.2f} FPS)")
    print(f"  CPU (4 threads): {results['cpu_latency_4_threads_ms']:.2f} ms ({results['cpu_fps_4_threads']:.2f} FPS)")
    if "gpu_latency_ms" in results:
        print(f"  GPU: {results['gpu_latency_ms']:.2f} ms ({results['gpu_fps']:.2f} FPS)")


if __name__ == "__main__":
    main()
