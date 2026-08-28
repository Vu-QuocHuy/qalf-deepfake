#!/usr/bin/env python3
"""Evaluate TextureSBI on Server using PyTorch and YuNet end-to-end on raw videos.

This script executes the Condition C experiment: Server + PyTorch + YuNet on raw video.
It loads a QALF manifest (e.g., Celeb-DF-v2), runs the exact same YuNet preprocessing
as on the Pi 4, and performs inference using PyTorch (CPU or GPU).
Outputs are saved in standard predictions.csv and metrics.json format.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.manifest import load_manifest, VideoRecord
from qalf.metrics import compute_metrics
from qalf.models import build_model_from_checkpoint
from qalf.data.landmarks import FaceLandmarkerExtractor, OpenCVYuNetLandmarker, ensure_face_landmarker_model
from scripts.infer_video import process_video_pipeline


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Server YuNet End-to-End Evaluation")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL (e.g., celebdf_test_518.jsonl)")
    parser.add_argument("--video-root", required=True, help="Root directory for raw videos")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="PyTorch inference device (cuda or cpu)")
    
    # Protocol args
    parser.add_argument("--num-frames", type=int, default=32, help="Temporal window size")
    parser.add_argument("--texture-frames", type=int, default=8, help="Texture frames per clip")
    parser.add_argument("--clips-per-video", type=int, default=3, help="Number of clips per video")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target temporal sampling rate")
    parser.add_argument("--image-size", type=int, default=160, help="Input resolution")
    parser.add_argument("--aggregation", choices=["mean", "topk"], default="mean", help="Clip score aggregation method")
    parser.add_argument("--top-k", type=int, default=1, help="Top-K clips to aggregate if using topk")
    parser.add_argument("--no-flip-tta", action="store_true", help="Disable horizontal flip TTA")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (auto-loaded if not set)")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root)
    flip_tta = not args.no_flip_tta

    # Setup Device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    # Load Model
    print(f"[Info] Loading PyTorch model from {args.checkpoint} on {device}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("threshold", checkpoint.get("optimal_threshold", 0.5)))
        print(f"[Info] Auto-loaded EER threshold: {threshold:.6f}")

    # Initialize YuNet & Landmarker (Running strictly on CPU as Pi 4 does)
    print("[Info] Initializing YuNet CPU Face Detector...")
    yunet_detector = OpenCVYuNetLandmarker(score_threshold=0.5)
    
    # We use OpenCV backend for landmarker to mimic Pi 4 if it's the default,
    # but the paper states MediaPipe 468 landmarks are used on Pi 4.
    # On Pi4 `process_video_pipeline` calls `ensure_face_landmarker_model` if backend is mediapipe or auto!
    # Let's use auto which downloads task and runs mediapipe CPU.
    lm_model_path = ensure_face_landmarker_model("models/face_landmarker.task", download=True)
    landmarker = FaceLandmarkerExtractor(
        lm_model_path,
        running_mode="image",
        min_confidence=0.5,
        backend="auto",
    )

    # Load Manifest
    if str(args.manifest).endswith(".txt"):
        print(f"[Info] Detected .txt manifest. Parsing as Celeb-DF List_of_testing_videos format...")
        records = []
        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    # e.g., "1 Celeb-synthesis/id0_id16_0000.mp4"
                    # "1" is FAKE, "0" or "0" might be REAL. But wait, in Celeb-DF txt:
                    # 1 means FAKE, 0 means REAL, usually, or vice-versa? Wait!
                    # In Celeb-DF: 1 means Real? Wait, NO. Usually Synthesis is fake.
                    # Let's map it explicitly by folder name to be safe.
                    vid_path = parts[1]
                    if "synthesis" in vid_path.lower() or "fake" in vid_path.lower():
                        lbl = 1
                    else:
                        lbl = 0
                    rec = VideoRecord(
                        dataset="celeb-df-v2",
                        split="test",
                        video_id=vid_path,
                        label=lbl,
                        method="unknown",
                        source_video="unknown"
                    )
                    records.append(rec)
    else:
        records = load_manifest(args.manifest)
        
    print(f"[Info] Loaded manifest with {len(records)} videos.")
    
    all_results: list[dict[str, object]] = []
    errors: list[str] = []
    total_start = time.perf_counter()

    for idx, record in enumerate(records):
        video_path = video_root / record.video_id
        if not video_path.is_file():
            # Sometimes manifest holds paths, check if video_id is a path
            video_path = video_root / record.video_id
            if not video_path.is_file():
                errors.append(f"{record.video_id}: File not found")
                print(f"[{idx+1:3d}/{len(records)}] [ERROR] File not found: {video_path}")
                continue

        try:
            rep = process_video_pipeline(
                video_path=video_path,
                landmarker=landmarker,
                model=model,
                onnx_session=None,
                mtcnn_detector=None,
                yunet_detector=yunet_detector,
                num_frames=args.num_frames,
                texture_frames=args.texture_frames,
                target_fps=args.target_fps,
                image_size=args.image_size,
                clips=args.clips_per_video,
                flip_tta=flip_tta,
                aggregation=args.aggregation,
                top_k=args.top_k,
                device=device,
                no_landmarks=False,
            )
            
            score = rep["detection"]["fake_probability"]
            prediction = "FAKE" if score >= threshold else "REAL"
            
            result = {
                "video_id": record.video_id,
                "label": record.label,
                "method": record.method,
                "dataset": record.dataset,
                "clip_scores": rep["detection"]["clip_scores"],
                "clip_count": len(rep["detection"]["clip_scores"]),
                "score": score,
                "prediction": prediction,
                "timings_ms": rep["timings_ms"]
            }
            all_results.append(result)

            label_str = "FAKE" if record.label == 1 else "REAL"
            correct = "✓" if (prediction == label_str) else "✗"
            print(
                f"[{idx+1:3d}/{len(records)}] {correct} [{prediction:4s}] "
                f"{record.video_id:<30} | Score: {score*100:6.2f}% "
                f"| {result['timings_ms']['total_end_to_end_ms']:7.1f}ms",
                flush=True,
            )

        except Exception as e:
            errors.append(f"{record.video_id}: {e}")
            print(f"[{idx+1:3d}/{len(records)}] [ERROR] {record.video_id}: {e}", flush=True)

    total_elapsed_s = time.perf_counter() - total_start
    landmarker.close()

    if not all_results:
        print("No videos processed successfully.", flush=True)
        sys.exit(1)

    # Compute metrics exactly like Server baseline
    labels = np.asarray([r["label"] for r in all_results], dtype=np.int64)
    scores = np.asarray([r["score"] for r in all_results], dtype=np.float64)
    metrics = compute_metrics(labels, scores, threshold)

    total_latencies = [r["timings_ms"]["total_end_to_end_ms"] for r in all_results]

    # Print comprehensive report
    report_lines = []
    def pline(s: str = "") -> None:
        report_lines.append(s)
        print(s, flush=True)

    pline()
    pline("=" * 80)
    pline("  TEXTURESBI EVALUATION ON SERVER (YuNet + PyTorch Raw Video)")
    pline("=" * 80)
    pline(f"Manifest           : {args.manifest}")
    pline(f"Checkpoint         : {args.checkpoint}")
    pline(f"Inference Device   : {device}")
    pline(f"Videos Processed   : {len(all_results)} / {len(records)} (Errors: {len(errors)})")
    pline(f"Protocol           : {args.clips_per_video} clips × {args.texture_frames}f, Flip-TTA={flip_tta}, Agg={args.aggregation}")
    pline(f"Threshold          : {threshold:.6f}")
    pline(f"Total Wall Time    : {total_elapsed_s:.1f}s ({total_elapsed_s/60:.1f} min)")
    pline()
    pline("-" * 80)
    pline("  RANKING METRICS (threshold independent)")
    pline("-" * 80)
    pline(f"  AUC-ROC                    : {metrics['auc']*100:>7.2f} %")
    pline(f"  Equal Error Rate (EER)     : {metrics['eer']*100:>7.2f} %")
    pline()
    pline("-" * 80)
    pline(f"  OPERATING POINT (threshold = {threshold:.6f})")
    pline("-" * 80)
    pline(f"  Accuracy                   : {metrics['accuracy']*100:>7.2f} %")
    pline(f"  Balanced Accuracy          : {metrics['balanced_accuracy']*100:>7.2f} %")
    pline(f"  F1 Fake                    : {metrics['f1_fake']*100:>7.2f} %")
    pline(f"  F1 Real                    : {metrics['f1_real']*100:>7.2f} %")
    pline()

    # Save metrics.json
    output_metrics = {
        "metrics": metrics,
        "protocol": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "environment": platform.platform(),
            "inference": {
                "num_frames": args.num_frames,
                "texture_frames": args.texture_frames,
                "clips_per_video": args.clips_per_video,
                "aggregation": args.aggregation,
                "top_k": args.top_k,
                "texture_flip_tta": flip_tta,
            },
            "threshold": threshold,
            "total_wall_time_s": round(total_elapsed_s, 1),
            "latency_summary": {
                "per_video_mean_ms": round(statistics.mean(total_latencies), 1),
                "per_video_p50_ms": round(percentile(total_latencies, 0.50), 1),
                "per_video_p95_ms": round(percentile(total_latencies, 0.95), 1),
            },
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=2, ensure_ascii=False)

    # Save predictions.csv
    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["score", "label", "video_id", "method", "dataset", "clip_count"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "score": r["score"],
                "label": r["label"],
                "video_id": r["video_id"],
                "method": r["method"],
                "dataset": r["dataset"],
                "clip_count": r["clip_count"],
            })

    with open(output_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n[Done] Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
