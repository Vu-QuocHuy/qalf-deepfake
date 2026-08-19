#!/usr/bin/env python3
"""Evaluate TextureSBI on Raspberry Pi 4 using pre-extracted frames and landmarks.

This script replicates the EXACT same evaluation pipeline as the server's
``scripts/evaluate.py`` + ``QALFVideoDataset.__getitem__()`` + ``engine.predict()``
+ ``engine.aggregate_predictions()`` + ``metrics.compute_metrics()``, but uses
ONNX Runtime instead of PyTorch for the neural forward pass.

It reads pre-extracted data produced by:
  1. ``scripts/extract_frames.py``  → MTCNN 35%-margin 256×256 JPEG crops
  2. ``scripts/extract_landmarks.py`` → MediaPipe 468-point landmark .npz caches

This guarantees 100% identical preprocessing with the server pipeline,
eliminating all five sources of divergence documented in the implementation plan.
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
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import IMAGE_MEAN, IMAGE_STD, _aligned_full_face, _clip_indices
from qalf.data.manifest import VideoRecord, load_manifest
from qalf.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_resource_usage() -> dict[str, object]:
    """Collect CPU/RAM/Temperature metrics on ARM Linux."""
    info: dict[str, object] = {}
    try:
        import psutil
        mem = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        info["ram_used_mb"] = round(proc.memory_info().rss / 1024 / 1024, 1)
        info["ram_total_mb"] = round(mem.total / 1024 / 1024, 1)
        info["ram_percent"] = round(proc.memory_info().rss / mem.total * 100, 1)
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass
    try:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            info["cpu_temp_c"] = round(int(temp_path.read_text().strip()) / 1000.0, 1)
    except Exception:
        pass
    return info


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


# ---------------------------------------------------------------------------
# Core: process one video exactly like QALFVideoDataset.__getitem__()
# ---------------------------------------------------------------------------

def process_video_from_manifest(
    record: VideoRecord,
    frame_root: Path,
    landmark_root: Path,
    num_frames: int,
    texture_frames: int,
    image_size: int,
    clips_per_video: int,
    flip_tta: bool,
    onnx_session: object,
    no_landmarks: bool = False,
) -> dict[str, object]:
    """Process one video using pre-extracted frames + landmarks, matching server exactly."""
    timing: dict[str, float] = {}
    video_start = time.perf_counter()

    # 1. Load landmark cache (same as QALFVideoDataset.__getitem__ line 294-296)
    lm_start = time.perf_counter()
    landmark_file = landmark_root / str(record.landmark_path)
    if not landmark_file.is_file() and (landmark_root / "landmarks" / str(record.landmark_path)).is_file():
        landmark_file = landmark_root / "landmarks" / str(record.landmark_path)
    with np.load(landmark_file) as cache:
        landmarks = cache["landmarks"].copy()
        detected = cache["detected"].copy()

    if len(landmarks) != len(record.frames) or len(detected) != len(record.frames):
        raise ValueError(
            f"{record.video_id}: landmark cache length ({len(landmarks)}) "
            f"does not match manifest frames ({len(record.frames)})"
        )
    timing["1_landmark_load_ms"] = (time.perf_counter() - lm_start) * 1000.0

    # 2. For each clip, compute indices and read+align frames
    # (same as QALFVideoDataset.__getitem__ lines 301-327)
    clip_scores_list: list[float] = []
    all_clip_data: list[np.ndarray] = []  # Each: (texture_frames, 3, H, W)

    read_start = time.perf_counter()
    for clip_idx in range(clips_per_video):
        clip = _clip_indices(
            len(record.frames),
            num_frames,
            training=False,
            clip_index=clip_idx,
            clips_per_video=clips_per_video,
        )
        texture_positions = np.rint(
            np.linspace(0, len(clip) - 1, texture_frames)
        ).astype(np.int64)

        texture_tensors: list[np.ndarray] = []
        for position in texture_positions:
            source_index = int(clip[position])
            frame_path = frame_root / record.frames[source_index]
            if not frame_path.is_file() and (frame_root / "frames" / record.frames[source_index]).is_file():
                frame_path = frame_root / "frames" / record.frames[source_index]
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Cannot read frame: {frame_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            canonical, _ = _aligned_full_face(
                image_rgb,
                landmarks[source_index],
                False if no_landmarks else bool(detected[source_index]),
                image_size,
            )
            normalized = canonical.astype(np.float32) / 255.0
            normalized = (normalized - IMAGE_MEAN) / IMAGE_STD
            texture_tensors.append(normalized.transpose(2, 0, 1))

        clip_array = np.stack(texture_tensors, axis=0).astype(np.float32)
        all_clip_data.append(clip_array)

    timing["2_frame_read_and_align_ms"] = (time.perf_counter() - read_start) * 1000.0

    # 3. Batch inference (same as engine.predict with flip_tta)
    infer_start = time.perf_counter()
    input_name = onnx_session.get_inputs()[0].name

    for clip_array in all_clip_data:
        # Shape: (1, texture_frames, 3, H, W)
        batch = clip_array[np.newaxis, ...]
        logits = onnx_session.run(None, {input_name: batch})[0]
        score = float(1.0 / (1.0 + np.exp(-float(logits.squeeze()))))

        if flip_tta:
            # Horizontal flip: flip along width axis (last dimension)
            flipped = batch[:, :, :, :, ::-1].copy()
            flipped_logits = onnx_session.run(None, {input_name: flipped})[0]
            flipped_score = float(1.0 / (1.0 + np.exp(-float(flipped_logits.squeeze()))))
            score = 0.5 * (score + flipped_score)

        clip_scores_list.append(score)

    timing["3_model_forward_ms"] = (time.perf_counter() - infer_start) * 1000.0
    timing["total_end_to_end_ms"] = (time.perf_counter() - video_start) * 1000.0

    return {
        "video_id": record.video_id,
        "label": record.label,
        "method": record.method,
        "dataset": record.dataset,
        "clip_scores": clip_scores_list,
        "clip_count": len(clip_scores_list),
        "timings_ms": timing,
    }


# ---------------------------------------------------------------------------
# Aggregation (exact replica of engine.aggregate_predictions)
# ---------------------------------------------------------------------------

def aggregate_video_scores(
    clip_scores: list[float],
    method: str = "mean",
    top_k: int = 1,
) -> float:
    if method == "topk":
        selected = sorted(clip_scores, reverse=True)[:min(top_k, len(clip_scores))]
        return float(np.mean(selected))
    elif method == "median":
        return float(np.median(clip_scores))
    else:  # mean
        return float(np.mean(clip_scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TextureSBI on Pi 4 using pre-extracted data (exact server match)"
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to landmark manifest JSONL")
    parser.add_argument("--frame-root", required=True,
                        help="Root directory for pre-extracted MTCNN 256x256 frame JPEGs")
    parser.add_argument("--landmark-root", required=True,
                        help="Root directory for MediaPipe 468 landmark .npz caches")
    parser.add_argument("--onnx", required=True,
                        help="Path to ONNX model (.onnx)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--num-frames", type=int, default=32,
                        help="Temporal clip window size (default 32, matching training)")
    parser.add_argument("--texture-frames", type=int, default=8,
                        help="Number of texture frames per clip (default 8)")
    parser.add_argument("--clips-per-video", type=int, default=3,
                        help="Number of clips per video (default 3)")
    parser.add_argument("--aggregation", choices=["mean", "median", "topk"], default="mean")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--texture-flip-tta", action="store_true", default=True,
                        help="Enable horizontal flip test-time augmentation (default True)")
    parser.add_argument("--no-flip-tta", action="store_true",
                        help="Disable flip TTA")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Ablation: Disable landmark alignment, only resize face crop")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (auto-loaded from ONNX metadata if not set)")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    flip_tta = not args.no_flip_tta

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup ONNX Runtime
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.cpu_threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_session = ort.InferenceSession(
        args.onnx, sess_options, providers=["CPUExecutionProvider"]
    )

    # Auto-load threshold from companion JSON
    threshold = args.threshold
    if threshold is None:
        json_meta_path = Path(args.onnx).with_suffix(".json")
        if json_meta_path.is_file():
            with open(json_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            threshold = float(meta.get("optimal_threshold", 0.5))
            print(f"[Info] Auto-loaded threshold: {threshold:.6f} from {json_meta_path.name}", flush=True)
        else:
            threshold = 0.5
            print("[Warning] No threshold metadata found, using 0.5", flush=True)

    # Load manifest
    frame_root = Path(args.frame_root)
    landmark_root = Path(args.landmark_root)
    records = load_manifest(args.manifest)
    print(f"[Info] Loaded manifest with {len(records)} videos "
          f"(Real: {sum(1 for r in records if r.label == 0)}, "
          f"Fake: {sum(1 for r in records if r.label == 1)})", flush=True)
    print(f"[Info] Protocol: {args.clips_per_video} clips x {args.texture_frames}f "
          f"across {args.num_frames}f window, Flip-TTA: {flip_tta}, "
          f"Aggregation: {args.aggregation}", flush=True)

    # Collect initial resource usage
    hw_before = _read_resource_usage()

    # Process all videos
    all_results: list[dict[str, object]] = []
    errors: list[str] = []
    total_start = time.perf_counter()

    for idx, record in enumerate(records):
        try:
            result = process_video_from_manifest(
                record=record,
                frame_root=frame_root,
                landmark_root=landmark_root,
                num_frames=args.num_frames,
                texture_frames=args.texture_frames,
                image_size=args.image_size,
                clips_per_video=args.clips_per_video,
                flip_tta=flip_tta,
                onnx_session=onnx_session,
                no_landmarks=args.no_landmarks,
            )
            agg_score = aggregate_video_scores(
                result["clip_scores"],
                method=args.aggregation,
                top_k=args.top_k,
            )
            result["score"] = agg_score
            prediction = "FAKE" if agg_score >= threshold else "REAL"
            result["prediction"] = prediction
            all_results.append(result)

            label_str = "FAKE" if record.label == 1 else "REAL"
            correct = "✓" if (prediction == label_str) else "✗"
            print(
                f"[{idx+1:3d}/{len(records)}] {correct} [{prediction:4s}] "
                f"{record.video_id:<30} | Score: {agg_score*100:6.2f}% "
                f"| {result['timings_ms']['total_end_to_end_ms']:7.1f}ms",
                flush=True,
            )
        except Exception as e:
            errors.append(f"{record.video_id}: {e}")
            print(f"[{idx+1:3d}/{len(records)}] [ERROR] {record.video_id}: {e}", flush=True)

    total_elapsed_s = time.perf_counter() - total_start
    hw_after = _read_resource_usage()

    if not all_results:
        print("No videos processed successfully.", flush=True)
        return

    # Compute metrics using the EXACT same function as server
    labels = np.asarray([r["label"] for r in all_results], dtype=np.int64)
    scores = np.asarray([r["score"] for r in all_results], dtype=np.float64)
    metrics = compute_metrics(labels, scores, threshold)

    # Compute latency statistics
    total_latencies = [r["timings_ms"]["total_end_to_end_ms"] for r in all_results]
    lm_latencies = [r["timings_ms"]["1_landmark_load_ms"] for r in all_results]
    read_latencies = [r["timings_ms"]["2_frame_read_and_align_ms"] for r in all_results]
    model_latencies = [r["timings_ms"]["3_model_forward_ms"] for r in all_results]

    def stats_str(data: list[float]) -> str:
        m = statistics.mean(data)
        s = statistics.stdev(data) if len(data) > 1 else 0.0
        p50 = percentile(data, 0.50)
        p95 = percentile(data, 0.95)
        return f"{m:>8.2f} ± {s:<7.2f} | {p50:>8.2f} | {p95:>8.2f}"

    # Print comprehensive report
    report_lines = []
    def pline(s: str = "") -> None:
        report_lines.append(s)
        print(s, flush=True)

    pline()
    pline("=" * 80)
    pline("  TEXTURESBI EVALUATION ON RASPBERRY PI 4 (PRE-EXTRACTED DATA)")
    pline("=" * 80)
    pline(f"Manifest           : {args.manifest}")
    pline(f"ONNX Model         : {args.onnx}")
    pline(f"Videos Processed   : {len(all_results)} / {len(records)} "
          f"(Errors: {len(errors)})")
    pline(f"Protocol           : {args.clips_per_video} clips × {args.texture_frames}f "
          f"across {args.num_frames}f, Flip-TTA={flip_tta}, Agg={args.aggregation}")
    pline(f"Threshold          : {threshold:.6f}")
    pline(f"Total Wall Time    : {total_elapsed_s:.1f}s "
          f"({total_elapsed_s/60:.1f} min)")
    pline()
    pline("-" * 80)
    pline("  RANKING METRICS (threshold independent)")
    pline("-" * 80)
    pline(f"  AUC-ROC                    : {metrics['auc']*100:>7.2f} %")
    pline(f"  Average Precision (PR-AUC) : {metrics['average_precision']*100:>7.2f} %")
    pline(f"  Equal Error Rate (EER)     : {metrics['eer']*100:>7.2f} %")
    pline()
    pline("-" * 80)
    pline(f"  OPERATING POINT (threshold = {threshold:.6f})")
    pline("-" * 80)
    pline(f"  Accuracy                   : {metrics['accuracy']*100:>7.2f} %")
    pline(f"  Balanced Accuracy          : {metrics['balanced_accuracy']*100:>7.2f} %")
    pline(f"  F1 Fake                    : {metrics['f1_fake']*100:>7.2f} %")
    pline(f"  F1 Real                    : {metrics['f1_real']*100:>7.2f} %")
    pline(f"  F1 Macro                   : {metrics['f1_macro']*100:>7.2f} %")
    pline(f"  Precision Fake             : {metrics['precision_fake']*100:>7.2f} %")
    pline(f"  Precision Real             : {metrics['precision_real']*100:>7.2f} %")
    pline(f"  Recall Fake (Sensitivity)  : {metrics['recall_fake']*100:>7.2f} %")
    pline(f"  Recall Real (Specificity)  : {metrics['recall_real']*100:>7.2f} %")
    pline(f"  APCER                      : {metrics['apcer']*100:>7.2f} %")
    pline(f"  BPCER                      : {metrics['bpcer']*100:>7.2f} %")
    pline(f"  ACER                       : {metrics['acer']*100:>7.2f} %")
    pline()
    pline("-" * 80)
    pline("  CONFUSION MATRIX (real=0, fake=1)")
    pline("-" * 80)
    pline(f"  TN (real→real)  : {metrics['true_negative']:>5d}    "
          f"FP (real→fake)  : {metrics['false_positive']:>5d}    "
          f"| Real count  : {metrics['real_count']:>5d}")
    pline(f"  FN (fake→real)  : {metrics['false_negative']:>5d}    "
          f"TP (fake→fake)  : {metrics['true_positive']:>5d}    "
          f"| Fake count  : {metrics['fake_count']:>5d}")
    pline()
    pline("-" * 80)
    pline("  HARDWARE PROFILING (per-video latency)")
    pline("-" * 80)
    pline(f"{'Stage':<36} | {'Mean ± Std (ms)':<18} | {'P50 (ms)':<9} | {'P95 (ms)':<9}")
    pline("-" * 80)
    pline(f"{'1. Landmark Cache Load':<36} | {stats_str(lm_latencies)}")
    pline(f"{'2. Frame Read & Affine Align':<36} | {stats_str(read_latencies)}")
    pline(f"{'3. ONNX Neural Forward':<36} | {stats_str(model_latencies)}")
    pline("-" * 80)
    pline(f"{'TOTAL PER VIDEO':<36} | {stats_str(total_latencies)}")
    pline("-" * 80)
    throughput_fps = len(all_results) * args.texture_frames * args.clips_per_video * (2 if flip_tta else 1) / (sum(model_latencies) / 1000.0)
    pline(f"  Neural Throughput (Model-Only)  : {throughput_fps:.2f} FPS")
    pline(f"  End-to-End Throughput           : {statistics.mean([1000.0/t for t in total_latencies]):.2f} videos/sec")
    if hw_after:
        pline()
        pline("-" * 80)
        pline("  HARDWARE RESOURCE USAGE")
        pline("-" * 80)
        for k, v in hw_after.items():
            pline(f"  {k:<28} : {v}")
    pline()
    pline("=" * 80)

    # Save metrics.json (same format as server)
    output_metrics = {
        "metrics": metrics,
        "protocol": {
            "manifest": str(args.manifest),
            "onnx_model": str(args.onnx),
            "device": "Raspberry Pi 4 Model B (ARM Cortex-A72 @ 1.5GHz, 4GB RAM)",
            "inference": {
                "num_frames": args.num_frames,
                "texture_frames": args.texture_frames,
                "clips_per_video": args.clips_per_video,
                "aggregation": args.aggregation,
                "top_k": args.top_k,
                "texture_flip_tta": flip_tta,
            },
            "threshold": threshold,
            "hardware": hw_after,
            "total_wall_time_s": round(total_elapsed_s, 1),
            "latency_summary": {
                "per_video_mean_ms": round(statistics.mean(total_latencies), 1),
                "per_video_p50_ms": round(percentile(total_latencies, 0.50), 1),
                "per_video_p95_ms": round(percentile(total_latencies, 0.95), 1),
                "model_throughput_fps": round(throughput_fps, 2),
            },
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=2, ensure_ascii=False)

    # Save predictions.csv (same format as server)
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

    # Save full report
    with open(output_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n[Done] Results saved to: {output_dir}", flush=True)
    print(f"  - metrics.json", flush=True)
    print(f"  - predictions.csv", flush=True)
    print(f"  - metrics.txt", flush=True)


if __name__ == "__main__":
    main()
