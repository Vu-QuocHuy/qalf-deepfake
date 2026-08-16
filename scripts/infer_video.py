#!/usr/bin/env python3
"""End-to-end video deepfake detection and latency profiling script for Raspberry Pi & Edge devices."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import IMAGE_MEAN, IMAGE_STD, _aligned_full_face
from qalf.data.landmarks import FaceLandmarkerExtractor, ensure_face_landmarker_model
from qalf.models import build_model_from_checkpoint


def sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    """Sample evenly spaced frame indices across video length."""
    if total_frames <= num_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()


def extract_video_frames(
    video_path: str | Path,
    indices: list[int],
) -> tuple[list[np.ndarray], float]:
    """Read specific frames from video and measure read time."""
    start_time = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    idx_set = set(indices)
    current_idx = 0

    max_target = max(indices) if indices else 0
    # Map index to frame
    frame_dict: dict[int, np.ndarray] = {}

    while cap.isOpened() and current_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in idx_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_dict[current_idx] = frame_rgb
        current_idx += 1
    cap.release()

    read_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Ensure all target indices are retrieved in order
    ordered_frames: list[np.ndarray] = []
    last_valid = None
    for idx in indices:
        if idx in frame_dict:
            last_valid = frame_dict[idx]
            ordered_frames.append(frame_dict[idx])
        elif last_valid is not None:
            ordered_frames.append(last_valid)

    return ordered_frames, read_elapsed_ms


def process_video_pipeline(
    video_path: str | Path,
    landmarker: FaceLandmarkerExtractor,
    model: torch.nn.Module | None,
    onnx_session: object | None,
    num_frames: int = 32,
    texture_frames: int = 8,
    image_size: int = 160,
    clips: int = 1,
    device: torch.device = torch.device("cpu"),
) -> dict[str, object]:
    """Run full pipeline: Decode -> Landmark -> Align -> Model Inference -> Output."""
    timing: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    # 1. Video Meta & Decoding
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    target_indices = sample_frame_indices(total_video_frames, max(num_frames, texture_frames))
    raw_frames, decode_ms = extract_video_frames(video_path, target_indices)
    timing["decode_ms"] = decode_ms

    if not raw_frames:
        raise RuntimeError(f"No frames could be extracted from: {video_path}")

    # 2. Facial Landmark Extraction
    lm_start = time.perf_counter()
    landmarks_list: list[np.ndarray | None] = []
    for frame in raw_frames:
        pts = landmarker.process(frame)
        landmarks_list.append(pts)
    timing["landmark_ms"] = (time.perf_counter() - lm_start) * 1000.0

    # 3. Face Alignment and Tensor Construction
    align_start = time.perf_counter()
    aligned_crops: list[np.ndarray] = []
    for frame, lms in zip(raw_frames, landmarks_list):
        has_lms = lms is not None
        crop, _ = _aligned_full_face(
            frame,
            lms if has_lms else np.zeros((468, 3), dtype=np.float32),
            has_lms,
            image_size,
        )
        # Normalize with ImageNet mean and std
        crop_norm = (crop.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        # Transpose HWC to CHW
        crop_chw = np.transpose(crop_norm, (2, 0, 1))
        aligned_crops.append(crop_chw)

    total_extracted = len(aligned_crops)
    # Form clips
    clip_tensors: list[np.ndarray] = []
    if clips == 1:
        start_idx = max(0, (total_extracted - texture_frames) // 2)
        clip_data = aligned_crops[start_idx : start_idx + texture_frames]
        # Pad if short
        while len(clip_data) < texture_frames:
            clip_data.append(clip_data[-1] if clip_data else np.zeros((3, image_size, image_size)))
        clip_tensors.append(np.stack(clip_data, axis=0))
    else:
        starts = np.rint(np.linspace(0, max(0, total_extracted - texture_frames), clips)).astype(int)
        for s in starts:
            clip_data = aligned_crops[s : s + texture_frames]
            while len(clip_data) < texture_frames:
                clip_data.append(clip_data[-1] if clip_data else np.zeros((3, image_size, image_size)))
            clip_tensors.append(np.stack(clip_data, axis=0))

    # (B, T, C, H, W)
    batch_array = np.stack(clip_tensors, axis=0).astype(np.float32)
    timing["align_and_preprocess_ms"] = (time.perf_counter() - align_start) * 1000.0

    # 4. Neural Network Inference
    infer_start = time.perf_counter()
    if onnx_session is not None:
        input_name = onnx_session.get_inputs()[0].name
        logits = onnx_session.run(None, {input_name: batch_array})[0]
        if isinstance(logits, np.ndarray):
            scores = 1.0 / (1.0 + np.exp(-logits.squeeze(-1) if logits.ndim > 1 else logits))
    elif model is not None:
        batch_tensor = torch.from_numpy(batch_array).to(device)
        with torch.inference_mode():
            out = model({"texture": batch_tensor})
            logits = out["logit"]
            scores = torch.sigmoid(logits).cpu().numpy()
    else:
        raise RuntimeError("Neither PyTorch model nor ONNX session was provided")

    timing["inference_ms"] = (time.perf_counter() - infer_start) * 1000.0

    scores_list = [float(s) for s in (scores if isinstance(scores, np.ndarray) and scores.ndim > 0 else [scores])]
    mean_score = float(np.mean(scores_list))

    total_pipeline_time_s = time.perf_counter() - pipeline_start
    timing["total_pipeline_ms"] = total_pipeline_time_s * 1000.0
    effective_fps = len(raw_frames) / total_pipeline_time_s if total_pipeline_time_s > 0 else 0.0

    return {
        "video_info": {
            "path": str(video_path),
            "total_video_frames": total_video_frames,
            "sampled_frames": len(raw_frames),
            "resolution": f"{width}x{height}",
            "original_fps": fps,
        },
        "timings_ms": {
            "1_video_decode_ms": round(timing["decode_ms"], 2),
            "2_landmark_extraction_ms": round(timing["landmark_ms"], 2),
            "3_face_align_and_preprocess_ms": round(timing["align_and_preprocess_ms"], 2),
            "4_model_forward_ms": round(timing["inference_ms"], 2),
            "total_end_to_end_ms": round(timing["total_pipeline_ms"], 2),
        },
        "performance": {
            "end_to_end_fps": round(effective_fps, 2),
            "model_only_fps": round((len(clip_tensors) * texture_frames) / (timing["inference_ms"] / 1000.0), 2) if timing["inference_ms"] > 0 else 0,
        },
        "detection": {
            "clip_scores": [round(s, 4) for s in scores_list],
            "fake_probability": round(mean_score, 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Video Deepfake Detection & Profiler for Pi 4")
    parser.add_argument("--video", required=True, help="Path to input video file (.mp4, .avi, etc.)")
    parser.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", default=None, help="Path to ONNX model (.onnx)")
    parser.add_argument("--model-task", default="models/face_landmarker.task", help="MediaPipe face landmarker model")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames sampled from video")
    parser.add_argument("--texture-frames", type=int, default=8, help="Frames per clip fed to model")
    parser.add_argument("--clips", type=int, default=1, help="Number of clips per video (default 1 for fast edge inference)")
    parser.add_argument("--image-size", type=int, default=160, help="Input resolution (160x160)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for Real vs Fake")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Number of CPU threads to use on Pi 4")
    parser.add_argument("--output-json", default=None, help="Save timing and prediction report to JSON")
    args = parser.parse_args()

    if not args.checkpoint and not args.onnx:
        parser.error("Must provide either --checkpoint (.pt) or --onnx (.onnx)")

    # 1. Setup Face Landmarker
    lm_model_path = ensure_face_landmarker_model(args.model_task, download=True)
    landmarker = FaceLandmarkerExtractor(lm_model_path, running_mode="image", min_confidence=0.5)

    # 2. Setup Model (PyTorch or ONNX)
    model = None
    onnx_session = None
    threshold = args.threshold

    if args.onnx:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = args.cpu_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_session = ort.InferenceSession(args.onnx, sess_options, providers=["CPUExecutionProvider"])
        if threshold is None:
            threshold = 0.5
    else:
        torch.set_num_threads(args.cpu_threads)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model = build_model_from_checkpoint(checkpoint).eval()
        if threshold is None:
            threshold = float(checkpoint.get("threshold", 0.5))

    # 3. Execute Pipeline
    report = process_video_pipeline(
        video_path=args.video,
        landmarker=landmarker,
        model=model,
        onnx_session=onnx_session,
        num_frames=args.num_frames,
        texture_frames=args.texture_frames,
        image_size=args.image_size,
        clips=args.clips,
        device=torch.device("cpu"),
    )
    landmarker.close()

    # 4. Add Classification Result
    fake_prob = report["detection"]["fake_probability"]
    prediction = "FAKE" if fake_prob >= threshold else "REAL"
    report["detection"]["threshold"] = round(threshold, 4)
    report["detection"]["prediction"] = prediction

    # Print summary
    print("\n" + "=" * 60)
    print("      QALF VIDEO DEEPFAKE DETECTION REPORT (PI 4 / EDGE)")
    print("=" * 60)
    print(f"Input Video      : {args.video}")
    print(f"Resolution / FPS : {report['video_info']['resolution']} @ {report['video_info']['original_fps']:.1f} FPS")
    print(f"Frames Sampled   : {report['video_info']['sampled_frames']} frames")
    print("-" * 60)
    print("TIMING BREAKDOWN:")
    for k, v in report["timings_ms"].items():
        print(f"  - {k:<32}: {v:>8.2f} ms")
    print(f"Throughput       : {report['performance']['end_to_end_fps']} FPS (End-to-End)")
    print("-" * 60)
    print(f"Fake Probability : {fake_prob * 100:.2f}% (Threshold: {threshold:.4f})")
    print(f"PREDICTION       : >>> {prediction} <<<")
    print("=" * 60 + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
