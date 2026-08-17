#!/usr/bin/env python3
"""End-to-end video deepfake detection and latency profiling script for Raspberry Pi & Edge devices."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import IMAGE_MEAN, IMAGE_STD, _aligned_full_face
from qalf.data.landmarks import FaceLandmarkerExtractor, ensure_face_landmarker_model
from qalf.models import build_model_from_checkpoint


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def sample_video_clip_windows(
    total_frames: int,
    source_fps: float,
    target_fps: float = 10.0,
    texture_frames: int = 8,
    clips: int = 1,
) -> tuple[list[int], list[list[int]]]:
    """
    Sample temporally coherent windows at target_fps (10.0 FPS) matching the training protocol.
    Returns:
      all_indices: unique sorted frame indices to decode from the video.
      clip_index_groups: list of index lists (one per clip) mapping to target frame sequence.
    """
    stride = max(1, int(round(source_fps / target_fps)))
    clip_span = (texture_frames - 1) * stride + 1

    if total_frames <= clip_span:
        starts = [0] * clips
    elif clips == 1:
        starts = [(total_frames - clip_span) // 2]
    else:
        starts = np.linspace(0, total_frames - clip_span, clips, dtype=int).tolist()

    clip_index_groups: list[list[int]] = []
    all_needed_set: set[int] = set()

    for s in starts:
        group = []
        for i in range(texture_frames):
            frame_idx = min(total_frames - 1, s + i * stride)
            group.append(frame_idx)
            all_needed_set.add(frame_idx)
        clip_index_groups.append(group)

    all_indices = sorted(list(all_needed_set))
    return all_indices, clip_index_groups


def extract_video_frames(
    video_path: str | Path,
    indices: list[int],
) -> tuple[dict[int, np.ndarray], float]:
    """Read specific frames from video and measure read time."""
    start_time = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx_set = set(indices)
    current_idx = 0
    max_target = max(indices) if indices else 0
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

    # Fill any missing frame with nearest available
    if frame_dict:
        fallback = next(iter(frame_dict.values()))
        for idx in indices:
            if idx not in frame_dict:
                frame_dict[idx] = fallback

    return frame_dict, read_elapsed_ms


def process_video_pipeline(
    video_path: str | Path,
    landmarker: FaceLandmarkerExtractor,
    model: torch.nn.Module | None,
    onnx_session: object | None,
    texture_frames: int = 8,
    target_fps: float = 10.0,
    image_size: int = 160,
    clips: int = 1,
    flip_tta: bool = False,
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

    all_indices, clip_index_groups = sample_video_clip_windows(
        total_frames=total_video_frames,
        source_fps=fps,
        target_fps=target_fps,
        texture_frames=texture_frames,
        clips=clips,
    )

    frame_dict, decode_ms = extract_video_frames(video_path, all_indices)
    timing["1_video_decode_ms"] = decode_ms

    if not frame_dict:
        raise RuntimeError(f"No frames could be extracted from: {video_path}")

    # 2. Facial Landmark Extraction
    lm_start = time.perf_counter()
    aligned_map: dict[int, np.ndarray] = {}

    for idx in all_indices:
        frame = frame_dict[idx]
        pts = landmarker.process(frame)
        has_lms = pts is not None
        crop, _ = _aligned_full_face(
            frame,
            pts if has_lms else np.zeros((468, 3), dtype=np.float32),
            has_lms,
            image_size,
        )
        crop_norm = (crop.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        crop_chw = np.transpose(crop_norm, (2, 0, 1))
        aligned_map[idx] = crop_chw

    timing["2_landmark_extraction_ms"] = (time.perf_counter() - lm_start) * 1000.0

    # 3. Face Alignment and Tensor Construction
    align_start = time.perf_counter()
    clip_tensors: list[np.ndarray] = []
    for group in clip_index_groups:
        clip_crops = [aligned_map[idx] for idx in group]
        clip_tensors.append(np.stack(clip_crops, axis=0))

    batch_array = np.stack(clip_tensors, axis=0).astype(np.float32)

    if flip_tta:
        # Horizontally flip along width dimension (axis -1)
        flipped_batch = batch_array[:, :, :, :, ::-1].copy()
        eval_batch = np.concatenate([batch_array, flipped_batch], axis=0)
    else:
        eval_batch = batch_array

    timing["3_face_align_and_preprocess_ms"] = (time.perf_counter() - align_start) * 1000.0

    # 4. Neural Network Inference
    infer_start = time.perf_counter()
    if onnx_session is not None:
        input_name = onnx_session.get_inputs()[0].name
        logits = onnx_session.run(None, {input_name: eval_batch})[0]
        if isinstance(logits, np.ndarray):
            raw_scores = 1.0 / (1.0 + np.exp(-logits.squeeze(-1) if logits.ndim > 1 else logits))
    elif model is not None:
        batch_tensor = torch.from_numpy(eval_batch).to(device)
        with torch.inference_mode():
            out = model({"texture": batch_tensor})
            logits = out["logit"]
            raw_scores = torch.sigmoid(logits).cpu().numpy().squeeze(-1)
    else:
        raise RuntimeError("Neither PyTorch model nor ONNX session was provided")

    timing["4_model_forward_ms"] = (time.perf_counter() - infer_start) * 1000.0

    raw_scores_arr = np.atleast_1d(raw_scores)
    if flip_tta:
        orig_scores = raw_scores_arr[:clips]
        flip_scores = raw_scores_arr[clips:]
        clip_scores = 0.5 * (orig_scores + flip_scores)
    else:
        clip_scores = raw_scores_arr

    mean_score = float(np.mean(clip_scores))
    total_pipeline_time_s = time.perf_counter() - pipeline_start
    timing["total_end_to_end_ms"] = total_pipeline_time_s * 1000.0
    effective_fps = len(all_indices) / total_pipeline_time_s if total_pipeline_time_s > 0 else 0.0

    return {
        "video_info": {
            "path": str(video_path),
            "total_video_frames": total_video_frames,
            "sampled_frames": len(all_indices),
            "original_fps": fps,
            "resolution": f"{width}x{height}",
        },
        "timings_ms": timing,
        "performance": {
            "end_to_end_fps": round(effective_fps, 2),
            "model_forward_fps": round((clips * (2 if flip_tta else 1) * texture_frames) / (timing["4_model_forward_ms"] / 1000.0), 2),
        },
        "detection": {
            "fake_probability": round(mean_score, 4),
            "clip_scores": [round(float(s), 4) for s in clip_scores],
            "clips": clips,
            "flip_tta": flip_tta,
        },
    }


def run_synthetic_benchmark(
    model: torch.nn.Module | None,
    onnx_session: object | None,
    texture_frames: int = 8,
    image_size: int = 160,
    clips: int = 1,
    iterations: int = 100,
    warmup: int = 20,
) -> dict[str, object]:
    """Run model forward benchmark with synthetic in-memory tensor (0MB disk usage)."""
    dummy_input = np.random.randn(clips, texture_frames, 3, image_size, image_size).astype(np.float32)

    print(f"\n[Synthetic Benchmark] Warming up {warmup} iterations...")
    if onnx_session is not None:
        input_name = onnx_session.get_inputs()[0].name
        for _ in range(warmup):
            _ = onnx_session.run(None, {input_name: dummy_input})
    else:
        dummy_tensor = torch.from_numpy(dummy_input)
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model({"texture": dummy_tensor})

    print(f"[Synthetic Benchmark] Running {iterations} timed benchmark iterations...")
    timings: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        if onnx_session is not None:
            _ = onnx_session.run(None, {input_name: dummy_input})
        else:
            with torch.inference_mode():
                _ = model({"texture": dummy_tensor})
        timings.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(timings)
    p50_ms = percentile(timings, 0.50)
    p95_ms = percentile(timings, 0.95)
    fps = (clips * texture_frames) / (mean_ms / 1000.0)

    return {
        "engine": "ONNXRuntime" if onnx_session is not None else "PyTorch-CPU",
        "input_shape": list(dummy_input.shape),
        "iterations": iterations,
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_p50": round(p50_ms, 2),
        "latency_ms_p95": round(p95_ms, 2),
        "latency_ms_min": round(min(timings), 2),
        "latency_ms_max": round(max(timings), 2),
        "throughput_fps": round(fps, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Video Deepfake Detection & Profiler for Pi 4")
    parser.add_argument("--video", default=None, help="Path to single input video file (.mp4, .avi, etc.)")
    parser.add_argument("--video-dir", default=None, help="Path to directory containing subset of test videos")
    parser.add_argument("--synthetic", action="store_true", help="Run 0MB synthetic benchmark in memory without disk video")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations for synthetic benchmark")
    parser.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", default=None, help="Path to ONNX model (.onnx)")
    parser.add_argument("--model-task", default="models/face_landmarker.task", help="MediaPipe face landmarker model")
    parser.add_argument("--texture-frames", type=int, default=8, help="Frames per clip fed to model")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target sampling rate (default 10.0 FPS matching training)")
    parser.add_argument("--clips", type=int, default=1, help="Clips per video (1 for fast edge, 3 for paper protocol)")
    parser.add_argument("--flip-tta", action="store_true", help="Enable test-time horizontal flip augmentation")
    parser.add_argument("--image-size", type=int, default=160, help="Input resolution (160x160)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for Real vs Fake (auto-loaded if None)")
    parser.add_argument("--landmark-backend", default="auto", choices=["auto", "opencv", "mediapipe"], help="Face landmark backend (auto defaults to opencv on ARM/Pi 4)")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Number of CPU threads to use on Pi 4")
    parser.add_argument("--output-json", default=None, help="Save timing and prediction report to JSON")
    args = parser.parse_args()

    if not args.checkpoint and not args.onnx:
        parser.error("Must provide either --checkpoint (.pt) or --onnx (.onnx)")

    # 1. Setup Model (PyTorch or ONNX) and Auto-load Threshold
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
            json_meta_path = Path(args.onnx).with_suffix(".json")
            if json_meta_path.is_file():
                try:
                    with open(json_meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    threshold = float(meta.get("optimal_threshold", 0.5))
                    print(f"[Info] Auto-loaded calibrated Youden-J optimal threshold: {threshold:.4f} from {json_meta_path.name}")
                except Exception:
                    threshold = 0.5
            else:
                threshold = 0.5
    else:
        torch.set_num_threads(args.cpu_threads)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model = build_model_from_checkpoint(checkpoint).eval()
        if threshold is None:
            threshold = float(checkpoint.get("threshold", checkpoint.get("optimal_threshold", 0.5)))
            print(f"[Info] Auto-loaded calibrated optimal threshold: {threshold:.4f} from checkpoint")

    # Mode A: Synthetic Benchmark (0MB disk usage)
    if args.synthetic:
        bench_result = run_synthetic_benchmark(
            model=model,
            onnx_session=onnx_session,
            texture_frames=args.texture_frames,
            image_size=args.image_size,
            clips=args.clips,
            iterations=args.iterations,
        )
        print("\n" + "=" * 60)
        print("    TEXTURE-SBI SYNTHETIC BENCHMARK REPORT (0MB DISK)")
        print("=" * 60)
        print(f"Engine           : {bench_result['engine']}")
        print(f"Input Shape      : {bench_result['input_shape']} (Batch x Frames x C x H x W)")
        print(f"Iterations       : {bench_result['iterations']} runs")
        print(f"Mean Latency     : {bench_result['latency_ms_mean']} ms (P50: {bench_result['latency_ms_p50']} ms | P95: {bench_result['latency_ms_p95']} ms)")
        print(f"Min / Max        : {bench_result['latency_ms_min']} ms / {bench_result['latency_ms_max']} ms")
        print(f"Throughput       : {bench_result['throughput_fps']} FPS (Frames per sec)")
        print("=" * 60 + "\n")
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(bench_result, f, indent=2)
        return

    # If video or video-dir is provided, setup Face Landmarker
    is_arm = platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l")
    lm_model_path = args.model_task
    if args.landmark_backend == "mediapipe" or (args.landmark_backend == "auto" and not is_arm):
        lm_model_path = ensure_face_landmarker_model(args.model_task, download=True)
    landmarker = FaceLandmarkerExtractor(
        lm_model_path,
        running_mode="image",
        min_confidence=0.5,
        backend=args.landmark_backend,
    )

    video_files = []
    if args.video:
        video_files.append(Path(args.video))
    elif args.video_dir:
        vdir = Path(args.video_dir)
        video_files = sorted([f for f in vdir.iterdir() if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
        if not video_files:
            parser.error(f"No video files found in directory: {args.video_dir}")
    else:
        parser.error("Must provide --video, --video-dir, or --synthetic")

    reports: list[dict[str, object]] = []
    print(f"\nProcessing {len(video_files)} video(s) (Clips: {args.clips}, Flip-TTA: {args.flip_tta}, Target FPS: {args.target_fps})...")

    for vpath in video_files:
        try:
            rep = process_video_pipeline(
                video_path=vpath,
                landmarker=landmarker,
                model=model,
                onnx_session=onnx_session,
                texture_frames=args.texture_frames,
                target_fps=args.target_fps,
                image_size=args.image_size,
                clips=args.clips,
                flip_tta=args.flip_tta,
                device=torch.device("cpu"),
            )
            fake_prob = rep["detection"]["fake_probability"]
            prediction = "FAKE" if fake_prob >= threshold else "REAL"
            rep["detection"]["threshold"] = round(threshold, 4)
            rep["detection"]["prediction"] = prediction
            reports.append(rep)
            print(f"[{prediction}] {vpath.name:<25} | Score: {fake_prob*100:5.2f}% | Latency: {rep['timings_ms']['total_end_to_end_ms']:6.1f}ms | FPS: {rep['performance']['end_to_end_fps']:4.1f}")
        except Exception as e:
            print(f"[ERROR] Failed processing {vpath.name}: {e}")

    landmarker.close()

    if not reports:
        print("No videos processed successfully.")
        return

    # Print summary
    if len(reports) == 1:
        rep = reports[0]
        print("\n" + "=" * 60)
        print("      TEXTURE-SBI VIDEO DETECTION REPORT (PI 4 / EDGE)")
        print("=" * 60)
        print(f"Input Video      : {rep['video_info']['path']}")
        print(f"Resolution / FPS : {rep['video_info']['resolution']} @ {rep['video_info']['original_fps']:.1f} FPS")
        print(f"Frames Sampled   : {rep['video_info']['sampled_frames']} frames")
        print("-" * 60)
        print("TIMING BREAKDOWN:")
        for k, v in rep["timings_ms"].items():
            print(f"  - {k:<32}: {v:>8.2f} ms")
        print(f"Throughput       : {rep['performance']['end_to_end_fps']} FPS (End-to-End)")
        print(f"Model-Only FPS   : {rep['performance']['model_forward_fps']} FPS")
        print("-" * 60)
        print(f"Fake Probability : {rep['detection']['fake_probability'] * 100:.2f}% (Threshold: {threshold:.4f})")
        print(f"PREDICTION       : >>> {rep['detection']['prediction']} <<<")
        print("=" * 60 + "\n")
    else:
        # Multi-video comprehensive scientific summary
        total_latencies = [r["timings_ms"]["total_end_to_end_ms"] for r in reports]
        decode_latencies = [r["timings_ms"]["1_video_decode_ms"] for r in reports]
        lm_latencies = [r["timings_ms"]["2_landmark_extraction_ms"] for r in reports]
        align_latencies = [r["timings_ms"]["3_face_align_and_preprocess_ms"] for r in reports]
        model_latencies = [r["timings_ms"]["4_model_forward_ms"] for r in reports]
        fps_list = [r["performance"]["end_to_end_fps"] for r in reports]
        correct_predictions = sum(1 for r in reports if r["detection"]["prediction"] == ("FAKE" if "synthesis" in str(r["video_info"]["path"]).lower() else "REAL"))

        def stats_str(data: list[float]) -> str:
            m = statistics.mean(data)
            s = statistics.stdev(data) if len(data) > 1 else 0.0
            p50 = percentile(data, 0.50)
            p95 = percentile(data, 0.95)
            return f"{m:>8.2f} ± {s:<6.2f} | {p50:>8.2f} | {p95:>8.2f}"

        print("\n" + "=" * 78)
        print(f"      TEXTURE-SBI SCIENTIFIC HARDWARE PROFILING REPORT ({len(reports)} VIDEOS)")
        print("=" * 78)
        print(f"Total Videos Processed  : {len(reports)}")
        print(f"Batch Detection Accuracy: {correct_predictions}/{len(reports)} ({correct_predictions/len(reports)*100:.1f}%)")
        print("-" * 78)
        print(f"{'Pipeline Stage':<35} | {'Mean ± Std (ms)':<17} | {'P50 (ms)':<8} | {'P95 (ms)':<8}")
        print("-" * 78)
        print(f"{'1. Video Decode & Sampling':<35} | {stats_str(decode_latencies)}")
        print(f"{'2. Face & Landmark Extraction':<35} | {stats_str(lm_latencies)}")
        print(f"{'3. Canonical Affine Alignment':<35} | {stats_str(align_latencies)}")
        print(f"{'4. Neural Model Forward':<35} | {stats_str(model_latencies)}")
        print("-" * 78)
        print(f"{'TOTAL END-TO-END PIPELINE':<35} | {stats_str(total_latencies)}")
        print("-" * 78)
        print(f"Throughput (End-to-End) : {statistics.mean(fps_list):.2f} ± {statistics.stdev(fps_list) if len(fps_list)>1 else 0.0:.2f} FPS")
        print(f"Throughput (Model-Only) : {(args.clips * (2 if args.flip_tta else 1) * args.texture_frames) / (statistics.mean(model_latencies) / 1000.0):.2f} FPS")
        print("=" * 78 + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(reports if len(reports) > 1 else reports[0], f, indent=2)
        print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
