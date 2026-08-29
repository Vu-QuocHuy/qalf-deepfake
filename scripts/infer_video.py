#!/usr/bin/env python3
"""End-to-end Video Deepfake Detection and Scientific Hardware Profiler for Raspberry Pi & Edge Devices.

Strictly follows the canonical QALF / TextureSBI evaluation protocol:
- 10 FPS temporal sampling
- 32-frame clip window with 8 distributed texture frames [0, 4, 9, 13, 18, 22, 27, 31]
- MTCNN 35% margin face localization to 256x256 canonical face crop
- 468 landmark affine alignment to 160x160 with eyes horizontal at 0 deg
- 3 clips per video with Horizontal Flip-TTA and Mean/Top-k Aggregation
- Auto-loaded calibrated EER threshold from checkpoint / JSON metadata
- Complete End-to-End Latency Breakdown and Hardware Profiling
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
try:
    import psutil
except ImportError:
    psutil = None

from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import IMAGE_MEAN, IMAGE_STD, _aligned_full_face
from qalf.data.landmarks import FaceLandmarkerExtractor, OpenCVYuNetLandmarker, ensure_face_landmarker_model


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def compute_auc(y_true: list[int], y_scores: list[float]) -> float:
    if len(set(y_true)) < 2: return 0.0
    desc_score_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    y_true_sorted = [y_true[i] for i in desc_score_indices]
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    auc = 0.0
    tp = 0
    for y in y_true_sorted:
        if y == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def compute_eer(y_true: list[int], y_scores: list[float]) -> float:
    if len(set(y_true)) < 2: return 0.0
    thresholds = sorted(set(y_scores))
    eer = 1.0
    for t in thresholds:
        tp = sum(1 for yt, ys in zip(y_true, y_scores) if yt == 1 and ys >= t)
        fn = sum(1 for yt, ys in zip(y_true, y_scores) if yt == 1 and ys < t)
        fp = sum(1 for yt, ys in zip(y_true, y_scores) if yt == 0 and ys >= t)
        tn = sum(1 for yt, ys in zip(y_true, y_scores) if yt == 0 and ys < t)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = 1 - tpr
        diff = abs(fnr - fpr)
        if diff < eer:
            eer = diff
    return eer


monitor_data = {"time_ms": [], "cpu": [], "ram": [], "temp": [], "power_w": []}
monitoring_active = True

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def monitor_hardware():
    if not psutil: return
    process = psutil.Process(os.getpid())
    start_time = time.time()
    while monitoring_active:
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            ram = process.memory_info().rss / (1024 * 1024)
            temp = get_cpu_temp()
            # Pi 4 estimated power model: 2.7W idle, 6.4W full load
            estimated_power = 2.7 + ((6.4 - 2.7) * (cpu / 100.0))
            
            monitor_data["time_ms"].append(int((time.time() - start_time) * 1000))
            monitor_data["cpu"].append(cpu)
            monitor_data["ram"].append(ram)
            monitor_data["temp"].append(temp)
            monitor_data["power_w"].append(estimated_power)
            time.sleep(0.4)
        except Exception:
            pass


def compute_protocol_frame_indices(
    total_video_frames: int,
    source_fps: float,
    target_fps: float = 10.0,
    num_frames: int = 32,
    texture_frames: int = 8,
    clips_per_video: int = 3,
) -> tuple[list[int], list[list[int]]]:
    """Compute exact frame indices matching QALFVideoDataset / run_test.sh protocol.
    
    1. Sample 64-frame sequence at target_fps (10 FPS, stride = round(source_fps / 10.0)).
    2. Draw clips_per_video (e.g. 3) windows of length num_frames (32 frames) with starts [0, 16, 32].
    3. In each 32-frame window, sample texture_frames (8) distributed positions:
       [0, 4, 9, 13, 18, 22, 27, 31].
    """
    stride = max(1, int(round(source_fps / target_fps)))
    total_seq_length = 64
    
    seq_indices = [min(total_video_frames - 1, i * stride) for i in range(total_seq_length)]
    
    if len(seq_indices) <= num_frames:
        starts = [0] * clips_per_video
    elif clips_per_video == 1:
        starts = [(len(seq_indices) - num_frames) // 2]
    else:
        starts = np.rint(np.linspace(0, len(seq_indices) - num_frames, clips_per_video)).astype(int).tolist()

    # Offsets within a 32-frame window: [0, 4, 9, 13, 18, 22, 27, 31]
    tex_offsets = np.rint(np.linspace(0, num_frames - 1, texture_frames)).astype(int).tolist()

    clip_index_groups: list[list[int]] = []
    all_needed_set: set[int] = set()

    for s in starts:
        group = []
        for offset in tex_offsets:
            frame_idx = seq_indices[min(len(seq_indices) - 1, s + offset)]
            group.append(frame_idx)
            all_needed_set.add(frame_idx)
        clip_index_groups.append(group)

    all_unique_indices = sorted(list(all_needed_set))
    return all_unique_indices, clip_index_groups


def extract_video_frames(
    video_path: str | Path,
    indices: list[int],
) -> tuple[dict[int, np.ndarray], float]:
    """Read specific target frames from video file and measure I/O decode latency."""
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

    if len(frame_dict) < len(idx_set):
        missing = [idx for idx in indices if idx not in frame_dict]
        raise RuntimeError(f"Failed to extract {len(missing)} target frames. Missing: {missing}")

    return frame_dict, read_elapsed_ms


def crop_face_to_256(image_rgb: np.ndarray, mtcnn_detector=None, yunet_detector=None) -> np.ndarray:
    """Crop face to square with 35% margin resized to 256x256 matching extract_frames.py."""
    height, width = image_rgb.shape[:2]
    
    def _crop_and_pad(bx1, by1, bx2, by2):
        bw, bh = bx2 - bx1, by2 - by1
        cx, cy = bx1 + bw / 2.0, by1 + bh / 2.0
        side = max(bw, bh) * 1.35
        
        x1_ideal, y1_ideal = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
        x2_ideal, y2_ideal = int(round(cx + side / 2.0)), int(round(cy + side / 2.0))

        pad_left, pad_top = max(0, -x1_ideal), max(0, -y1_ideal)
        pad_right, pad_bottom = max(0, x2_ideal - width), max(0, y2_ideal - height)

        x1, y1 = max(0, x1_ideal), max(0, y1_ideal)
        x2, y2 = min(width, x2_ideal), min(height, y2_ideal)

        crop = image_rgb[y1:y2, x1:x2]
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        
        if crop.shape[0] > 16 and crop.shape[1] > 16:
            return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        return None

    # Prefer YuNet over MTCNN
    if yunet_detector is not None:
        bbox = yunet_detector.detect_bbox(image_rgb)
        if bbox is not None:
            res = _crop_and_pad(*bbox)
            if res is not None: return res

    if mtcnn_detector is not None:
        try:
            boxes, _ = mtcnn_detector.detect(image_rgb)
            if boxes is not None and len(boxes) > 0 and boxes[0] is not None:
                res = _crop_and_pad(*boxes[0])
                if res is not None: return res
        except Exception:
            pass
            
    return cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)


def process_video_pipeline(
    video_path: str | Path,
    landmarker: FaceLandmarkerExtractor | None,
    model: torch.nn.Module | None,
    onnx_session: object | None,
    mtcnn_detector: object | None = None,
    yunet_detector: object | None = None,
    num_frames: int = 32,
    texture_frames: int = 8,
    target_fps: float = 10.0,
    image_size: int = 160,
    clips: int = 3,
    flip_tta: bool = True,
    aggregation: str = "mean",
    top_k: int = 1,
    device: torch.device = torch.device("cpu"),
    no_landmarks: bool = False,
) -> dict[str, object]:
    """Run exact end-to-end edge pipeline matching training/testing protocol."""
    timing: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    # 1. Video Meta & Targeted Decoding
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    all_indices, clip_index_groups = compute_protocol_frame_indices(
        total_video_frames=total_video_frames,
        source_fps=fps,
        target_fps=target_fps,
        num_frames=num_frames,
        texture_frames=texture_frames,
        clips_per_video=clips,
    )

    frame_dict, decode_ms = extract_video_frames(video_path, all_indices)
    timing["1_video_decode_ms"] = decode_ms

    if not frame_dict:
        raise RuntimeError(f"No frames could be extracted from: {video_path}")

    # 2. MTCNN Face Crop & Facial Landmark Extraction
    lm_start = time.perf_counter()
    aligned_map: dict[int, np.ndarray] = {}

    for idx in all_indices:
        raw_frame = frame_dict[idx]
        
        # Optimize YuNet: Single pass on raw frame if YuNet is the landmarker
        # This resolves the performance bottleneck (YuNet running twice).
        # We MUST perform the 256x256 intermediate crop using INTER_AREA to prevent aliasing.
        is_yunet_lm = isinstance(landmarker, OpenCVYuNetLandmarker) or getattr(landmarker, "backend", "") == "yunet"
        
        if is_yunet_lm and yunet_detector is not None:
            # 1. Single YuNet pass
            bbox, pts = yunet_detector.process_with_bbox(raw_frame)
            has_lms = pts is not None
            
            # 2. Crop to 256x256 with padding (using bbox)
            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                bw, bh = bx2 - bx1, by2 - by1
                cx, cy = bx1 + bw / 2.0, by1 + bh / 2.0
                side = max(bw, bh) * 1.35
                
                # We need to compute x1, y1 for mapping the landmarks
                x1_ideal = cx - side / 2.0
                y1_ideal = cy - side / 2.0
                
                height, width = raw_frame.shape[:2]
                pad_left, pad_top = max(0, int(round(-x1_ideal))), max(0, int(round(-y1_ideal)))
                pad_right, pad_bottom = max(0, int(round(x1_ideal + side - width))), max(0, int(round(y1_ideal + side - height)))

                x1, y1 = max(0, int(round(x1_ideal))), max(0, int(round(y1_ideal)))
                x2, y2 = min(width, int(round(x1_ideal + side))), min(height, int(round(y1_ideal + side)))

                crop = raw_frame[y1:y2, x1:x2]
                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
                
                face_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
                
                # 3. Map normalized landmarks from raw_frame to the new face_256 image
                if has_lms:
                    # Un-normalize from raw_frame, subtract x1_ideal/y1_ideal, and re-normalize to 256x256
                    mapped_pts = np.zeros_like(pts)
                    for i in range(5):
                        px_raw = pts[i, 0] * width
                        py_raw = pts[i, 1] * height
                        px_256 = (px_raw - x1_ideal) * (256.0 / side)
                        py_256 = (py_raw - y1_ideal) * (256.0 / side)
                        mapped_pts[i, 0] = px_256 / 256.0
                        mapped_pts[i, 1] = py_256 / 256.0
                    pts = mapped_pts
            else:
                face_256 = cv2.resize(raw_frame, (256, 256), interpolation=cv2.INTER_AREA)
                pts = np.zeros((5, 3), dtype=np.float32)
                has_lms = False
                
            crop, _ = _aligned_full_face(
                face_256,
                pts if has_lms else np.zeros((5, 3), dtype=np.float32),
                detected=has_lms,
                output_size=image_size,
            )
        else:
            # Legacy MTCNN + MediaPipe (or mixed) two-stage pipeline
            face_256 = crop_face_to_256(raw_frame, mtcnn_detector=mtcnn_detector, yunet_detector=yunet_detector)
            
            if no_landmarks or landmarker is None:
                has_lms = False
                pts = None
            else:
                pts = landmarker.process(face_256)
                has_lms = pts is not None

            crop, _ = _aligned_full_face(
                face_256,
                pts if has_lms else np.zeros((468, 3), dtype=np.float32),
                detected=has_lms,
                output_size=image_size,
            )
        crop_norm = (crop.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        crop_chw = np.transpose(crop_norm, (2, 0, 1))
        aligned_map[idx] = crop_chw

    timing["2_landmark_and_crop_ms"] = (time.perf_counter() - lm_start) * 1000.0

    # 3. Canonical Affine Alignment & Clip Batch Tensor Construction
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

    # 4. Neural Network Inference (ONNX or PyTorch)
    infer_start = time.perf_counter()
    if onnx_session is not None:
        input_name = onnx_session.get_inputs()[0].name
        logits = onnx_session.run(None, {input_name: eval_batch})[0]
        if isinstance(logits, np.ndarray):
            logits_sq = logits.squeeze(-1) if logits.ndim > 1 else logits
            raw_scores = 1.0 / (1.0 + np.exp(-logits_sq))
    elif model is not None:
        batch_tensor = torch.from_numpy(eval_batch).to(device)
        with torch.inference_mode():
            out = model({"texture": batch_tensor})
            logits = out["logit"]
            raw_scores = torch.sigmoid(logits).cpu().numpy()
            if raw_scores.ndim > 1 and raw_scores.shape[-1] == 1:
                raw_scores = raw_scores.squeeze(-1)
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

    # Aggregate across clips
    if aggregation == "topk":
        final_score = float(np.mean(np.sort(clip_scores)[-top_k:]))
    else:
        final_score = float(np.mean(clip_scores))

    total_pipeline_time_s = time.perf_counter() - pipeline_start
    timing["total_end_to_end_ms"] = total_pipeline_time_s * 1000.0
    effective_fps = len(all_indices) / total_pipeline_time_s if total_pipeline_time_s > 0 else 0.0

    return {
        "video_info": {
            "path": str(video_path),
            "total_video_frames": total_video_frames,
            "sampled_unique_frames": len(all_indices),
            "original_fps": fps,
            "resolution": f"{width}x{height}",
        },
        "timings_ms": timing,
        "performance": {
            "end_to_end_fps": round(effective_fps, 2),
            "model_forward_fps": round((clips * (2 if flip_tta else 1) * texture_frames) / (timing["4_model_forward_ms"] / 1000.0), 2),
        },
        "detection": {
            "fake_probability": float(final_score),
            "clip_scores": [float(s) for s in clip_scores],
            "clips": clips,
            "texture_frames": texture_frames,
            "flip_tta": flip_tta,
            "aggregation": aggregation,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Video Deepfake Detection & Scientific Profiler for Pi 4")
    parser.add_argument("--video", default=None, help="Path to single input video file (.mp4, .avi, etc.)")
    parser.add_argument("--video-dir", default=None, help="Path to directory containing subset of test videos")
    parser.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", default=None, help="Path to ONNX model (.onnx)")
    parser.add_argument("--model-task", default="models/face_landmarker.task", help="MediaPipe face landmarker model")
    parser.add_argument("--num-frames", type=int, default=32, help="Temporal window size in 10 FPS sequence (default 32)")
    parser.add_argument("--texture-frames", type=int, default=8, help="Number of texture frames sampled across window (default 8)")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target temporal sampling rate (default 10.0 FPS)")
    parser.add_argument("--clips", type=int, default=3, help="Clips per video (default 3 matching paper protocol, 1 for fast edge)")
    parser.add_argument("--aggregation", choices=["mean", "topk"], default="mean", help="Clip score aggregation method")
    parser.add_argument("--top-k", type=int, default=1, help="Top-K clips to aggregate if using topk")
    parser.add_argument("--image-size", type=int, default=160, help="Input resolution (160x160)")
    parser.add_argument("--no-flip-tta", action="store_true", help="Disable test-time horizontal flip augmentation")
    parser.add_argument("--no-landmarks", action="store_true", help="Ablation: Disable landmark alignment, only resize face crop (MASSIVELY FASTER)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for Real vs Fake (auto-loaded if None)")
    parser.add_argument("--landmark-backend", default="auto", choices=["auto", "mediapipe", "opencv", "yunet"], help="Landmark backend")
    parser.add_argument("--cpu-threads", type=int, default=4, help="Number of CPU threads to use on Pi 4")
    parser.add_argument("--use-mtcnn", action="store_true", default=False, help="Use MTCNN for face cropping (legacy, slower)")
    parser.add_argument("--use-yunet", action="store_true", default=True, help="Use YuNet DNN for face detection (default, faster)")
    parser.add_argument("--output-json", default=None, help="Save timing and prediction report to JSON")
    parser.add_argument("--output-log", default=None, help="Save stdout logs (hardware, metrics) to a text file")
    args = parser.parse_args()

    if args.output_log:
        class Logger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                self.log = open(filename, "w", encoding="utf-8")
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
            def flush(self):
                self.terminal.flush()
                self.log.flush()
        sys.stdout = Logger(args.output_log)

    if not args.checkpoint and not args.onnx:
        parser.error("Must provide either --checkpoint (.pt) or --onnx (.onnx)")

    # 1. Setup Model (PyTorch or ONNX) and Auto-load Threshold
    model = None
    onnx_session = None
    threshold = args.threshold
    flip_tta = not args.no_flip_tta

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
                    threshold = float(meta.get("optimal_threshold", 0.6712))
                    print(f"[Info] Auto-loaded EER optimal threshold: {threshold:.6f} from {json_meta_path.name}")
                except Exception:
                    threshold = 0.6712
            else:
                threshold = 0.6712
    else:
        from qalf.models import build_model_from_checkpoint
        torch.set_num_threads(args.cpu_threads)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model = build_model_from_checkpoint(checkpoint).eval()
        if threshold is None:
            threshold = float(checkpoint.get("threshold", checkpoint.get("optimal_threshold", 0.6712)))
            print(f"[Info] Auto-loaded EER optimal threshold: {threshold:.6f} from checkpoint")

    # 2. Setup Face Detector (YuNet preferred over MTCNN)
    mtcnn_detector = None
    yunet_detector = None
    if args.use_mtcnn and not args.use_yunet:
        try:
            from facenet_pytorch import MTCNN
            mtcnn_detector = MTCNN(image_size=256, margin=0, keep_all=False, post_process=False, device="cpu")
        except Exception:
            mtcnn_detector = None
    else:
        yunet_detector = OpenCVYuNetLandmarker(score_threshold=0.5)

    # 3. Setup Face Landmarker
    landmarker = None
    if not args.no_landmarks:
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
        video_files = sorted([f for f in vdir.rglob("*") if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
        if not video_files:
            parser.error(f"No video files found in directory: {args.video_dir}")
    else:
        parser.error("Must provide --video or --video-dir")

    reports: list[dict[str, object]] = []
    print(f"\nProcessing {len(video_files)} video(s) (Protocol: {args.clips} clips x {args.texture_frames}f across {args.num_frames}f @ {args.target_fps} FPS, Flip-TTA: {flip_tta}, Aggregation: {args.aggregation})...")

    global monitoring_active
    monitor_thread = threading.Thread(target=monitor_hardware)
    monitor_thread.start()

    for vpath in video_files:
        try:
            rep = process_video_pipeline(
                video_path=vpath,
                landmarker=landmarker,
                model=model,
                onnx_session=onnx_session,
                mtcnn_detector=mtcnn_detector,
                yunet_detector=yunet_detector,
                num_frames=args.num_frames,
                texture_frames=args.texture_frames,
                target_fps=args.target_fps,
                image_size=args.image_size,
                clips=args.clips,
                flip_tta=flip_tta,
                aggregation=args.aggregation,
                top_k=args.top_k,
                device=torch.device("cpu"),
                no_landmarks=args.no_landmarks,
            )
            fake_prob = rep["detection"]["fake_probability"]
            prediction = "FAKE" if fake_prob >= threshold else "REAL"
            rep["detection"]["threshold"] = round(threshold, 4)
            rep["detection"]["prediction"] = prediction
            reports.append(rep)
            print(f"[{prediction}] {vpath.name:<25} | Score: {fake_prob*100:5.2f}% | Latency: {rep['timings_ms']['total_end_to_end_ms']:6.1f}ms | FPS: {rep['performance']['end_to_end_fps']:4.1f}", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed processing {vpath.name}: {e}", flush=True)

    if landmarker is not None:
        landmarker.close()
    
    monitoring_active = False
    monitor_thread.join()

    # Save telemetry data to CSV
    try:
        import csv
        telemetry_path = Path("hardware_telemetry.csv")
        with open(telemetry_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_ms", "cpu_percent", "ram_mb", "temp_c", "power_w"])
            for i in range(len(monitor_data["time_ms"])):
                writer.writerow([
                    monitor_data["time_ms"][i],
                    monitor_data["cpu"][i],
                    monitor_data["ram"][i],
                    monitor_data["temp"][i],
                    monitor_data["power_w"][i]
                ])
        print(f"\n[+] Hardware telemetry saved to {telemetry_path.absolute()}")
    except Exception as e:
        print(f"[!] Failed to save hardware telemetry: {e}")

    if not reports:
        print("No videos processed successfully.")
        return

    # Print summary
    if len(reports) == 1:
        rep = reports[0]
        print("\n" + "=" * 65)
        print("      TEXTURE-SBI END-TO-END VIDEO INFERENCE REPORT (PI 4)")
        print("=" * 65)
        print(f"Input Video         : {rep['video_info']['path']}")
        print(f"Resolution / FPS    : {rep['video_info']['resolution']} @ {rep['video_info']['original_fps']:.1f} FPS")
        print(f"Unique Frames Read  : {rep['video_info']['sampled_unique_frames']} frames")
        print(f"Protocol Windows    : {rep['detection']['clips']} clips x {rep['detection']['texture_frames']}f (Flip-TTA: {rep['detection']['flip_tta']})")
        print("-" * 65)
        print("EXACT TIMING BREAKDOWN (END-TO-END):")
        for k, v in rep["timings_ms"].items():
            print(f"  - {k:<34}: {v:>8.2f} ms")
        print(f"Throughput (End-to-End): {rep['performance']['end_to_end_fps']} FPS")
        print(f"Throughput (Model-Only): {rep['performance']['model_forward_fps']} FPS")
        print("-" * 65)
        print(f"Clip Probabilities  : {rep['detection']['clip_scores']}")
        print(f"Fake Probability    : {rep['detection']['fake_probability'] * 100:.2f}% (Threshold: {threshold:.4f})")
        print(f"FINAL PREDICTION    : >>> {rep['detection']['prediction']} <<<")
        print("=" * 65 + "\n")
    else:
        total_latencies = [r["timings_ms"]["total_end_to_end_ms"] for r in reports]
        decode_latencies = [r["timings_ms"]["1_video_decode_ms"] for r in reports]
        lm_latencies = [r["timings_ms"]["2_landmark_and_crop_ms"] for r in reports]
        align_latencies = [r["timings_ms"]["3_face_align_and_preprocess_ms"] for r in reports]
        model_latencies = [r["timings_ms"]["4_model_forward_ms"] for r in reports]
        fps_list = [r["performance"]["end_to_end_fps"] for r in reports]
        
        y_true = [1 if "synthesis" in str(r["video_info"]["path"]).lower() else 0 for r in reports]
        y_scores = [r["detection"]["fake_probability"] for r in reports]
        y_preds = [1 if r["detection"]["prediction"] == "FAKE" else 0 for r in reports]
        
        correct_predictions = sum(1 for yt, yp in zip(y_true, y_preds) if yt == yp)
        auc = compute_auc(y_true, y_scores)
        eer = compute_eer(y_true, y_scores)

        def stats_str(data: list[float]) -> str:
            m = statistics.mean(data)
            s = statistics.stdev(data) if len(data) > 1 else 0.0
            p50 = percentile(data, 0.50)
            p95 = percentile(data, 0.95)
            return f"{m:>8.2f} ± {s:<6.2f} | {p50:>8.2f} | {p95:>8.2f}"

        print("\n" + "=" * 80)
        print(f"      TEXTURE-SBI SCIENTIFIC HARDWARE PROFILING REPORT ({len(reports)} VIDEOS)")
        print("=" * 80)
        print(f"Total Videos Processed  : {len(reports)}")
        print(f"Batch Detection Accuracy: {correct_predictions}/{len(reports)} ({correct_predictions/len(reports)*100:.1f}%)")
        print("-" * 80)
        print(f"{'Pipeline Stage':<36} | {'Mean ± Std (ms)':<17} | {'P50 (ms)':<8} | {'P95 (ms)':<8}")
        print("-" * 80)
        print(f"{'1. Video Decode & 10 FPS Sampling':<36} | {stats_str(decode_latencies)}")
        print(f"{'2. Face Localization & Landmarks':<36} | {stats_str(lm_latencies)}")
        print(f"{'3. Canonical Affine Alignment':<36} | {stats_str(align_latencies)}")
        print(f"{'4. Neural Forward (ONNX FP32)':<36} | {stats_str(model_latencies)}")
        print("-" * 80)
        print(f"{'TOTAL END-TO-END PIPELINE':<36} | {stats_str(total_latencies)}")
        print("-" * 80)
        print(f"Throughput (End-to-End) : {statistics.mean(fps_list):.2f} ± {statistics.stdev(fps_list) if len(fps_list)>1 else 0.0:.2f} FPS")
        print(f"Throughput (Model-Only) : {(args.clips * (2 if flip_tta else 1) * args.texture_frames) / (statistics.mean(model_latencies) / 1000.0):.2f} FPS")
        print("-" * 80)
        print(f"EVALUATION METRICS (Threshold: {threshold:.4f})")
        print("-" * 80)
        print(f"  AUC-ROC                    : {auc * 100:>7.2f} %")
        print(f"  Equal Error Rate (EER)     : {eer * 100:>7.2f} %")
        print(f"  Batch Detection Accuracy   : {correct_predictions/len(reports)*100:>7.2f} % ({correct_predictions}/{len(reports)})")
        print("-" * 80)
        print("HARDWARE RESOURCE USAGE")
        print("-" * 80)
        if monitor_data["cpu"]:
            avg_cpu = sum(monitor_data["cpu"]) / max(1, len(monitor_data["cpu"]))
            max_cpu = max(monitor_data["cpu"]) if monitor_data["cpu"] else 0
            avg_ram = sum(monitor_data["ram"]) / max(1, len(monitor_data["ram"]))
            max_ram = max(monitor_data["ram"]) if monitor_data["ram"] else 0
            avg_temp = sum(monitor_data["temp"]) / max(1, len(monitor_data["temp"]))
            max_temp = max(monitor_data["temp"]) if monitor_data["temp"] else 0
            print(f"  CPU Usage (System) : Avg: {avg_cpu:5.1f}% | Peak: {max_cpu:5.1f}%")
            print(f"  RAM Usage (Process): Avg: {avg_ram:5.1f}MB | Peak: {max_ram:5.1f}MB")
            print(f"  CPU Temperature    : Avg: {avg_temp:5.1f}°C | Peak: {max_temp:5.1f}°C")
        else:
            print("  Hardware metrics not available (install psutil on Pi)")
        print("=" * 80 + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(reports if len(reports) > 1 else reports[0], f, indent=2)
        print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
