#!/usr/bin/env python3
"""Publication-grade hardware profiling for Raspberry Pi 4 edge deployment.

Measures end-to-end latency breakdown, CPU/RAM/temperature utilisation, and
saves raw per-video timings and resource samples for offline analysis.

Improvements over the original profiler:
- Balanced real/fake sampling or explicit video list
- Deterministic seed-based selection
- Configurable warm-up (excluded from summary statistics)
- Per-video CSV with full-precision scores
- Raw resource samples CSV with timestamps
- Environment metadata (OS, Python, ORT, governor, throttling)
- Summary statistics: mean/std/median/P50/P95 for latency; mean/peak for resources

Usage (minimal — 50 balanced videos):
    python scripts/profile_pi4_pipeline.py \\
        --onnx models/model.onnx \\
        --video-root /mnt/usb_data/celebdf_test_518 \\
        --balanced --n-videos 50 --seed 42 \\
        --warmup-videos 5 --clips 3 --backend yunet \\
        --output-dir eval_pi4_profile_yunet_3clip_n50

Usage (full — all 518 videos):
    python scripts/profile_pi4_pipeline.py \\
        --onnx models/model.onnx \\
        --video-root /mnt/usb_data/celebdf_test_518 \\
        --n-videos 0 --warmup-videos 5 --clips 3 --backend yunet \\
        --output-dir eval_pi4_profile_yunet_3clip_all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required.  Install with:  pip install psutil", file=sys.stderr)
    sys.exit(1)

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def _read_cpu_temp() -> float:
    """Read SoC temperature on Linux (°C).  Returns 0.0 on failure."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as fh:
            return float(fh.read().strip()) / 1000.0
    except Exception:
        return 0.0


def _read_cpu_freq_mhz() -> float:
    """Read current CPU frequency in MHz.  Returns 0.0 on failure."""
    try:
        freq = psutil.cpu_freq()
        return float(freq.current) if freq else 0.0
    except Exception:
        return 0.0


def _read_governor() -> str:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r") as fh:
            return fh.read().strip()
    except Exception:
        return "unknown"


def _vcgencmd_get_throttled() -> str:
    """Query Raspberry Pi throttle status.  Returns hex string or 'unavailable'."""
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip().split("=")[-1] if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Resource monitoring thread
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Background thread that samples CPU / RAM / temperature at regular intervals."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._samples: list[dict] = []
        self._active = False
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())
        self._current_video: str = ""
        self._lock = threading.Lock()

    def start(self) -> None:
        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join(timeout=5)

    def set_current_video(self, video_id: str) -> None:
        with self._lock:
            self._current_video = video_id

    @property
    def samples(self) -> list[dict]:
        return list(self._samples)

    def _run(self) -> None:
        while self._active:
            with self._lock:
                vid = self._current_video
            self._samples.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "rss_mb": round(self._process.memory_info().rss / (1024 * 1024), 1),
                "cpu_temp_c": round(_read_cpu_temp(), 1),
                "cpu_freq_mhz": round(_read_cpu_freq_mhz(), 0),
                "current_video": vid,
            })
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Video collection helpers
# ---------------------------------------------------------------------------

def _collect_videos(video_root: Path) -> tuple[list[Path], list[Path]]:
    """Walk *video_root* and classify mp4s as real/fake based on directory name."""
    real_videos: list[Path] = []
    fake_videos: list[Path] = []
    for mp4 in sorted(video_root.rglob("*.mp4")):
        parent_lower = mp4.parent.name.lower()
        if "real" in parent_lower:
            real_videos.append(mp4)
        elif "synthesis" in parent_lower or "fake" in parent_lower:
            fake_videos.append(mp4)
        else:
            # Ambiguous — include in both for manual review
            fake_videos.append(mp4)
    return real_videos, fake_videos


def _select_videos(
    video_root: Path,
    n_videos: int,
    balanced: bool,
    seed: int,
    video_list: Path | None,
) -> list[tuple[Path, int]]:
    """Return list of (video_path, label) tuples.  label: 0=real, 1=fake."""

    if video_list is not None:
        entries = []
        for line in video_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = video_root / line
            label = 0 if "real" in path.parent.name.lower() else 1
            entries.append((path, label))
        return entries

    real_videos, fake_videos = _collect_videos(video_root)
    all_videos = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]

    if n_videos <= 0 or n_videos >= len(all_videos):
        rng = random.Random(seed)
        rng.shuffle(all_videos)
        return all_videos

    rng = random.Random(seed)
    if balanced:
        n_each = n_videos // 2
        selected_real = rng.sample(real_videos, min(n_each, len(real_videos)))
        selected_fake = rng.sample(fake_videos, min(n_each, len(fake_videos)))
        result = [(v, 0) for v in selected_real] + [(v, 1) for v in selected_fake]
    else:
        result = rng.sample(all_videos, min(n_videos, len(all_videos)))

    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _latency_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "std": 0, "median": 0, "p50": 0, "p95": 0, "min": 0, "max": 0, "count": 0}
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 1),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 1),
        "median": round(float(np.median(arr)), 1),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "min": round(float(arr.min()), 1),
        "max": round(float(arr.max()), 1),
        "count": len(arr),
    }


def _resource_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "peak": 0}
    return {
        "mean": round(float(np.mean(values)), 1),
        "peak": round(float(np.max(values)), 1),
    }


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def _collect_environment(stage: str) -> dict:
    env: dict = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "node": platform.node(),
            "python": platform.python_version(),
        },
        "cpu_governor": _read_governor(),
        "cpu_temp_c": round(_read_cpu_temp(), 1),
        "cpu_freq_mhz": round(_read_cpu_freq_mhz(), 0),
        "vcgencmd_get_throttled": _vcgencmd_get_throttled(),
    }
    try:
        import cv2
        env["opencv_version"] = cv2.__version__
    except Exception:
        pass
    try:
        import onnxruntime as ort
        env["onnxruntime_version"] = ort.__version__
    except Exception:
        pass
    try:
        mem = psutil.virtual_memory()
        env["ram_total_mb"] = round(mem.total / (1024 * 1024), 0)
    except Exception:
        pass
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication-grade Pi 4 hardware profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--onnx", default="models/qalf.onnx", help="Path to ONNX model")
    parser.add_argument("--video-root", required=True, help="Root directory containing videos (searched recursively)")
    parser.add_argument("--video-list", default=None, help="Text file listing relative video paths (one per line)")
    parser.add_argument("--balanced", action="store_true", help="Select N/2 real + N/2 fake videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for video selection")
    parser.add_argument("--n-videos", type=int, default=50, help="Number of videos (0 = all)")
    parser.add_argument("--warmup-videos", type=int, default=5, help="Warm-up videos excluded from stats")
    parser.add_argument("--clips", type=int, default=3, help="Clips per video")
    parser.add_argument("--texture-frames", type=int, default=8, help="Texture frames per clip")
    parser.add_argument("--num-frames", type=int, default=32, help="Temporal window size")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target temporal sampling rate")
    parser.add_argument("--image-size", type=int, default=160, help="Input resolution")
    parser.add_argument("--backend", default="yunet", choices=["yunet", "mediapipe", "opencv", "auto"], help="Face detector backend")
    parser.add_argument("--cpu-threads", type=int, default=4, help="ONNX Runtime intra-op threads")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--resource-interval", type=float, default=0.5, help="Resource sampling interval (seconds)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root)

    print("=" * 72)
    print("  PI 4 HARDWARE PROFILER — Publication Grade")
    print("=" * 72)

    # ---- Select videos ----
    video_list = Path(args.video_list) if args.video_list else None
    videos = _select_videos(video_root, args.n_videos, args.balanced, args.seed, video_list)
    if not videos:
        print(f"ERROR: No videos found in {video_root}", file=sys.stderr)
        sys.exit(1)

    n_real = sum(1 for _, l in videos if l == 0)
    n_fake = sum(1 for _, l in videos if l == 1)
    print(f"Videos selected  : {len(videos)} ({n_real} real, {n_fake} fake)")
    print(f"Warm-up          : {args.warmup_videos} videos (excluded from stats)")
    print(f"Protocol         : {args.clips} clips × {args.texture_frames}f, Flip-TTA, {args.backend}")
    print(f"Output           : {output_dir}")

    # ---- Load ONNX model ----
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.cpu_threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_session = ort.InferenceSession(str(args.onnx), sess_options, providers=["CPUExecutionProvider"])
    print(f"ONNX model       : {args.onnx}")

    # ---- Init face detector + landmarker ----
    from qalf.data.landmarks import FaceLandmarkerExtractor, OpenCVYuNetLandmarker
    yunet_detector = OpenCVYuNetLandmarker(score_threshold=0.5) if args.backend == "yunet" else None
    landmarker = FaceLandmarkerExtractor(running_mode="image", min_confidence=0.5, backend=args.backend)

    # ---- Import pipeline ----
    import torch
    from scripts.infer_video import process_video_pipeline

    # ---- Environment snapshot BEFORE ----
    env_before = _collect_environment("before")

    # ---- Start resource monitor ----
    monitor = ResourceMonitor(interval=args.resource_interval)
    monitor.start()

    # ---- Process videos ----
    per_video_rows: list[dict] = []
    n_errors = 0

    print(f"\n{'=' * 72}")
    total_start = time.perf_counter()

    for idx, (video_path, label) in enumerate(videos):
        is_warmup = idx < args.warmup_videos
        tag = "WARMUP" if is_warmup else "MAIN"
        video_id = video_path.stem
        monitor.set_current_video(video_id)

        try:
            rep = process_video_pipeline(
                video_path=video_path,
                landmarker=landmarker,
                model=None,
                onnx_session=onnx_session,
                mtcnn_detector=None,
                yunet_detector=yunet_detector,
                num_frames=args.num_frames,
                texture_frames=args.texture_frames,
                target_fps=args.target_fps,
                image_size=args.image_size,
                clips=args.clips,
                flip_tta=True,
                aggregation="mean",
                top_k=1,
                device=torch.device("cpu"),
                no_landmarks=False,
            )
            timings = rep["timings_ms"]
            score = rep["detection"]["fake_probability"]
            status = "ok"

            print(
                f"  [{idx+1:3d}/{len(videos)}] [{tag:6s}] {video_id:35s} "
                f"label={label} score={score:.6f} "
                f"total={timings.get('total_end_to_end_ms', 0):.0f}ms"
            )
        except Exception as exc:
            timings = {}
            score = float("nan")
            status = f"error: {exc}"
            n_errors += 1
            print(f"  [{idx+1:3d}/{len(videos)}] [{tag:6s}] {video_id:35s} ERROR: {exc}")

        per_video_rows.append({
            "video_id": video_id,
            "label": label,
            "is_warmup": is_warmup,
            "status": status,
            "score": score,
            "decode_ms": timings.get("1_video_decode_ms", ""),
            "landmark_ms": timings.get("2_landmark_and_crop_ms", ""),
            "align_ms": timings.get("3_face_align_and_preprocess_ms", ""),
            "model_ms": timings.get("4_model_forward_ms", ""),
            "total_ms": timings.get("total_end_to_end_ms", ""),
        })

    total_elapsed = time.perf_counter() - total_start
    monitor.set_current_video("")
    monitor.stop()

    # ---- Environment snapshot AFTER ----
    env_after = _collect_environment("after")

    # ---- Close resources ----
    try:
        landmarker.close()
    except Exception:
        pass

    # ---- Save per_video.csv ----
    csv_fields = ["video_id", "label", "is_warmup", "status", "score",
                  "decode_ms", "landmark_ms", "align_ms", "model_ms", "total_ms"]
    with open(output_dir / "per_video.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(per_video_rows)

    # ---- Save resource_samples.csv ----
    samples = monitor.samples
    if samples:
        sample_fields = list(samples[0].keys())
        with open(output_dir / "resource_samples.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sample_fields)
            writer.writeheader()
            writer.writerows(samples)

    # ---- Compute summary (non-warmup only) ----
    main_rows = [r for r in per_video_rows if not r["is_warmup"] and r["status"] == "ok"]

    timing_keys = {"decode_ms", "landmark_ms", "align_ms", "model_ms", "total_ms"}
    latency_summary = {}
    for key in timing_keys:
        values = [float(r[key]) for r in main_rows if r[key] != ""]
        latency_summary[key] = _latency_stats(values)

    # Resource summary (all samples taken during non-warmup, approximate)
    # Use all samples since warmup is short and resource usage is relevant overall
    cpu_vals = [s["cpu_percent"] for s in samples]
    rss_vals = [s["rss_mb"] for s in samples]
    temp_vals = [s["cpu_temp_c"] for s in samples if s["cpu_temp_c"] > 0]
    freq_vals = [s["cpu_freq_mhz"] for s in samples if s["cpu_freq_mhz"] > 0]

    resource_summary = {
        "cpu_percent": _resource_stats(cpu_vals),
        "rss_mb": _resource_stats(rss_vals),
        "cpu_temp_c": _resource_stats(temp_vals),
        "cpu_freq_mhz": _resource_stats(freq_vals),
    }

    # ---- Save environment.json ----
    environment = {
        "before": env_before,
        "after": env_after,
        "run_config": {
            "onnx_model": str(args.onnx),
            "video_root": str(args.video_root),
            "video_list": str(args.video_list) if args.video_list else None,
            "n_videos_requested": args.n_videos,
            "n_videos_actual": len(videos),
            "n_real": n_real,
            "n_fake": n_fake,
            "warmup_videos": args.warmup_videos,
            "balanced": args.balanced,
            "seed": args.seed,
            "clips": args.clips,
            "texture_frames": args.texture_frames,
            "num_frames": args.num_frames,
            "target_fps": args.target_fps,
            "image_size": args.image_size,
            "backend": args.backend,
            "cpu_threads": args.cpu_threads,
            "flip_tta": True,
            "aggregation": "mean",
            "resource_interval_s": args.resource_interval,
        },
        "total_wall_time_s": round(total_elapsed, 1),
        "videos_ok": len(main_rows),
        "videos_error": n_errors,
    }
    with open(output_dir / "environment.json", "w", encoding="utf-8") as fh:
        json.dump(environment, fh, indent=2, ensure_ascii=False)

    # ---- Save summary.json ----
    summary = {
        "latency": latency_summary,
        "resources": resource_summary,
        "videos_profiled": len(main_rows),
        "videos_error": n_errors,
        "warmup_videos": args.warmup_videos,
        "total_wall_time_s": round(total_elapsed, 1),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # ---- Print & save summary.txt ----
    lines: list[str] = []

    def pline(text: str = "") -> None:
        lines.append(text)
        print(text)

    pline(f"\n{'=' * 72}")
    pline("  LATENCY BREAKDOWN (non-warmup videos)")
    pline(f"{'=' * 72}")
    for key in ["decode_ms", "landmark_ms", "align_ms", "model_ms", "total_ms"]:
        s = latency_summary.get(key, {})
        pline(
            f"  {key:<20s}: mean={s.get('mean',0):>8.1f}ms  "
            f"std={s.get('std',0):>6.1f}ms  "
            f"P50={s.get('p50',0):>8.1f}ms  "
            f"P95={s.get('p95',0):>8.1f}ms"
        )

    pline(f"\n{'=' * 72}")
    pline("  HARDWARE RESOURCE UTILISATION")
    pline(f"{'=' * 72}")
    pline(f"  CPU (system)     : mean={resource_summary['cpu_percent']['mean']:>5.1f}%  peak={resource_summary['cpu_percent']['peak']:>5.1f}%")
    pline(f"  RAM (process RSS): mean={resource_summary['rss_mb']['mean']:>5.1f}MB peak={resource_summary['rss_mb']['peak']:>5.1f}MB")
    pline(f"  Temperature      : mean={resource_summary['cpu_temp_c']['mean']:>5.1f}°C peak={resource_summary['cpu_temp_c']['peak']:>5.1f}°C")
    if freq_vals:
        pline(f"  CPU frequency    : mean={resource_summary['cpu_freq_mhz']['mean']:>5.0f}MHz peak={resource_summary['cpu_freq_mhz']['peak']:>5.0f}MHz")

    pline(f"\n{'=' * 72}")
    pline("  RUN SUMMARY")
    pline(f"{'=' * 72}")
    pline(f"  Videos profiled  : {len(main_rows)} (+ {args.warmup_videos} warmup)")
    pline(f"  Errors           : {n_errors}")
    pline(f"  Total wall time  : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    pline(f"  Throttled before : {env_before['vcgencmd_get_throttled']}")
    pline(f"  Throttled after  : {env_after['vcgencmd_get_throttled']}")
    pline(f"  CPU governor     : {env_before['cpu_governor']}")
    pline(f"{'=' * 72}")

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\n[Done] Results saved to: {output_dir}")
    print(f"  - per_video.csv          ({len(per_video_rows)} rows)")
    print(f"  - resource_samples.csv   ({len(samples)} samples)")
    print(f"  - environment.json")
    print(f"  - summary.json")
    print(f"  - summary.txt")


if __name__ == "__main__":
    main()
