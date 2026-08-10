#!/usr/bin/env python3
"""Pilot IMAGE versus VIDEO Face Landmarker modes on the same extracted frames."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.geometry import DEFAULT_LANDMARK_INDICES, build_geometry_features
from qalf.data.landmarks import FaceLandmarkerExtractor, _video_timestamps_ms
from qalf.data.manifest import VideoRecord, load_manifest


def _stratified_sample(records: list[VideoRecord], count: int, seed: int) -> list[VideoRecord]:
    groups: defaultdict[tuple[int, str], list[VideoRecord]] = defaultdict(list)
    for record in records:
        groups[(record.label, record.method)].append(record)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    selected: list[VideoRecord] = []
    ordered_groups = sorted(groups)
    while len(selected) < min(count, len(records)):
        progressed = False
        for key in ordered_groups:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _extract_record(
    record: VideoRecord,
    frame_root: Path,
    extractor: FaceLandmarkerExtractor,
    running_mode: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    timestamps_ms = _video_timestamps_ms(record)
    landmarks: list[np.ndarray] = []
    detected: list[bool] = []
    started = time.perf_counter()
    for order, relative in enumerate(record.frames):
        image = cv2.imread(str(frame_root / relative))
        if image is None:
            raise FileNotFoundError(frame_root / relative)
        points = extractor.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            timestamps_ms[order] if running_mode == "video" else None,
        )
        if points is None:
            landmarks.append(np.full((468, 3), np.nan, dtype=np.float32))
            detected.append(False)
        else:
            landmarks.append(points)
            detected.append(True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return np.stack(landmarks), np.asarray(detected), elapsed_ms


def _motion_metrics(
    landmarks: np.ndarray,
    detected: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, float]:
    features, quality = build_geometry_features(
        landmarks,
        detected,
        timestamps,
        feature_mode="motion_3d",
    )
    point_count = len(DEFAULT_LANDMARK_INDICES)
    velocity = features[:, : point_count * 3]
    acceleration = features[:, point_count * 3 : point_count * 6]
    return {
        "detected_ratio": float(detected.mean()),
        "mean_abs_velocity": float(np.abs(velocity).mean()),
        "mean_abs_acceleration": float(np.abs(acceleration).mean()),
        "alignment_residual": float(quality[2]),
        "scale_cv": float(quality[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-videos", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()
    if args.sample_videos < 1:
        parser.error("--sample-videos must be positive")

    records = _stratified_sample(load_manifest(args.manifest), args.sample_videos, args.seed)
    frame_root = Path(args.frame_root)
    rows: list[dict[str, object]] = []
    image_extractor = FaceLandmarkerExtractor(
        args.model_path,
        "image",
        args.min_confidence,
    )
    try:
        for record in records:
            image_landmarks, image_detected, image_ms = _extract_record(
                record,
                frame_root,
                image_extractor,
                "image",
            )
            video_extractor = FaceLandmarkerExtractor(
                args.model_path,
                "video",
                args.min_confidence,
            )
            try:
                video_landmarks, video_detected, video_ms = _extract_record(
                    record,
                    frame_root,
                    video_extractor,
                    "video",
                )
            finally:
                video_extractor.close()
            timestamps = np.asarray(record.timestamps_sec, dtype=np.float32)
            image_metrics = _motion_metrics(image_landmarks, image_detected, timestamps)
            video_metrics = _motion_metrics(video_landmarks, video_detected, timestamps)
            rows.append(
                {
                    "dataset": record.dataset,
                    "method": record.method,
                    "label": record.label,
                    "video_id": record.video_id,
                    "image_ms_per_frame": image_ms / len(record.frames),
                    "video_ms_per_frame": video_ms / len(record.frames),
                    **{f"image_{key}": value for key, value in image_metrics.items()},
                    **{f"video_{key}": value for key, value in video_metrics.items()},
                }
            )
    finally:
        image_extractor.close()

    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    summary = {
        key: {
            "mean": statistics.mean(float(row[key]) for row in rows),
            "median": statistics.median(float(row[key]) for row in rows),
        }
        for key in numeric
        if key != "label"
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "sample_videos": len(rows),
        "seed": args.seed,
        "summary": summary,
        "videos": rows,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"sample_videos": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
