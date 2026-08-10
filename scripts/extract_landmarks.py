#!/usr/bin/env python3
"""Extract MediaPipe landmark caches and emit an enriched manifest."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.landmarks import (
    ensure_face_landmarker_model,
    extract_landmark_cache,
    is_landmark_qc_exclusion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--model-path",
        default="models/face_landmarker.task",
        help="MediaPipe Face Landmarker .task model; downloaded and verified if absent.",
    )
    parser.add_argument("--no-download-model", action="store_true")
    parser.add_argument(
        "--video-mode",
        action="store_true",
        help="Use VIDEO tracking; IMAGE mode remains the reproducible default.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-detected-ratio", type=float, default=0.75)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return an error for documented detection-ratio exclusions too.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    model_path = ensure_face_landmarker_model(
        args.model_path,
        download=not args.no_download_model,
    )
    report = extract_landmark_cache(
        args.manifest,
        args.frame_root,
        args.landmark_root,
        args.output_manifest,
        model_path,
        running_mode="video" if args.video_mode else "image",
        min_confidence=args.min_confidence,
        min_detected_ratio=args.min_detected_ratio,
        fail_fast=False,
        resume=not args.no_resume,
        checkpoint_every=args.checkpoint_every,
    )
    errors = report["errors"]
    qc_exclusions = [error for error in errors if is_landmark_qc_exclusion(error)]
    unexpected_errors = [error for error in errors if not is_landmark_qc_exclusion(error)]
    print(report)
    print(f"Documented landmark-QC exclusions: {len(qc_exclusions)}")
    print(f"Unexpected landmark extraction errors: {len(unexpected_errors)}")
    if unexpected_errors or (args.strict and qc_exclusions):
        raise RuntimeError(
            f"Landmark extraction failed with {len(unexpected_errors)} unexpected errors and "
            f"{len(qc_exclusions)} documented QC exclusions"
        )
    print("LANDMARK EXTRACTION PASSED")


if __name__ == "__main__":
    main()
