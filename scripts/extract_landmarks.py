#!/usr/bin/env python3
"""Extract MediaPipe landmark caches and emit an enriched manifest."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.landmarks import extract_landmark_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--model-path",
        required=True,
        help="MediaPipe Face Landmarker .task model.",
    )
    parser.add_argument(
        "--video-mode",
        action="store_true",
        help="Use VIDEO tracking; IMAGE mode remains the reproducible default.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-detected-ratio", type=float, default=0.75)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    report = extract_landmark_cache(
        args.manifest,
        args.frame_root,
        args.landmark_root,
        args.output_manifest,
        args.model_path,
        running_mode="video" if args.video_mode else "image",
        min_confidence=args.min_confidence,
        min_detected_ratio=args.min_detected_ratio,
        fail_fast=not args.skip_errors,
        resume=not args.no_resume,
        checkpoint_every=args.checkpoint_every,
    )
    print(report)


if __name__ == "__main__":
    main()
