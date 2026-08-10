#!/usr/bin/env python3
"""Extract the canonical 64-frame main-face sequences on a local machine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.extraction import (
    DEFAULT_FFPP_METHODS,
    ExtractionConfig,
    build_celebdf_test_specs,
    build_ffpp_specs,
    extract_specs,
    is_frame_qc_exclusion,
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frames-per-video", type=int, default=64)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--mtcnn-batch-size", type=int, default=8)
    parser.add_argument("--min-face-size", type=int, default=64)
    parser.add_argument("--min-detection-probability", type=float, default=0.90)
    parser.add_argument("--min-direct-detection-ratio", type=float, default=0.75)
    parser.add_argument("--max-consecutive-missing", type=int, default=6)
    parser.add_argument("--square-margin", type=float, default=0.35)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-crop-side", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return an error for documented face-QC exclusions as well as unexpected failures.",
    )


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _config(args: argparse.Namespace) -> ExtractionConfig:
    return ExtractionConfig(
        frames_per_video=args.frames_per_video,
        target_fps=args.target_fps,
        mtcnn_batch_size=args.mtcnn_batch_size,
        min_face_size=args.min_face_size,
        min_detection_probability=args.min_detection_probability,
        min_direct_detection_ratio=args.min_direct_detection_ratio,
        max_consecutive_missing=args.max_consecutive_missing,
        square_margin=args.square_margin,
        output_size=args.output_size,
        jpeg_quality=args.jpeg_quality,
        min_crop_side=args.min_crop_side,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    ffpp = subparsers.add_parser("ffpp", help="Extract official FF++ train/val/test splits")
    ffpp.add_argument("--dataset-root", required=True)
    ffpp.add_argument("--split-root", required=True)
    ffpp.add_argument("--compression", default="c23")
    ffpp.add_argument("--methods", nargs="+", default=list(DEFAULT_FFPP_METHODS))
    ffpp.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    _add_common_arguments(ffpp)

    celebdf = subparsers.add_parser("celebdf", help="Extract official Celeb-DF-v2 test list")
    celebdf.add_argument("--dataset-root", required=True)
    celebdf.add_argument("--test-list")
    _add_common_arguments(celebdf)

    args = parser.parse_args()
    output_root = Path(args.output_root)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    config = _config(args)

    if args.dataset == "ffpp":
        specs_by_split = build_ffpp_specs(
            args.dataset_root,
            args.split_root,
            methods=args.methods,
            compression=args.compression,
        )
        jobs = [
            (
                split,
                specs_by_split[split],
                manifest_root / f"ffpp_{split}.jsonl",
            )
            for split in args.splits
        ]
    else:
        jobs = [
            (
                "test",
                build_celebdf_test_specs(args.dataset_root, args.test_list),
                manifest_root / "celebdf_test.jsonl",
            )
        ]

    all_errors: list[dict[str, str]] = []
    summaries: dict[str, dict[str, object]] = {}
    for split, specs, manifest_path in jobs:
        records, errors = extract_specs(
            specs,
            output_root,
            manifest_path,
            config,
            device=device,
            fail_fast=False,
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
        )
        all_errors.extend(errors)
        summaries[split] = {
            "requested": len(specs),
            "output": len(records),
            "errors": len(errors),
            "manifest": str(manifest_path),
        }
        print(
            f"{split}: requested={len(specs)}, output={len(records)}, "
            f"errors={len(errors)}, manifest={manifest_path}"
        )

    qc_exclusions = [error for error in all_errors if is_frame_qc_exclusion(error)]
    unexpected_errors = [error for error in all_errors if not is_frame_qc_exclusion(error)]
    report = {
        "dataset": args.dataset,
        "device": device,
        "splits": summaries,
        "documented_qc_exclusions": qc_exclusions,
        "unexpected_errors": unexpected_errors,
        "config": config.__dict__,
    }
    report_path = output_root / "frame_extraction_qc.json"
    temporary = report_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(report_path)

    print(f"Documented frame-QC exclusions: {len(qc_exclusions)}")
    print(f"Unexpected extraction errors: {len(unexpected_errors)}")
    print(f"QC report: {report_path}")
    if unexpected_errors or (args.strict and qc_exclusions):
        raise RuntimeError(
            f"Frame extraction failed with {len(unexpected_errors)} unexpected errors and "
            f"{len(qc_exclusions)} documented QC exclusions"
        )
    print("FRAME EXTRACTION PASSED")


if __name__ == "__main__":
    main()
