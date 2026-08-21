#!/usr/bin/env python3
"""Create a strided manifest/landmark view without copying extracted JPEGs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.subsample import derive_strided_landmark_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--input-landmark-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-landmark-root", required=True)
    parser.add_argument("--source-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    report = derive_strided_landmark_dataset(
        args.input_manifest,
        args.input_landmark_root,
        args.output_manifest,
        args.output_landmark_root,
        source_frames=args.source_frames,
        stride=args.stride,
        offset=args.offset,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
