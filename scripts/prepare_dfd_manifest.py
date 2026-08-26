#!/usr/bin/env python3
"""Merge and rebase flattened DFD real/fake exports for evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dfd import prepare_dfd_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge enriched DFD real/fake manifests and rewrite frame/landmark paths "
            "relative to a common dataset root. All referenced files are validated."
        )
    )
    parser.add_argument("--real-manifest", required=True)
    parser.add_argument("--fake-manifest", required=True)
    parser.add_argument("--real-frame-dir", required=True)
    parser.add_argument("--fake-frame-dir", required=True)
    parser.add_argument("--real-landmark-dir", required=True)
    parser.add_argument("--fake-landmark-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    report = prepare_dfd_manifest(
        real_manifest=args.real_manifest,
        fake_manifest=args.fake_manifest,
        real_frame_directory=args.real_frame_dir,
        fake_frame_directory=args.fake_frame_dir,
        real_landmark_directory=args.real_landmark_dir,
        fake_landmark_directory=args.fake_landmark_dir,
        dataset_root=args.dataset_root,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("DFD MANIFEST PREPARATION PASSED")


if __name__ == "__main__":
    main()
