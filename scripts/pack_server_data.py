#!/usr/bin/env python3
"""Package server-extracted Celeb-DF frames + landmarks for transfer to Raspberry Pi 4.

Usage:
    python scripts/pack_server_data.py \
        --extracted-root /path/to/celebdf_extracted \
        --output celebdf_server_extracted.tar.gz
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack server-extracted frames + landmarks + manifests into a tar.gz archive"
    )
    parser.add_argument(
        "--extracted-root",
        required=True,
        help="Root directory of extracted data (contains frames/, landmarks/, manifests/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output tar.gz path (default: celebdf_server_extracted.tar.gz in current dir)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Create uncompressed .tar instead of .tar.gz (faster, larger)",
    )
    args = parser.parse_args()

    root = Path(args.extracted_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Extracted root not found: {root}")

    required_dirs = ["frames", "landmarks", "manifests"]
    for name in required_dirs:
        child = root / name
        if not child.is_dir():
            raise FileNotFoundError(f"Required subdirectory not found: {child}")

    manifest_candidates = list((root / "manifests").glob("*_landmarks.jsonl"))
    if not manifest_candidates:
        raise FileNotFoundError("No *_landmarks.jsonl manifest found in manifests/")

    extension = ".tar" if args.no_compress else ".tar.gz"
    output = Path(args.output) if args.output else Path(f"celebdf_server_extracted{extension}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.no_compress else "w:gz"

    # Count files to pack
    frame_count = sum(1 for _ in (root / "frames").rglob("*.jpg"))
    landmark_count = sum(1 for _ in (root / "landmarks").rglob("*.npz"))
    manifest_count = sum(1 for _ in (root / "manifests").iterdir() if _.is_file())

    print("=" * 72)
    print("  PACK SERVER-EXTRACTED DATA FOR PI4 TRANSFER")
    print("=" * 72)
    print(f"Source Root  : {root}")
    print(f"Output       : {output}")
    print(f"Compression  : {'none' if args.no_compress else 'gzip'}")
    print(f"Frames       : {frame_count} JPEG files")
    print(f"Landmarks    : {landmark_count} NPZ files")
    print(f"Manifests    : {manifest_count} files")
    print("=" * 72)

    if frame_count == 0:
        raise RuntimeError("No JPEG frames found — nothing to pack")
    if landmark_count == 0:
        raise RuntimeError("No NPZ landmark caches found — nothing to pack")

    packed = 0
    with tarfile.open(str(output), mode) as tar:
        # Pack manifests first (small, needed for verification)
        for manifest_file in sorted((root / "manifests").iterdir()):
            if manifest_file.is_file():
                arcname = str(manifest_file.relative_to(root))
                tar.add(str(manifest_file), arcname=arcname)
                packed += 1

        # Pack QC report if present
        qc_report = root / "frame_extraction_qc.json"
        if qc_report.is_file():
            tar.add(str(qc_report), arcname="frame_extraction_qc.json")
            packed += 1

        # Pack frames
        print(f"\nPacking {frame_count} frames...", flush=True)
        for idx, frame_path in enumerate(sorted((root / "frames").rglob("*.jpg")), 1):
            arcname = str(frame_path.relative_to(root))
            tar.add(str(frame_path), arcname=arcname)
            packed += 1
            if idx % 5000 == 0:
                print(f"  ... {idx}/{frame_count} frames packed", flush=True)

        # Pack landmarks
        print(f"\nPacking {landmark_count} landmark caches...", flush=True)
        for landmark_path in sorted((root / "landmarks").rglob("*.npz")):
            arcname = str(landmark_path.relative_to(root))
            tar.add(str(landmark_path), arcname=arcname)
            packed += 1

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 72}")
    print(f"[DONE] Packed {packed} files → {output}")
    print(f"       Archive size: {size_mb:.1f} MB")
    print(f"{'=' * 72}")
    print(f"\nTransfer to Pi4:")
    print(f"  scp {output.name} pi@<PI4_IP>:/mnt/usb_data/")
    print(f"\nOn Pi4:")
    print(f"  cd /mnt/usb_data")
    print(f"  mv extracted_celebdf extracted_celebdf_old")
    print(f"  mkdir extracted_celebdf && cd extracted_celebdf")
    print(f"  tar x{'z' if not args.no_compress else ''}f ../{output.name}")


if __name__ == "__main__":
    main()
