#!/usr/bin/env python3
"""Verify ONNX model output parity against PyTorch checkpoint.

Loads the same pre-extracted frames and landmarks, runs inference through both
the PyTorch model and the ONNX model, and reports per-video score differences.

Usage:
    python scripts/verify_onnx_parity.py \
        --checkpoint path/to/best.pt \
        --onnx models/qalf.onnx \
        --manifest path/to/celebdf_test_landmarks.jsonl \
        --frame-root path/to/extracted/ \
        --landmark-root path/to/extracted/landmarks/ \
        --num-samples 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import IMAGE_MEAN, IMAGE_STD, _aligned_full_face, _clip_indices
from qalf.data.manifest import load_manifest
from qalf.models import build_model_from_checkpoint


def build_clip_tensor(
    record,
    frame_root: Path,
    landmark_root: Path,
    num_frames: int,
    texture_frames: int,
    image_size: int,
    clip_index: int = 0,
    clips_per_video: int = 1,
) -> np.ndarray:
    """Build a single clip tensor exactly matching QALFVideoDataset.__getitem__."""
    with np.load(landmark_root / str(record.landmark_path)) as cache:
        landmarks = cache["landmarks"].copy()
        detected = cache["detected"].copy()

    clip = _clip_indices(
        len(record.frames), num_frames, training=False,
        clip_index=clip_index, clips_per_video=clips_per_video,
    )
    texture_positions = np.rint(
        np.linspace(0, len(clip) - 1, texture_frames)
    ).astype(np.int64)

    tensors = []
    for position in texture_positions:
        source_index = int(clip[position])
        image_bgr = cv2.imread(str(frame_root / record.frames[source_index]))
        if image_bgr is None:
            raise FileNotFoundError(frame_root / record.frames[source_index])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        canonical, _ = _aligned_full_face(
            image_rgb, landmarks[source_index],
            bool(detected[source_index]), image_size,
        )
        normalized = canonical.astype(np.float32) / 255.0
        normalized = (normalized - IMAGE_MEAN) / IMAGE_STD
        tensors.append(normalized.transpose(2, 0, 1))

    return np.stack(tensors, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ONNX vs PyTorch parity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--landmark-root", required=True)
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of videos to compare (0 = all)")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--texture-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--clips-per-video", type=int, default=3)
    parser.add_argument("--flip-tta", action="store_true", default=True)
    parser.add_argument("--no-flip-tta", action="store_true")
    args = parser.parse_args()

    flip_tta = not args.no_flip_tta
    frame_root = Path(args.frame_root)
    landmark_root = Path(args.landmark_root)

    # Load PyTorch model
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint).eval()

    # Load ONNX model
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    onnx_session = ort.InferenceSession(args.onnx, sess_options, providers=["CPUExecutionProvider"])
    input_name = onnx_session.get_inputs()[0].name

    # Load manifest
    records = load_manifest(args.manifest)
    if args.num_samples > 0:
        records = records[:args.num_samples]

    print("=" * 72)
    print("  ONNX vs PyTorch PARITY VERIFICATION")
    print("=" * 72)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"ONNX Model  : {args.onnx}")
    print(f"Videos      : {len(records)}")
    print(f"Protocol    : {args.clips_per_video} clips × {args.texture_frames}f, flip_tta={flip_tta}")
    print("=" * 72)

    diffs_logit = []
    diffs_score = []
    max_diff_video = None
    max_diff_val = 0.0

    for idx, record in enumerate(records):
        try:
            clip_scores_pt = []
            clip_scores_onnx = []

            for clip_idx in range(args.clips_per_video):
                clip_array = build_clip_tensor(
                    record, frame_root, landmark_root,
                    args.num_frames, args.texture_frames, args.image_size,
                    clip_idx, args.clips_per_video,
                )
                batch = clip_array[np.newaxis, ...]  # (1, T, 3, H, W)

                # PyTorch forward
                with torch.inference_mode():
                    pt_out = model({"texture": torch.from_numpy(batch)})
                    pt_logit = float(pt_out["logit"].squeeze().cpu().numpy())
                    pt_score = float(torch.sigmoid(pt_out["logit"]).squeeze().cpu().numpy())

                # ONNX forward
                onnx_logit = float(onnx_session.run(None, {input_name: batch})[0].squeeze())
                onnx_score = float(1.0 / (1.0 + np.exp(-onnx_logit)))

                # TTA
                if flip_tta:
                    flipped = batch[:, :, :, :, ::-1].copy()
                    with torch.inference_mode():
                        pt_flip_out = model({"texture": torch.from_numpy(flipped)})
                        pt_flip_score = float(torch.sigmoid(pt_flip_out["logit"]).squeeze().cpu().numpy())
                    onnx_flip_logit = float(onnx_session.run(None, {input_name: flipped})[0].squeeze())
                    onnx_flip_score = float(1.0 / (1.0 + np.exp(-onnx_flip_logit)))

                    pt_score = 0.5 * (pt_score + pt_flip_score)
                    onnx_score = 0.5 * (onnx_score + onnx_flip_score)

                clip_scores_pt.append(pt_score)
                clip_scores_onnx.append(onnx_score)
                diffs_logit.append(abs(pt_logit - onnx_logit))

            # Aggregate
            final_pt = float(np.mean(clip_scores_pt))
            final_onnx = float(np.mean(clip_scores_onnx))
            diff = abs(final_pt - final_onnx)
            diffs_score.append(diff)

            label_str = "REAL" if record.label == 0 else "FAKE"
            status = "OK" if diff < 1e-4 else "WARN" if diff < 1e-2 else "FAIL"
            print(
                f"[{idx+1:3d}/{len(records)}] [{status:4s}] {record.video_id:35s} "
                f"[{label_str}] PT={final_pt:.6f} ONNX={final_onnx:.6f} diff={diff:.2e}"
            )

            if diff > max_diff_val:
                max_diff_val = diff
                max_diff_video = record.video_id

        except Exception as e:
            print(f"[{idx+1:3d}/{len(records)}] [ERR ] {record.video_id}: {e}")

    if not diffs_score:
        print("\nNo videos processed successfully.")
        return

    diffs_score = np.array(diffs_score)
    diffs_logit = np.array(diffs_logit)

    print(f"\n{'=' * 72}")
    print("  PARITY SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Videos compared  : {len(diffs_score)}")
    print(f"  Score diff mean  : {diffs_score.mean():.2e}")
    print(f"  Score diff max   : {diffs_score.max():.2e} ({max_diff_video})")
    print(f"  Score diff p95   : {np.percentile(diffs_score, 95):.2e}")
    print(f"  Logit diff mean  : {diffs_logit.mean():.2e}")
    print(f"  Logit diff max   : {diffs_logit.max():.2e}")

    if diffs_score.max() < 1e-4:
        print(f"\n  ✅ PASS: ONNX model is numerically equivalent to PyTorch")
    elif diffs_score.max() < 1e-2:
        print(f"\n  ⚠️  WARN: Minor numerical differences detected (max={diffs_score.max():.2e})")
    else:
        print(f"\n  ❌ FAIL: Significant divergence between ONNX and PyTorch")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
