#!/usr/bin/env python3
"""Verify ONNX model output parity against PyTorch checkpoint.

Loads the same pre-extracted frames and landmarks, runs inference through both
the PyTorch model and the ONNX model, and reports per-video score differences
plus label-flip statistics at a given decision threshold.

Outputs:
  - parity_results.csv   Per-video scores and comparisons
  - parity_summary.json  Aggregate statistics
  - stdout log           Human-readable summary

Usage (50 samples):
    python scripts/verify_onnx_parity.py \\
        --checkpoint path/to/best.pt \\
        --onnx models/qalf.onnx \\
        --manifest path/to/celebdf_test_landmarks.jsonl \\
        --frame-root path/to/extracted/ \\
        --landmark-root path/to/landmarks/ \\
        --num-samples 50 \\
        --threshold 0.671156 \\
        --output-dir eval_pi4_onnx_parity_n50

Usage (all videos):
    python scripts/verify_onnx_parity.py \\
        --checkpoint path/to/best.pt \\
        --onnx models/qalf.onnx \\
        --manifest path/to/celebdf_test_landmarks.jsonl \\
        --frame-root path/to/extracted/ \\
        --landmark-root path/to/landmarks/ \\
        --num-samples 0 \\
        --threshold 0.671156 \\
        --output-dir eval_pi4_onnx_parity_all
"""

from __future__ import annotations

import argparse
import csv
import json
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
    parser = argparse.ArgumentParser(
        description="Verify ONNX vs PyTorch parity with label-flip analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model (.onnx)")
    parser.add_argument("--manifest", required=True, help="Path to test manifest (.jsonl)")
    parser.add_argument("--frame-root", required=True, help="Root directory for extracted frames")
    parser.add_argument("--landmark-root", required=True, help="Root directory for landmark caches")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of videos to compare (0 = all)")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--texture-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--clips-per-video", type=int, default=3)
    parser.add_argument("--flip-tta", action="store_true", default=True)
    parser.add_argument("--no-flip-tta", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.671156,
                        help="Decision threshold for label-flip analysis (default: 0.671156)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save parity_results.csv and parity_summary.json")
    args = parser.parse_args()

    flip_tta = not args.no_flip_tta
    frame_root = Path(args.frame_root)
    landmark_root = Path(args.landmark_root)
    threshold = args.threshold

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    print(f"Threshold   : {threshold:.6f} (for label-flip analysis)")
    if output_dir:
        print(f"Output      : {output_dir}")
    print("=" * 72)

    diffs_logit: list[float] = []
    diffs_score: list[float] = []
    max_diff_video: str | None = None
    max_diff_val = 0.0
    label_flips = 0
    n_errors = 0

    # Per-video results for CSV
    csv_rows: list[dict] = []

    for idx, record in enumerate(records):
        try:
            clip_scores_pt: list[float] = []
            clip_scores_onnx: list[float] = []

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

            # Label-flip analysis
            pred_pt = 1 if final_pt >= threshold else 0
            pred_onnx = 1 if final_onnx >= threshold else 0
            flipped_label = pred_pt != pred_onnx
            if flipped_label:
                label_flips += 1

            label_str = "REAL" if record.label == 0 else "FAKE"
            status = "OK" if diff < 1e-4 else "WARN" if diff < 1e-2 else "FAIL"
            flip_marker = " *** FLIP ***" if flipped_label else ""
            print(
                f"[{idx+1:3d}/{len(records)}] [{status:4s}] {record.video_id:35s} "
                f"[{label_str}] PT={final_pt:.6f} ONNX={final_onnx:.6f} diff={diff:.2e}{flip_marker}"
            )

            if diff > max_diff_val:
                max_diff_val = diff
                max_diff_video = record.video_id

            csv_rows.append({
                "video_id": record.video_id,
                "label": record.label,
                "score_pytorch": final_pt,
                "score_onnx": final_onnx,
                "score_diff": diff,
                "pred_pytorch": pred_pt,
                "pred_onnx": pred_onnx,
                "label_flipped": int(flipped_label),
            })

        except Exception as e:
            n_errors += 1
            print(f"[{idx+1:3d}/{len(records)}] [ERR ] {record.video_id}: {e}")
            csv_rows.append({
                "video_id": record.video_id,
                "label": record.label,
                "score_pytorch": "",
                "score_onnx": "",
                "score_diff": "",
                "pred_pytorch": "",
                "pred_onnx": "",
                "label_flipped": "",
            })

    if not diffs_score:
        print("\nNo videos processed successfully.")
        return

    diffs_score_arr = np.array(diffs_score)
    diffs_logit_arr = np.array(diffs_logit)

    # ---- Summary ----
    summary = {
        "videos_compared": len(diffs_score),
        "videos_error": n_errors,
        "threshold": threshold,
        "score_diff": {
            "mean": float(diffs_score_arr.mean()),
            "max": float(diffs_score_arr.max()),
            "p95": float(np.percentile(diffs_score_arr, 95)),
            "p99": float(np.percentile(diffs_score_arr, 99)),
            "max_video": max_diff_video,
        },
        "logit_diff": {
            "mean": float(diffs_logit_arr.mean()),
            "max": float(diffs_logit_arr.max()),
        },
        "label_flip_count": label_flips,
        "label_flip_rate": round(label_flips / len(diffs_score), 4) if diffs_score else 0,
        "verdict": (
            "PASS" if diffs_score_arr.max() < 1e-4
            else "WARN" if diffs_score_arr.max() < 1e-2
            else "FAIL"
        ),
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "onnx": str(args.onnx),
            "manifest": str(args.manifest),
            "clips_per_video": args.clips_per_video,
            "texture_frames": args.texture_frames,
            "flip_tta": flip_tta,
        },
    }

    print(f"\n{'=' * 72}")
    print("  PARITY SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Videos compared  : {summary['videos_compared']}")
    print(f"  Errors           : {summary['videos_error']}")
    print(f"  Score diff mean  : {summary['score_diff']['mean']:.2e}")
    print(f"  Score diff max   : {summary['score_diff']['max']:.2e} ({max_diff_video})")
    print(f"  Score diff P95   : {summary['score_diff']['p95']:.2e}")
    print(f"  Score diff P99   : {summary['score_diff']['p99']:.2e}")
    print(f"  Logit diff mean  : {summary['logit_diff']['mean']:.2e}")
    print(f"  Logit diff max   : {summary['logit_diff']['max']:.2e}")
    print(f"  Label flips      : {label_flips} / {len(diffs_score)} (at threshold={threshold:.6f})")

    if summary["verdict"] == "PASS":
        print(f"\n  ✅ PASS: ONNX model is numerically equivalent to PyTorch")
    elif summary["verdict"] == "WARN":
        print(f"\n  ⚠️  WARN: Minor numerical differences detected (max={diffs_score_arr.max():.2e})")
    else:
        print(f"\n  ❌ FAIL: Significant divergence between ONNX and PyTorch")

    if label_flips > 0:
        print(f"  ⚠️  {label_flips} video(s) changed prediction at threshold {threshold:.6f}")
    else:
        print(f"  ✅ No label flips at threshold {threshold:.6f}")

    print(f"{'=' * 72}")

    # ---- Save outputs ----
    if output_dir:
        csv_fields = ["video_id", "label", "score_pytorch", "score_onnx",
                       "score_diff", "pred_pytorch", "pred_onnx", "label_flipped"]
        with open(output_dir / "parity_results.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)

        with open(output_dir / "parity_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

        print(f"\n[Done] Results saved to: {output_dir}")
        print(f"  - parity_results.csv   ({len(csv_rows)} rows)")
        print(f"  - parity_summary.json")


if __name__ == "__main__":
    main()
