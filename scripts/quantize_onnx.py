#!/usr/bin/env python3
"""Quantize an ONNX TextureSBI model to INT8 for optimized ARM NEON CPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static, CalibrationDataReader
except ImportError as e:
    raise ImportError("onnxruntime is required for quantization. Run: pip install onnxruntime") from e

# Import processing logic for calibration data if doing static quantization
# Adjust sys.path to find qalf and scripts modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class QALFCalibrationDataReader(CalibrationDataReader):
    def __init__(self, video_paths: list[str], input_name: str = "texture", num_frames: int = 32, texture_frames: int = 8, image_size: int = 256):
        self.video_paths = video_paths
        self.input_name = input_name
        self.num_frames = num_frames
        self.texture_frames = texture_frames
        self.image_size = image_size
        self.data_iter = iter(self._generate_batches())

    def _generate_batches(self):
        import cv2
        import numpy as np
        from scripts.infer_video import OpenCVYuNetLandmarker, compute_protocol_frame_indices, extract_video_frames, align_face

        print(f"[Calib] Initializing YuNet landmarker for calibration...", flush=True)
        detector = OpenCVYuNetLandmarker(score_threshold=0.5)

        for vpath_str in self.video_paths:
            vpath = Path(vpath_str)
            if not vpath.exists():
                print(f"[Calib] Skipping missing video: {vpath}")
                continue

            print(f"[Calib] Extracting features from: {vpath.name}...")
            cap = cv2.VideoCapture(str(vpath))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if total_frames <= 0 or fps <= 0:
                continue

            unique_indices, clip_groups = compute_protocol_frame_indices(
                total_frames, fps, target_fps=10.0, num_frames=self.num_frames, texture_frames=self.texture_frames, clips_per_video=3
            )
            frames, _ = extract_video_frames(vpath, unique_indices)

            for group in clip_groups:
                crops = []
                for idx in group:
                    if idx not in frames:
                        continue
                    image_rgb = frames[idx]
                    landmarks = detector.process(image_rgb, timestamp_ms=0)
                    
                    if landmarks is not None and landmarks.shape[0] == 5:
                        crop, _ = align_face(image_rgb, landmarks)
                    else:
                        # Fallback simple crop if no face detected
                        h, w = image_rgb.shape[:2]
                        crop = image_rgb[max(0, h//2-80):min(h, h//2+80), max(0, w//2-80):min(w, w//2+80)]
                    
                    if crop.shape[0] != self.image_size or crop.shape[1] != self.image_size:
                        crop = cv2.resize(crop, (self.image_size, self.image_size))
                    
                    crop = crop.astype(np.float32) / 255.0
                    crop = crop.transpose(2, 0, 1) # HWC to CHW
                    
                    # ImageNet normalization
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    crop = (crop - mean) / std
                    crops.append(crop)
                
                if len(crops) == self.texture_frames:
                    batch = np.stack(crops, axis=0) # (8, 3, 256, 256)
                    # Add batch dimension to match ONNX graph: (1, 8, 3, 256, 256)
                    # Actually wait, export_onnx exported with dynamic batch. 
                    # If model expects (B, 3, 256, 256) we provide that.
                    # BUT export_onnx has: dynamic_axes={"texture": {0: "batch", 1: "texture_frames"}}
                    # No, `batch` in infer_video is passed directly as `eval_batch`.
                    # In infer_video.py: `eval_batch = np.stack(crops, axis=0)`. Shape is (8, 3, 256, 256).
                    # Let's yield exactly what infer_video provides.
                    batch_f32 = batch.astype(np.float32)
                    yield {self.input_name: batch_f32}
        
        print("[Calib] Calibration data extraction completed.", flush=True)

    def get_next(self):
        return next(self.data_iter, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8 for Raspberry Pi 4 ARM CPU")
    parser.add_argument("--input", required=True, help="Path to input FP32 ONNX model (e.g. models/qalf.onnx)")
    parser.add_argument("--output", default=None, help="Path to output INT8 ONNX model (default: auto)")
    parser.add_argument("--method", choices=["dynamic", "static"], default="dynamic", help="Quantization method (dynamic for Linear, static for CNNs)")
    parser.add_argument("--calib-videos", nargs="+", default=[], help="List of videos to use for Static Quantization calibration")
    parser.add_argument("--image-size", type=int, default=256, help="Image size for calibration data")
    parser.add_argument("--per-channel", action="store_true", default=True, help="Per-channel weight quantization")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    suffix = f"_{args.method}_int8.onnx"
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}{suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Quantizing FP32 model '{input_path.name}' to INT8 using {args.method.upper()} method...")

    if args.method == "dynamic":
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            reduce_range=False,
            extra_options={"DisableShapeInference": True},
        )
    elif args.method == "static":
        if not args.calib_videos:
            raise ValueError("Static quantization requires at least one video passed via --calib-videos")
        
        dr = QALFCalibrationDataReader(video_paths=args.calib_videos, image_size=args.image_size)
        
        quantize_static(
            model_input=str(input_path),
            model_output=str(output_path),
            calibration_data_reader=dr,
            quant_format=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            reduce_range=False,
            extra_options={"DisableShapeInference": True},
        )

    orig_size_mb = input_path.stat().st_size / (1024 * 1024)
    quant_size_mb = output_path.stat().st_size / (1024 * 1024)
    reduction = (1.0 - quant_size_mb / orig_size_mb) * 100.0

    print(f"[+] Quantization successful!")
    print(f"    - Original FP32 Model : {orig_size_mb:.2f} MB")
    print(f"    - Quantized INT8 Model: {quant_size_mb:.2f} MB ({reduction:.1f}% size reduction)")
    print(f"    - Output saved to     : {output_path}")

    # Copy companion JSON metadata if present
    input_json = input_path.with_suffix(".json")
    output_json = output_path.with_suffix(".json")
    if input_json.is_file():
        with open(input_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["precision"] = f"INT8 ({args.method.upper()})"
        meta["output_onnx"] = str(output_path)
        meta["bytes"] = output_path.stat().st_size
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[+] Companion metadata updated: {output_json}")


if __name__ == "__main__":
    main()
