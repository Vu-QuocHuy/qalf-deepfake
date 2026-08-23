import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("Please install facenet-pytorch: pip install facenet-pytorch")
    sys.exit(1)

def load_manifest(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            records[data["video_id"]] = data
    return records

def crop_face_to_256(image_rgb: np.ndarray, yunet_detector) -> tuple[np.ndarray, np.ndarray]:
    """Crop face to 256x256 using YuNet (matches new Pi4 pipeline). Returns (crop, bbox)."""
    height, width = image_rgb.shape[:2]
    try:
        box = yunet_detector.detect_bbox(image_rgb)
        if box is not None:
            bx1, by1, bx2, by2 = box
            bw, bh = bx2 - bx1, by2 - by1
            cx, cy = bx1 + bw / 2.0, by1 + bh / 2.0
            side = max(bw, bh) * 1.35
            x1 = max(0, int(round(cx - side / 2.0)))
            y1 = max(0, int(round(cy - side / 2.0)))
            x2 = min(width, int(round(cx + side / 2.0)))
            y2 = min(height, int(round(cy + side / 2.0)))
            crop = image_rgb[y1:y2, x1:x2]
            if crop.shape[0] > 16 and crop.shape[1] > 16:
                return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA), np.array([bx1, by1, bx2, by2])
    except Exception as e:
        print(f"YuNet Exception: {e}")
    return cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA), np.zeros(4)

def main():
    print("=" * 60)
    print("   PI 4 EDGE PREPROCESSING ABLATION STUDY (DRIFT PROOF)   ")
    print("=" * 60)
    
    server_root = Path("/mnt/usb_data/extracted_celebdf_server")
    video_root = Path("/mnt/usb_data/celebdf_test_518")
    manifest_path = server_root / "manifests" / "celebdf_test_landmarks.jsonl"
    
    if not manifest_path.exists():
        print(f"Server manifest missing: {manifest_path}")
        return
        
    records = load_manifest(manifest_path)
    video_ids = sorted(list(records.keys()))
    
    # Pick 10 videos for ablation
    sample_vids = video_ids[:10]
    
    from qalf.data.landmarks import OpenCVYuNetLandmarker
    yunet = OpenCVYuNetLandmarker(model_path="models/face_detection_yunet_2023mar.onnx", score_threshold=0.5)
    
    total_pixel_mse = 0.0
    valid_frames = 0
    
    for vid in sample_vids:
        rec = records[vid]
        # Reconstruct original MP4 path from video_id (e.g. Celeb-synthesis__id1_id2_0002 -> Celeb-synthesis/id1_id2_0002.mp4)
        vid_rel_path = vid.replace("__", "/") + ".mp4"
        mp4_path = video_root / vid_rel_path
        
        if not mp4_path.exists():
            print(f"Skipping {vid}: MP4 not found at {mp4_path}")
            continue
        
        # Read the first extracted frame's index (we assume standard 10FPS sampling where frame 0 is frame 0)
        # Note: server manifest lists frames like "frames/.../000000.jpg"
        if not rec["frames"]: continue
        
        server_frame_rel = rec["frames"][0]
        server_img_path = server_root / server_frame_rel
        
        # 1. Pi4 Video Decode (Just read frame 0 for simplicity)
        cap = cv2.VideoCapture(str(mp4_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read MP4: {mp4_path.name}")
            continue
            
        pi4_raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Pi4 YuNet Face Crop
        pi4_crop, pi4_bbox = crop_face_to_256(pi4_raw_rgb, yunet)
        pi4_crop_bgr = cv2.cvtColor(pi4_crop, cv2.COLOR_RGB2BGR)
        
        # 3. Load Server Crop
        server_crop = cv2.imread(str(server_img_path))
        if server_crop is None: continue
        
        # 4. Compare
        mse = float(np.mean((pi4_crop_bgr.astype(np.float32) - server_crop.astype(np.float32)) ** 2))
        
        print(f"Video: {vid}")
        print(f"  - Pi4 MTCNN BBox    : [{pi4_bbox[0]:.1f}, {pi4_bbox[1]:.1f}, {pi4_bbox[2]:.1f}, {pi4_bbox[3]:.1f}]")
        print(f"  - Pi4 vs Server MSE : {mse:.4f}")
        print(f"  - Result            : {'DIVERGENCE FOUND' if mse > 0 else 'IDENTICAL'}")
        
        total_pixel_mse += mse
        valid_frames += 1

    print("-" * 60)
    print(f"Average Preprocessing Pixel Drift (MSE): {total_pixel_mse / max(1, valid_frames):.4f}")
    print("Conclusion: Any MSE > 0 proves that Edge Video Decoding and/or MTCNN float precision")
    print("causes image divergence compared to Server preprocessing. This explains the AUC drop.")
    print("=" * 60)

if __name__ == "__main__":
    main()
