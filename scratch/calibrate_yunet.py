import cv2
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.data.dataset import _aligned_full_face
from qalf.data.landmarks import FaceLandmarkerExtractor, OpenCVYuNetLandmarker, ensure_face_landmarker_model

def main():
    import glob
    videos = glob.glob("/mnt/usb_data/celebdf_test_518/Celeb-synthesis/*.mp4")
    if not videos: 
        print("No videos found!")
        return
    
    mp = FaceLandmarkerExtractor(ensure_face_landmarker_model("models/face_landmarker.task"), running_mode="image", backend="mediapipe")
    yunet = OpenCVYuNetLandmarker(score_threshold=0.5)
    
    yunet_targets = []
    
    for v in videos[:10]:
        cap = cv2.VideoCapture(v)
        for _ in range(5):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox = yunet.detect_bbox(frame)
            if bbox is None: continue
            bx1, by1, bx2, by2 = bbox
            bw, bh = bx2 - bx1, by2 - by1
            cx, cy = bx1 + bw/2.0, by1 + bh/2.0
            side = max(bw, bh) * 1.35
            x1, y1 = max(0, int(cx - side/2)), max(0, int(cy - side/2))
            x2, y2 = min(frame.shape[1], int(cx + side/2)), min(frame.shape[0], int(cy + side/2))
            crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256))
            
            # MediaPipe alignment
            pts = mp.process(crop)
            if pts is None: continue
            aligned, _ = _aligned_full_face(crop, pts, True, 160)
            
            # Now find YuNet landmarks on the ALIGNED image
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            ylms = yunet.detect_landmarks(aligned_bgr)
            if ylms is not None:
                # Need to normalize to 0-1 range to act as affine targets!
                yunet_targets.append(ylms[:, :2] / 160.0)
        cap.release()
        
    mp.close()
    
    if yunet_targets:
        avg_ylms = np.mean(yunet_targets, axis=0)
        print("Calibrated YuNet Target Coordinates (relative 0-1):")
        print(f"Right Eye (x,y): {avg_ylms[0]}")
        print(f"Left Eye (x,y): {avg_ylms[1]}")
        print(f"Nose (x,y): {avg_ylms[2]}")
        print(f"Right Mouth (x,y): {avg_ylms[3]}")
        print(f"Left Mouth (x,y): {avg_ylms[4]}")
        
        # calculate mouth average like in dataset.py
        mouth = (avg_ylms[3] + avg_ylms[4]) / 2.0
        print(f"Mouth center (x,y): {mouth}")
        
        print("\nTarget Array to use in dataset.py:")
        print("target = np.float32([")
        print(f"    [{avg_ylms[1][0]:.4f} * output_size, {avg_ylms[1][1]:.4f} * output_size], # Left Eye (subject's left)")
        print(f"    [{avg_ylms[0][0]:.4f} * output_size, {avg_ylms[0][1]:.4f} * output_size], # Right Eye (subject's right)")
        print(f"    [{mouth[0]:.4f} * output_size, {mouth[1]:.4f} * output_size], # Mouth")
        print("])")

if __name__ == "__main__":
    main()
