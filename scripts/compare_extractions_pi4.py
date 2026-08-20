import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2

def load_manifest(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            records[data["video_id"]] = data
    return records

def main():
    pi4_root = Path("/mnt/usb_data/extracted_celebdf")
    server_root = Path("/mnt/usb_data/extracted_celebdf_server")
    
    pi4_manifest = pi4_root / "manifests" / "celebdf_test_landmarks.jsonl"
    server_manifest = server_root / "manifests" / "celebdf_test_landmarks.jsonl"
    
    pi4_records = load_manifest(pi4_manifest)
    server_records = load_manifest(server_manifest)
    common_videos = sorted(list(set(pi4_records.keys()) & set(server_records.keys())))
    
    total_pixel_mse = 0.0
    frames_checked = 0
    diff_pixel_videos = 0
    diff_pixel_frames = 0
    
    print("\nScanning ALL frames of all 518 videos for pixel drift...")
    for vid in common_videos:
        p_rec = pi4_records[vid]
        s_rec = server_records[vid]
        
        vid_has_diff = False
        common_frames = set(p_rec["frames"]) & set(s_rec["frames"])
        for frame_rel in common_frames:
            p_img_path = pi4_root / frame_rel
            s_img_path = server_root / frame_rel
            
            p_img = cv2.imread(str(p_img_path))
            s_img = cv2.imread(str(s_img_path))
            
            if p_img is not None and s_img is not None and p_img.shape == s_img.shape:
                mse = float(np.mean((p_img.astype(np.float32) - s_img.astype(np.float32)) ** 2))
                total_pixel_mse += mse
                frames_checked += 1
                if mse > 0:
                    diff_pixel_frames += 1
                    vid_has_diff = True
        
        if vid_has_diff:
            diff_pixel_videos += 1

    print("\n==============================================")
    print("FINAL SUMMARY (All 518 Videos, ALL Frames)")
    print("==============================================")
    print(f"Total Frames Checked: {frames_checked}")
    print(f"Frames with Pixel differences: {diff_pixel_frames}")
    print(f"Videos with Pixel differences: {diff_pixel_videos} / {len(common_videos)}")
    if frames_checked > 0:
        print(f"Average Pixel MSE: {total_pixel_mse / frames_checked:.6f}")
    
if __name__ == "__main__":
    main()
