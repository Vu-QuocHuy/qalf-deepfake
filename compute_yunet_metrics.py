import json
import os
import sys
from pathlib import Path
import numpy as np

# Add qalf to path
sys.path.insert(0, os.path.dirname(__file__))
from qalf.metrics import compute_metrics

def process_file(json_file, out_dir, clips, detector="yunet"):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    labels = []
    scores = []
    errors = 0
    
    for item in data:
        if "error" in item:
            errors += 1
            continue
            
        path = item["video_info"]["path"]
        prob = item["detection"]["fake_probability"]
        
        # Celeb-DF v2 labels: Celeb-synthesis = 1, Celeb-real = 0, YouTube-real = 0
        if "Celeb-synthesis" in path:
            label = 1
        elif "Celeb-real" in path or "YouTube-real" in path:
            label = 0
        else:
            print(f"Unknown label for path: {path}")
            continue
            
        labels.append(label)
        scores.append(prob)
        
    labels = np.array(labels)
    scores = np.array(scores)
    
    # 0.671156 is the frozen threshold mentioned in the brief
    threshold = 0.671156
    
    metrics = compute_metrics(labels, scores, threshold)
    
    protocol = {
        "checkpoint": "models/qalf.onnx", # I'll assume this is the name
        "threshold": threshold,
        "videos_processed": len(labels),
        "errors": errors,
        "face_detector": detector,
        "clips": clips
    }
    
    final_output = {
        "protocol": protocol,
        "metrics": metrics
    }
    
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(final_output, f, indent=4)
        
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    print(f"Saved metrics to {out_dir}")

process_file("yunet_e2e_518_clips1_final.json", "eval_pi4_yunet_1clip", 1)
process_file("yunet_e2e_518_clips3_final.json", "eval_pi4_yunet_3clip", 3)
