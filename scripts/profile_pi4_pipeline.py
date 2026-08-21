import json
import os
import sys
import time
import threading
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Please install psutil: pip install psutil")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing E2E pipeline logic
try:
    from scripts.infer_video import process_video_pipeline
    import torch
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Hardware monitoring
monitor_data = {"cpu": [], "ram": [], "temp": []}
monitoring_active = True

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def monitor_hardware():
    process = psutil.Process(os.getpid())
    while monitoring_active:
        cpu = psutil.cpu_percent(interval=0.1)
        ram = process.memory_info().rss / (1024 * 1024)
        temp = get_cpu_temp()
        
        monitor_data["cpu"].append(cpu)
        monitor_data["ram"].append(ram)
        monitor_data["temp"].append(temp)
        time.sleep(0.4)

def main():
    print("=" * 70)
    print("   PI 4 END-TO-END PIPELINE & HARDWARE PROFILER (1 VIDEO)   ")
    print("=" * 70)
    
    video_path = Path("/mnt/usb_data/celebdf_test_518/Celeb-synthesis/id1_id2_0002.mp4")
    onnx_model = Path("models/qalf.onnx")
    
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
        
    print(f"Loading ONNX model: {onnx_model.name}...")
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_session = ort.InferenceSession(str(onnx_model), sess_options, providers=["CPUExecutionProvider"])
    
    print("Initializing MTCNN...")
    from facenet_pytorch import MTCNN
    mtcnn_detector = MTCNN(image_size=256, margin=0, keep_all=False, post_process=False, device="cpu")
    
    print("Initializing Landmarker Extractor (Auto-backend)...")
    from qalf.data.landmarks import FaceLandmarkerExtractor, ensure_face_landmarker_model
    lm_model_path = ensure_face_landmarker_model("models/face_landmarker.task", download=True)
    landmarker = FaceLandmarkerExtractor(lm_model_path, running_mode="image", min_confidence=0.5, backend="auto")
    
    video_dir = Path("/mnt/usb_data/celebdf_test_518/Celeb-synthesis")
    video_paths = list(video_dir.glob("*.mp4"))[:5]
    if not video_paths:
        print(f"No videos found in {video_dir}")
        return
        
    print(f"\nProcessing {len(video_paths)} videos E2E for stable latency metrics...")
    print("Starting hardware monitor thread...")
    global monitoring_active
    monitor_thread = threading.Thread(target=monitor_hardware)
    monitor_thread.start()
    
    all_timings = {"1_video_decode_ms": [], "2_landmark_and_crop_ms": [], "3_face_align_and_preprocess_ms": [], "4_model_forward_ms": [], "total_end_to_end_ms": []}
    
    try:
        for idx, video_path in enumerate(video_paths):
            print(f"  [{idx+1}/{len(video_paths)}] Processing {video_path.name}...")
            rep = process_video_pipeline(
                video_path=video_path,
                landmarker=landmarker,
                model=None,
                onnx_session=onnx_session,
                mtcnn_detector=mtcnn_detector,
                num_frames=32,
                texture_frames=8,
                target_fps=10.0,
                image_size=160,
                clips=3,
                flip_tta=True,
                aggregation="mean",
                top_k=1,
                device=torch.device("cpu"),
                no_landmarks=False
            )
            for k, v in rep["timings_ms"].items():
                if k in all_timings:
                    all_timings[k].append(v)
    finally:
        monitoring_active = False
        monitor_thread.join()
        landmarker.close()

    print("\n" + "=" * 70)
    print("             END-TO-END LATENCY BREAKDOWN (Mean of 5 Videos)    ")
    print("=" * 70)
    import statistics
    for k, v_list in all_timings.items():
        mean_val = statistics.mean(v_list)
        std_val = statistics.stdev(v_list) if len(v_list) > 1 else 0.0
        print(f"  {k:<35}: {mean_val:>8.2f} ms ± {std_val:>5.2f} ms")
        
    print("\n" + "=" * 70)
    print("                  HARDWARE RESOURCE UTILIZATION                 ")
    print("=" * 70)
    avg_cpu = sum(monitor_data["cpu"]) / max(1, len(monitor_data["cpu"]))
    max_cpu = max(monitor_data["cpu"]) if monitor_data["cpu"] else 0
    avg_ram = sum(monitor_data["ram"]) / max(1, len(monitor_data["ram"]))
    max_ram = max(monitor_data["ram"]) if monitor_data["ram"] else 0
    avg_temp = sum(monitor_data["temp"]) / max(1, len(monitor_data["temp"]))
    max_temp = max(monitor_data["temp"]) if monitor_data["temp"] else 0
    
    print(f"  CPU Usage (System) : Avg: {avg_cpu:5.1f}% | Peak: {max_cpu:5.1f}%")
    print(f"  RAM Usage (Process): Avg: {avg_ram:5.1f}MB | Peak: {max_ram:5.1f}MB")
    print(f"  CPU Temperature    : Avg: {avg_temp:5.1f}°C | Peak: {max_temp:5.1f}°C")
    
    print("\n" + "=" * 70)
    print("Note: This is the exact End-to-End time (including MP4 decoding and MTCNN).")
    print("Compare this with the Pre-extracted latency to see the preprocessing bottleneck.")

if __name__ == "__main__":
    main()
