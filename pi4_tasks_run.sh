#!/bin/bash
echo "Kích hoạt môi trường ảo..."
source venv_qalf/bin/activate

echo "=========================================================="
echo "Task 2: Profiling Hardware & Latency (N=5) - YuNet 1 Clip"
echo "=========================================================="
python scripts/profile_pi4_pipeline.py --backend yunet --clips 1 --n-videos 5 > /mnt/usb_data/hardware_profiling_yunet_1clip_n5.log 2>&1
echo "Đã lưu vào /mnt/usb_data/hardware_profiling_yunet_1clip_n5.log"

echo "=========================================================="
echo "Task 2: Profiling Hardware & Latency (N=5) - YuNet 3 Clips"
echo "=========================================================="
python scripts/profile_pi4_pipeline.py --backend yunet --clips 3 --n-videos 5 > /mnt/usb_data/hardware_profiling_yunet_3clip_n5.log 2>&1
echo "Đã lưu vào /mnt/usb_data/hardware_profiling_yunet_3clip_n5.log"

echo "=========================================================="
echo "Task 3: Ablation Pixel Diff (YuNet vs MediaPipe/Server)"
echo "=========================================================="
python scripts/prove_extraction_drift_yunet.py > /mnt/usb_data/ablation_pixel_diff_yunet.log 2>&1
echo "Đã lưu vào /mnt/usb_data/ablation_pixel_diff_yunet.log"

echo "=========================================================="
echo "Task 4: Định danh mô hình YuNet"
echo "=========================================================="
echo "OpenCV Version:" > /mnt/usb_data/yunet_model_info.txt
python -c "import cv2; print(cv2.__version__)" >> /mnt/usb_data/yunet_model_info.txt
echo "Model file info:" >> /mnt/usb_data/yunet_model_info.txt
ls -la models/face_detection_yunet_2023mar.onnx >> /mnt/usb_data/yunet_model_info.txt
echo "Đã lưu vào /mnt/usb_data/yunet_model_info.txt"

echo "HOÀN TẤT! Vui lòng gửi lại nội dung 4 file log trên cho Gemini."
