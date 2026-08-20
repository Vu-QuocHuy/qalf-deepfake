#!/usr/bin/env bash
# run_pi4_e2e_extraction.sh
# End-to-End Extraction Pipeline for Pi 4
# This script extracts 518 videos, detects faces (MTCNN), extracts landmarks, and evaluates the model.
# NOTE: This is extremely CPU intensive on Pi 4 and will take ~8-12 hours.

set -e

USB_ROOT="/mnt/usb_data"
VENV_PYTHON="$USB_ROOT/venv_qalf/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment not found at $VENV_PYTHON"
    echo "Please ensure the venv is created."
    exit 1
fi

DATASET_ROOT="$USB_ROOT/celebdf_v2/test"
TEST_LIST="$USB_ROOT/celebdf_test.txt"
OUTPUT_DIR="$USB_ROOT/extracted_celebdf_pi4_true"
MANIFEST="$OUTPUT_DIR/manifests/celebdf_test.jsonl"
EVAL_DIR="$USB_ROOT/eval_pi4_true_e2e"

echo "=========================================================="
echo "    STARTING FULL END-TO-END EXTRACTION ON PI 4           "
echo "=========================================================="
echo "Output Directory : $OUTPUT_DIR"
echo "Start Time       : $(date)"

# 1. Extract Frames (Video Decode + MTCNN Face Crop)
echo -e "\n---> [1/3] Extracting Frames and Face Crops using MTCNN..."
$VENV_PYTHON scripts/extract_frames.py celebdf \
    --dataset-root "$DATASET_ROOT" \
    --test-list "$TEST_LIST" \
    --output-root "$OUTPUT_DIR" \
    --device cpu \
    --mtcnn-batch-size 1 \
    --cpu-threads 4

# 2. Extract Landmarks (MediaPipe 468 points)
echo -e "\n---> [2/3] Extracting Facial Landmarks using MediaPipe..."
$VENV_PYTHON scripts/extract_landmarks.py \
    --manifest "$MANIFEST" \
    --frame-root "$OUTPUT_DIR" \
    --output-root "$OUTPUT_DIR/landmarks" \
    --backend mediapipe \
    --device cpu

# 3. Evaluate Model on the newly extracted E2E data
echo -e "\n---> [3/3] Evaluating ONNX Model on E2E Data..."
$VENV_PYTHON scripts/evaluate_pi4.py \
    --manifest "$OUTPUT_DIR/manifests/celebdf_test_landmarks.jsonl" \
    --frame-root "$OUTPUT_DIR" \
    --landmark-root "$OUTPUT_DIR/landmarks" \
    --onnx models/qalf.onnx \
    --output-dir "$EVAL_DIR" \
    --num-frames 32 \
    --texture-frames 8 \
    --clips-per-video 3 \
    --aggregation mean \
    --texture-flip-tta \
    --cpu-threads 4

echo "=========================================================="
echo "    EXTRACTION AND EVALUATION COMPLETED!                  "
echo "=========================================================="
echo "End Time : $(date)"
echo "Metrics saved to : $EVAL_DIR/metrics.txt"
