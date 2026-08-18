#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# BASH SCRIPT: OFFICIAL EVALUATION ON RASPBERRY PI 4
# Exact match with server pipeline using pre-extracted data
# ==============================================================================

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "/mnt/usb_data/venv_qalf/bin/activate" ]]; then
    source /mnt/usb_data/venv_qalf/bin/activate
fi

export PYTHONUNBUFFERED=1

USB_ROOT="/mnt/usb_data"
EXTRACTED_ROOT="$USB_ROOT/extracted_celebdf"
MANIFEST="$EXTRACTED_ROOT/manifests/celebdf_test_landmarks.jsonl"
FRAME_ROOT="$EXTRACTED_ROOT"
LANDMARK_ROOT="$EXTRACTED_ROOT"
ONNX_MODEL="models/qalf.onnx"
OUTPUT_DIR="$USB_ROOT/eval_pi4_official"

echo "=============================================================================="
echo "    OFFICIAL TEXTURESBI EVALUATION ON RASPBERRY PI 4"
echo "    (Using pre-extracted MTCNN frames + MediaPipe landmarks)"
echo "=============================================================================="
echo "Manifest       : $MANIFEST"
echo "Frame Root     : $FRAME_ROOT"
echo "Landmark Root  : $LANDMARK_ROOT"
echo "ONNX Model     : $ONNX_MODEL"
echo "Output Dir     : $OUTPUT_DIR"
echo "Protocol       : 3 clips × 8 frames, Flip-TTA, Mean aggregation"
echo "=============================================================================="
echo ""

python -u scripts/evaluate_pi4.py \
    --manifest "$MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --onnx "$ONNX_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --num-frames 32 \
    --texture-frames 8 \
    --clips-per-video 3 \
    --aggregation mean \
    --texture-flip-tta \
    --cpu-threads 4

echo ""
echo "=============================================================================="
echo "[DONE] Results saved to: $OUTPUT_DIR"
echo "=============================================================================="
