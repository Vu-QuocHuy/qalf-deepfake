#!/usr/bin/env bash
set -euo pipefail

WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
WSL_PROJECT_ROOT='/mnt/e/DeepFakeData'

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
        STORAGE_ROOT="$WINDOWS_PROJECT_ROOT"
        ;;
    Linux*)
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
        STORAGE_ROOT="$WSL_PROJECT_ROOT"
        ;;
    *)
        echo "ERROR: unsupported shell platform: $(uname -s)" >&2
        exit 1
        ;;
esac
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtual-environment Python not found: $PYTHON" >&2
    exit 1
fi

DATA_ROOT="$STORAGE_ROOT/data"
FFPP_FRAME_ROOT="$DATA_ROOT/extracted/ffpp"
FFPP_LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/ffpp-landmark"
FFPP_LANDMARK_ROOT="$FFPP_LANDMARK_OUTPUT_ROOT/landmarks"
FFPP_VAL_MANIFEST="$FFPP_LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl"
CELEBDF_FRAME_ROOT="$DATA_ROOT/extracted/celebdf"
CELEBDF_LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/celebdf-landmark"
CELEBDF_LANDMARK_ROOT="$CELEBDF_LANDMARK_OUTPUT_ROOT/landmarks"
CELEBDF_TEST_MANIFEST="$CELEBDF_LANDMARK_OUTPUT_ROOT/manifests/celebdf_test_landmarks.jsonl"
CHECKPOINT="$STORAGE_ROOT/experiments/qalf_ffpp4_effb1_240_8f_ema_texture/best.pt"
OUTPUT_DIR="$STORAGE_ROOT/experiments/qalf_ffpp4_effb1_240_8f_ema_texture_to_celebdf_flip_tta_ffpp_threshold"

echo "Python: $PYTHON"
echo "Checkpoint: $CHECKPOINT"
echo "Evaluation output: $OUTPUT_DIR"
echo "Texture backbone: efficientnet_b1 (240px)"
echo "Checkpoint weights: EMA"
echo "Texture flip TTA: enabled"
echo "Threshold calibration: $FFPP_VAL_MANIFEST"

"$PYTHON" scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --manifest "$CELEBDF_TEST_MANIFEST" \
    --frame-root "$CELEBDF_FRAME_ROOT" \
    --landmark-root "$CELEBDF_LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 4 \
    --num-workers 4 \
    --clips-per-video 3 \
    --aggregation mean \
    --top-k 1 \
    --texture-flip-tta \
    --threshold-manifest "$FFPP_VAL_MANIFEST" \
    --threshold-frame-root "$FFPP_FRAME_ROOT" \
    --threshold-landmark-root "$FFPP_LANDMARK_ROOT" \
    --threshold-clips-per-video 3
