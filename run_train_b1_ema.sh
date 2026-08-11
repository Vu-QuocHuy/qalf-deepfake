#!/usr/bin/env bash
set -euo pipefail

# EfficientNet-B1 protocol: native 240px resolution, EMA, and texture emphasis.
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
FRAME_ROOT="$DATA_ROOT/extracted/ffpp"
LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/ffpp-landmark"
LANDMARK_ROOT="$LANDMARK_OUTPUT_ROOT/landmarks"
TRAIN_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_train_landmarks.jsonl"
VAL_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl"
OUTPUT_DIR="$STORAGE_ROOT/experiments/qalf_ffpp4_effb1_240_8f_ema_texture"

for required_path in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$FRAME_ROOT" "$LANDMARK_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required path does not exist: $required_path" >&2
        exit 1
    fi
done

echo "Python: $PYTHON"
echo "Training output: $OUTPUT_DIR"
echo "Protocol: EfficientNet-B1 240px, EMA=0.999, texture emphasis"
"$PYTHON" scripts/train.py \
    --config configs/ffpp_to_celebdf_b1_ema.json \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --epochs 35 \
    --batch-size 4 \
    --num-workers 4 \
    --learning-rate 0.0002 \
    --backbone-learning-rate 0.00001 \
    --weight-decay 0.0005 \
    --early-stop-patience 5 \
    --num-frames 32 \
    --texture-frames 8 \
    --image-size 240 \
    --eval-clips-per-video 3 \
    --fake-methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --texture-backbone efficientnet_b1 \
    --geometry-hidden 128 \
    --geometry-layers 3 \
    --embedding-dim 192 \
    --dropout 0.3 \
    --geometry-mode aligned_motion_3d \
    --fusion-mode quality \
    --geometry-loss-weight 0.10 \
    --texture-loss-weight 0.50 \
    --texture-gate-bias 1.0 \
    --ema-decay 0.999 \
    --deterministic
