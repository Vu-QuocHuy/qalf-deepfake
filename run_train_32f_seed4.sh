#!/usr/bin/env bash
set -euo pipefail

# Seed-4 experiment: 32 stored frames, 16-frame windows, 8 texture frames.
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
FRAME_ROOT="${QALF_FFPP_FRAME_ROOT:-$DATA_ROOT/extracted/ffpp}"
LANDMARK_OUTPUT_ROOT="${QALF_FFPP_32F_LANDMARK_OUTPUT:-$DATA_ROOT/landmarks/ffpp-landmark-32f-stride2}"
LANDMARK_ROOT="$LANDMARK_OUTPUT_ROOT/landmarks"
TRAIN_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_train_landmarks.jsonl"
VAL_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl"
OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_32f5fps_16w_8f_texture_sbi_ema_seed4}"
SEED="${QALF_SEED:-4}"
EPOCHS="${QALF_EPOCHS:-50}"

export CUBLAS_WORKSPACE_CONFIG=':4096:8'

"$PYTHON" scripts/train.py \
    --config configs/ffpp_to_celebdf_32f_seed4.json \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size 8 \
    --num-workers 4 \
    --learning-rate 0.0003 \
    --backbone-learning-rate 0.00003 \
    --weight-decay 0.0003 \
    --early-stop-patience 5 \
    --ema-decay 0.999 \
    --validation-weights ema \
    --num-frames 16 \
    --texture-frames 8 \
    --image-size 160 \
    --eval-clips-per-video 3 \
    --fake-methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --texture-backbone efficientnet_b0 \
    --texture-mode full_face \
    --embedding-dim 192 \
    --dropout 0.3 \
    --sbi \
    --deterministic
