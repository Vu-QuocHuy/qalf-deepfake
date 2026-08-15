#!/usr/bin/env bash
set -euo pipefail

# Canonical texture-only + SBI + EMA training entry point.
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
    echo 'Create a Windows venv for Git Bash or a separate Linux venv for WSL.' >&2
    exit 1
fi

DATA_ROOT="$STORAGE_ROOT/data"
FRAME_ROOT="$DATA_ROOT/extracted/ffpp"
LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/ffpp-landmark"
LANDMARK_ROOT="$LANDMARK_OUTPUT_ROOT/landmarks"
TRAIN_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_train_landmarks.jsonl"
VAL_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl"
OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema}"
SEED="${QALF_SEED:-42}"
EPOCHS="${QALF_EPOCHS:-50}"
TEXTURE_BACKBONE="${QALF_TEXTURE_BACKBONE:-efficientnet_b0}"
IMAGE_SIZE="${QALF_IMAGE_SIZE:-160}"
BATCH_SIZE="${QALF_BATCH_SIZE:-8}"
LEARNING_RATE="${QALF_LEARNING_RATE:-0.0003}"
BACKBONE_LEARNING_RATE="${QALF_BACKBONE_LEARNING_RATE:-0.00003}"
WEIGHT_DECAY="${QALF_WEIGHT_DECAY:-0.0003}"
EARLY_STOP_PATIENCE="${QALF_EARLY_STOP_PATIENCE:-5}"
EMA_DECAY="${QALF_EMA_DECAY:-0.999}"
TEXTURE_FRAMES="${QALF_TEXTURE_FRAMES:-8}"
EMBEDDING_DIM="${QALF_EMBEDDING_DIM:-192}"
DROPOUT="${QALF_DROPOUT:-0.3}"

export CUBLAS_WORKSPACE_CONFIG=':4096:8'

"$PYTHON" scripts/train.py \
    --config configs/ffpp_to_celebdf.json \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 4 \
    --learning-rate "$LEARNING_RATE" \
    --backbone-learning-rate "$BACKBONE_LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --ema-decay "$EMA_DECAY" \
    --validation-weights ema \
    --num-frames 32 \
    --texture-frames "$TEXTURE_FRAMES" \
    --image-size "$IMAGE_SIZE" \
    --eval-clips-per-video 3 \
    --fake-methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --texture-backbone "$TEXTURE_BACKBONE" \
    --texture-mode full_face \
    --embedding-dim "$EMBEDDING_DIM" \
    --dropout "$DROPOUT" \
    --sbi \
    --deterministic
