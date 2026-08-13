#!/usr/bin/env bash
set -euo pipefail

# QALF v2 texture + frequency + multiscale + temporal-attention training.
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
OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_v2_ffpp4_effb0_224_10f_freq_ms_ta_sbi_ema}"
SEED="${QALF_SEED:-42}"
EPOCHS="${QALF_EPOCHS:-50}"

export CUBLAS_WORKSPACE_CONFIG=':4096:8'
echo "Python: $PYTHON"
echo "Training output: $OUTPUT_DIR"
echo "Architecture: QALF v2 (freq + multiscale + temporal-attention)"

"$PYTHON" scripts/train.py \
    --config configs/ffpp_to_celebdf_v2.json \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size 6 \
    --num-workers 4 \
    --learning-rate 0.0002 \
    --backbone-learning-rate 0.00002 \
    --weight-decay 0.0005 \
    --early-stop-patience 7 \
    --ema-decay 0.999 \
    --validation-weights ema \
    --num-frames 32 \
    --texture-frames 10 \
    --image-size 224 \
    --eval-clips-per-video 3 \
    --fake-methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --texture-backbone efficientnet_b0 \
    --texture-mode full_face \
    --embedding-dim 192 \
    --dropout 0.3 \
    --sbi \
    --deterministic \
    --frequency-preprocess \
    --multiscale \
    --temporal-attention \
    --label-smoothing 0.05
