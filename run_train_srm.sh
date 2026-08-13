#!/usr/bin/env bash
set -euo pipefail

# SRM preprocessing experiment: same RGB/SBI/EMA baseline, with three fixed
# 5x5 residual kernels injected before EfficientNet-B0. There is no auxiliary
# encoder or fusion gate.
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
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtual-environment Python not found: $PYTHON" >&2
    exit 1
fi

DATA_ROOT="$STORAGE_ROOT/data"
OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_srm_ema}"
export CUBLAS_WORKSPACE_CONFIG=':4096:8'

"$PYTHON" scripts/train.py \
    --config configs/ffpp_to_celebdf_srm_preprocess.json \
    --train-manifest "$DATA_ROOT/landmarks/ffpp-landmark/manifests/ffpp_train_landmarks.jsonl" \
    --val-manifest "$DATA_ROOT/landmarks/ffpp-landmark/manifests/ffpp_val_landmarks.jsonl" \
    --frame-root "$DATA_ROOT/extracted/ffpp" \
    --landmark-root "$DATA_ROOT/landmarks/ffpp-landmark/landmarks" \
    --output-dir "$OUTPUT_DIR" \
    --seed "${QALF_SEED:-42}" \
    --epochs "${QALF_EPOCHS:-50}" \
    --batch-size 8 \
    --num-workers 4 \
    --learning-rate 0.0003 \
    --backbone-learning-rate 0.00003 \
    --weight-decay 0.0003 \
    --early-stop-patience 5 \
    --ema-decay 0.999 \
    --validation-weights ema \
    --num-frames 32 \
    --texture-frames 8 \
    --image-size 160 \
    --eval-clips-per-video 3 \
    --fake-methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --texture-backbone efficientnet_b0 \
    --texture-mode full_face \
    --embedding-dim 192 \
    --dropout 0.3 \
    --sbi \
    --srm-preprocess \
    --deterministic
