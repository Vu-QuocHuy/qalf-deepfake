#!/usr/bin/env bash
set -euo pipefail

# Focused cross-dataset profiles built from the best full-face configuration.
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

PROFILE="${1:-full_face_sbi}"
PROFILE_ARGS=()
AUXILIARY_BRANCH='geometry'
case "$PROFILE" in
    full_face_sbi)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi'
        DESCRIPTION='retained geometry + texture SBI baseline'
        PROFILE_ARGS+=(--texture-mode full_face --sbi)
        ;;
    texture_only_sbi)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi_texture_only'
        DESCRIPTION='retained texture-only SBI control'
        AUXILIARY_BRANCH='none'
        PROFILE_ARGS+=(--texture-mode full_face --sbi)
        ;;
    srm_sbi)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi_srm'
        DESCRIPTION='fixed SRM residual bank + lightweight CNN + texture SBI candidate'
        AUXILIARY_BRANCH='srm'
        PROFILE_ARGS+=(--texture-mode full_face --sbi)
        ;;
    *)
        echo "ERROR: unknown profile '$PROFILE'" >&2
        echo 'Use: full_face_sbi, texture_only_sbi, or srm_sbi' >&2
        exit 2
        ;;
esac

SEED="${QALF_SEED:-42}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: QALF_SEED must be a non-negative integer; got '$SEED'" >&2
    exit 2
fi
if [[ "$SEED" != '42' ]]; then
    EXPERIMENT="${EXPERIMENT}_seed${SEED}"
fi

DATA_ROOT="$STORAGE_ROOT/data"
FRAME_ROOT="$DATA_ROOT/extracted/ffpp"
LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/ffpp-landmark"
LANDMARK_ROOT="$LANDMARK_OUTPUT_ROOT/landmarks"
TRAIN_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_train_landmarks.jsonl"
VAL_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl"
OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-$STORAGE_ROOT/experiments/$EXPERIMENT}"
BATCH_SIZE="${QALF_BATCH_SIZE:-8}"
NUM_WORKERS="${QALF_NUM_WORKERS:-4}"

export CUBLAS_WORKSPACE_CONFIG=':4096:8'
echo "Python: $PYTHON"
echo "Profile: $PROFILE ($DESCRIPTION)"
echo "Seed: $SEED"
echo "Training output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
"$PYTHON" -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"

"$PYTHON" scripts/train.py \
    --config configs/ffpp_to_celebdf.json \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --epochs 35 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --learning-rate 0.0003 \
    --backbone-learning-rate 0.00003 \
    --weight-decay 0.0003 \
    --early-stop-patience 5 \
    --num-frames 32 \
    --texture-frames 8 \
    --image-size 160 \
    --eval-clips-per-video 3 \
    --fake-methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --texture-backbone efficientnet_b0 \
    --geometry-hidden 128 \
    --geometry-layers 3 \
    --embedding-dim 192 \
    --dropout 0.3 \
    --geometry-mode aligned_motion_3d \
    --auxiliary-branch "$AUXILIARY_BRANCH" \
    --auxiliary-loss-weight 0.25 \
    --texture-loss-weight 0.25 \
    --texture-gate-bias 0.0 \
    --deterministic \
    "${PROFILE_ARGS[@]}"
