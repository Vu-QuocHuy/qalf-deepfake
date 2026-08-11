#!/usr/bin/env bash
set -euo pipefail

# Cross-dataset research profiles. Each ablation changes one mechanism from the
# deterministic B0 control; temporal_dg combines the mechanisms after ablation.
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

PROFILE="${1:-temporal_dg}"
PROFILE_ARGS=()
case "$PROFILE" in
    control)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_control_deterministic'
        DESCRIPTION='deterministic reproduction control'
        ;;
    full_face)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_deterministic'
        DESCRIPTION='full aligned face instead of four rectangular skin regions'
        PROFILE_ARGS+=(--texture-mode full_face)
        ;;
    mixstyle)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_mixstyle_deterministic'
        DESCRIPTION='video-coherent MixStyle at shallow EfficientNet stages'
        PROFILE_ARGS+=(
            --texture-mixstyle-probability 0.5
            --texture-mixstyle-alpha 0.1
            --texture-mixstyle-layers 1 2
        )
        ;;
    dynamics)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_dynamics_deterministic'
        DESCRIPTION='mean plus temporal variance/velocity/acceleration pooling'
        PROFILE_ARGS+=(--texture-temporal-pooling dynamics)
        ;;
    temporal_dg)
        EXPERIMENT='qalf_ffpp4_effb0_224_16f_temporal_dg'
        DESCRIPTION='full face, 224px, 16f dynamics, Video MixStyle, and EMA'
        PROFILE_ARGS+=(
            --texture-mode full_face
            --texture-frames 16
            --image-size 224
            --texture-temporal-pooling dynamics
            --texture-mixstyle-probability 0.5
            --texture-mixstyle-alpha 0.1
            --texture-mixstyle-layers 1 2
            --ema-decay 0.999
        )
        ;;
    *)
        echo "ERROR: unknown profile '$PROFILE'" >&2
        echo 'Use: control, full_face, mixstyle, dynamics, or temporal_dg' >&2
        exit 2
        ;;
esac

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
    --seed 42 \
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
    --fusion-mode quality \
    --geometry-loss-weight 0.25 \
    --texture-loss-weight 0.25 \
    --texture-gate-bias 0.0 \
    --deterministic \
    "${PROFILE_ARGS[@]}"
