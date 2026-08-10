#!/usr/bin/env bash
set -euo pipefail

# =========================== EDIT CONFIGURATION HERE ===========================
WINDOWS_DATA_ROOT='F:/DeepFakedata/outputs_duong_huy'
WSL_DATA_ROOT='/mnt/f/DeepFakedata/outputs_duong_huy'

EPOCHS=25
BATCH_SIZE=16
NUM_WORKERS=4
LEARNING_RATE=0.0003
WEIGHT_DECAY=0.0001
EARLY_STOP_PATIENCE=6

NUM_FRAMES=32
TEXTURE_FRAMES=4
IMAGE_SIZE=128
EVAL_CLIPS_PER_VIDEO=2

GEOMETRY_MODE='aligned_motion_3d'
FUSION_MODE='quality'
GEOMETRY_LOSS_WEIGHT=0.25
TEXTURE_LOSS_WEIGHT=0.25
SEED=42

USE_AMP=1
USE_BALANCED_SAMPLER=1
RUN_DATA_AUDIT=1
# ==============================================================================

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
        DATA_ROOT="$WINDOWS_DATA_ROOT"
        ;;
    Linux*)
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
        DATA_ROOT="$WSL_DATA_ROOT"
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

FRAME_ROOT="$DATA_ROOT/data/extracted/ffpp"
LANDMARK_ROOT="$DATA_ROOT/data/landmarks/ffpp-landmark"
TRAIN_MANIFEST="$LANDMARK_ROOT/manifests/ffpp_train_landmarks.jsonl"
VAL_MANIFEST="$LANDMARK_ROOT/manifests/ffpp_val_landmarks.jsonl"
OUTPUT_DIR="$DATA_ROOT/experiments/qalf_ffpp"

for required_path in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$FRAME_ROOT" "$LANDMARK_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required path does not exist: $required_path" >&2
        exit 1
    fi
done

echo "Python: $PYTHON"
echo "Training output: $OUTPUT_DIR"
"$PYTHON" -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"

if [[ "$RUN_DATA_AUDIT" == '1' ]]; then
    "$PYTHON" scripts/audit_manifest.py \
        --manifest "$TRAIN_MANIFEST" "$VAL_MANIFEST" \
        --frame-root "$FRAME_ROOT" \
        --landmark-root "$LANDMARK_ROOT" \
        --expected-frames 64
fi

TRAIN_ARGS=(
    scripts/train.py
    --config configs/ffpp_to_celebdf.json
    --train-manifest "$TRAIN_MANIFEST"
    --val-manifest "$VAL_MANIFEST"
    --frame-root "$FRAME_ROOT"
    --landmark-root "$LANDMARK_ROOT"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --early-stop-patience "$EARLY_STOP_PATIENCE"
    --num-frames "$NUM_FRAMES"
    --texture-frames "$TEXTURE_FRAMES"
    --image-size "$IMAGE_SIZE"
    --eval-clips-per-video "$EVAL_CLIPS_PER_VIDEO"
    --geometry-mode "$GEOMETRY_MODE"
    --fusion-mode "$FUSION_MODE"
    --geometry-loss-weight "$GEOMETRY_LOSS_WEIGHT"
    --texture-loss-weight "$TEXTURE_LOSS_WEIGHT"
)

if [[ "$USE_AMP" != '1' ]]; then
    TRAIN_ARGS+=(--no-amp)
fi
if [[ "$USE_BALANCED_SAMPLER" != '1' ]]; then
    TRAIN_ARGS+=(--no-balanced-sampler)
fi

"$PYTHON" "${TRAIN_ARGS[@]}"
