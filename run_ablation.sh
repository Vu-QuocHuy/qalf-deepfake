#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# QALF v2 Ablation Study — Full 2^3 Factorial Design
#
# Trains all 8 ablation configurations using the same hyper-parameters and
# data.  Only the three architecture flags differ between runs:
#   --frequency-preprocess, --multiscale, --temporal-attention
#
# Usage:
#   bash run_ablation.sh              # run all 8 ablations
#   bash run_ablation.sh --dry-run    # only print commands
#   bash run_ablation.sh --resume     # skip completed runs
# ============================================================================

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
OUTPUT_DIR="${QALF_ABLATION_OUTPUT_DIR:-$STORAGE_ROOT/experiments/ablation}"

SEED="${QALF_SEED:-42}"
EPOCHS="${QALF_EPOCHS:-50}"

export CUBLAS_WORKSPACE_CONFIG=':4096:8'

echo "============================================================"
echo "  QALF v2 Ablation Study"
echo "  Python:  $PYTHON"
echo "  Output:  $OUTPUT_DIR"
echo "  Seed:    $SEED"
echo "  Epochs:  $EPOCHS"
echo "============================================================"

"$PYTHON" scripts/ablation_study.py \
    --config configs/ffpp_to_celebdf_v2.json \
    --output-dir "$OUTPUT_DIR" \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --python "$PYTHON" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size 6 \
    --num-workers 4 \
    --learning-rate 0.0002 \
    --backbone-learning-rate 0.00002 \
    --weight-decay 0.0005 \
    --early-stop-patience 7 \
    --ema-decay 0.999 \
    --num-frames 32 \
    --texture-frames 10 \
    --image-size 224 \
    --embedding-dim 192 \
    --dropout 0.3 \
    --label-smoothing 0.05 \
    "$@"
