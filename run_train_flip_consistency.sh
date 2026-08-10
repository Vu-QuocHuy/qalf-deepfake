#!/usr/bin/env bash
set -euo pipefail

# Controlled baseline ablation: paired texture-flip consistency only.
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
TEXTURE_BACKBONE="${QALF_TEXTURE_BACKBONE:-efficientnet_b0}"
case "$TEXTURE_BACKBONE" in
    efficientnet_b0) EXPERIMENT_BACKBONE_TAG='effb0' ;;
    efficientnet_b1) EXPERIMENT_BACKBONE_TAG='effb1' ;;
    *)
        echo "ERROR: unsupported texture backbone: $TEXTURE_BACKBONE" >&2
        exit 1
        ;;
esac
FRAME_ROOT="$DATA_ROOT/extracted/ffpp"
LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/ffpp-landmark"
LANDMARK_ROOT="$LANDMARK_OUTPUT_ROOT/landmarks"
TRAIN_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_train_landmarks.jsonl"
VAL_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl"
OUTPUT_DIR="$STORAGE_ROOT/experiments/qalf_ffpp4_${EXPERIMENT_BACKBONE_TAG}_160_8f_flip_consistency"

for required_path in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$FRAME_ROOT" "$LANDMARK_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required path does not exist: $required_path" >&2
        exit 1
    fi
done

echo "Python: $PYTHON"
echo "Training output: $OUTPUT_DIR"
echo "Texture backbone: $TEXTURE_BACKBONE"
echo "Flip consistency weight: 0.10"
echo "Validation texture flip TTA: enabled"
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
    --batch-size 8 \
    --num-workers 4 \
    --learning-rate 0.0003 \
    --backbone-learning-rate 0.00003 \
    --weight-decay 0.0003 \
    --early-stop-patience 5 \
    --num-frames 32 \
    --texture-frames 8 \
    --image-size 160 \
    --eval-clips-per-video 3 \
    --fake-methods Deepfakes Face2Face FaceSwap NeuralTextures \
    --texture-backbone "$TEXTURE_BACKBONE" \
    --geometry-hidden 128 \
    --geometry-layers 3 \
    --embedding-dim 192 \
    --dropout 0.3 \
    --geometry-mode aligned_motion_3d \
    --fusion-mode quality \
    --geometry-loss-weight 0.25 \
    --texture-loss-weight 0.25 \
    --flip-consistency-weight 0.10 \
    --validation-texture-flip-tta
