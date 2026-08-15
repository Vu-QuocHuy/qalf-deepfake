#!/usr/bin/env bash
set -euo pipefail

# Controlled backbone ablation. The runner uses a MobileNetV3-Large-specific
# transfer-learning schedule while keeping the video/SBI/EMA protocol fixed.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${QALF_MOBILENET_OUTPUT_DIR:-E:/DeepFakeData/experiments/qalf_ffpp4_mobilenet_v3_large_224_8f_texture_sbi_ema}"
if [[ "$(uname -s)" == Linux* ]]; then
    OUTPUT_DIR="${QALF_MOBILENET_OUTPUT_DIR:-/mnt/e/DeepFakeData/experiments/qalf_ffpp4_mobilenet_v3_large_224_8f_texture_sbi_ema}"
fi

echo "Texture backbone: mobilenet_v3_large"
echo "Image size: 224"
echo "Train config: head_lr=3e-4 backbone_lr=1e-5 weight_decay=3e-4 batch=8 dropout=0.3 embedding=192"
echo "Training output: $OUTPUT_DIR"

QALF_TEXTURE_BACKBONE=mobilenet_v3_large \
QALF_IMAGE_SIZE=224 \
QALF_BATCH_SIZE=8 \
QALF_LEARNING_RATE=0.0003 \
QALF_BACKBONE_LEARNING_RATE=0.00001 \
QALF_WEIGHT_DECAY=0.0003 \
QALF_EARLY_STOP_PATIENCE=5 \
QALF_EMA_DECAY=0.999 \
QALF_TEXTURE_FRAMES=8 \
QALF_EMBEDDING_DIM=192 \
QALF_DROPOUT=0.3 \
QALF_TRAIN_OUTPUT_DIR="$OUTPUT_DIR" \
    ./run_train.sh
