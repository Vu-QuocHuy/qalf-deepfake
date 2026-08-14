#!/usr/bin/env bash
set -euo pipefail

# Diagnostic only: EfficientNet-B0 baseline without the second landmark
# alignment step. The extracted frames are already MTCNN square face crops.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${QALF_NO_LANDMARK_OUTPUT_DIR:-E:/DeepFakeData/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema_no_landmark}"
if [[ "$(uname -s)" == Linux* ]]; then
    OUTPUT_DIR="${QALF_NO_LANDMARK_OUTPUT_DIR:-/mnt/e/DeepFakeData/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema_no_landmark}"
fi

echo "Texture backbone: efficientnet_b0"
echo "Landmark alignment: disabled"
echo "Training output: $OUTPUT_DIR"

QALF_TEXTURE_BACKBONE=efficientnet_b0 \
QALF_LANDMARK_ALIGNMENT=0 \
QALF_TRAIN_OUTPUT_DIR="$OUTPUT_DIR" \
    ./run_train.sh
