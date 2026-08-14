#!/usr/bin/env bash
set -euo pipefail

# Controlled backbone ablation. All baseline settings remain in run_train.sh;
# only the RGB encoder and output directory are changed here.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${QALF_MOBILENET_OUTPUT_DIR:-E:/DeepFakeData/experiments/qalf_ffpp4_mobilenet_v3_large_160_8f_texture_sbi_ema}"
if [[ "$(uname -s)" == Linux* ]]; then
    OUTPUT_DIR="${QALF_MOBILENET_OUTPUT_DIR:-/mnt/e/DeepFakeData/experiments/qalf_ffpp4_mobilenet_v3_large_160_8f_texture_sbi_ema}"
fi

echo "Texture backbone: mobilenet_v3_large"
echo "Training output: $OUTPUT_DIR"

QALF_TEXTURE_BACKBONE=mobilenet_v3_large \
QALF_TRAIN_OUTPUT_DIR="$OUTPUT_DIR" \
    ./run_train.sh
