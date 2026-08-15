#!/usr/bin/env bash
set -euo pipefail

# Evaluates the checkpoint produced by run_train_mobilenet_v3_large.sh using
# the same 8-frame/3-clip/mean/EER protocol as the baseline.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT_DIR="${QALF_MOBILENET_OUTPUT_DIR:-E:/DeepFakeData/experiments/qalf_ffpp4_mobilenet_v3_large_224_8f_texture_sbi_ema}"
if [[ "$(uname -s)" == Linux* ]]; then
    EXPERIMENT_DIR="${QALF_MOBILENET_OUTPUT_DIR:-/mnt/e/DeepFakeData/experiments/qalf_ffpp4_mobilenet_v3_large_224_8f_texture_sbi_ema}"
fi
CHECKPOINT="$EXPERIMENT_DIR/best.pt"
OUTPUT_DIR="${QALF_MOBILENET_TEST_OUTPUT_DIR:-${EXPERIMENT_DIR}_to_celebdf_8f_3clips_mean_eer_tta}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    echo "Train first with ./run_train_mobilenet_v3_large.sh" >&2
    exit 1
fi

echo "Texture backbone: mobilenet_v3_large"
echo "Image size: 224 (read from checkpoint config)"
echo "Checkpoint: $CHECKPOINT"
echo "Evaluation output: $OUTPUT_DIR"

QALF_TEST_CHECKPOINT="$CHECKPOINT" \
QALF_TEST_TEXTURE_FRAMES=8 \
QALF_TEST_CLIPS_PER_VIDEO=3 \
QALF_TEST_AGGREGATION=mean \
QALF_THRESHOLD_SELECTION=eer \
QALF_TEST_OUTPUT_DIR="$OUTPUT_DIR" \
    ./run_test.sh
