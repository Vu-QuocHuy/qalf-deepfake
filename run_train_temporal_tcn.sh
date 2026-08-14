#!/usr/bin/env bash
set -euo pipefail

# Experimental lightweight temporal candidate. The mean-pooling baseline is
# unchanged; this runner only selects the residual TCN through the shared train
# entry point and writes to a separate checkpoint directory.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-E:/DeepFakeData/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema_temporal_tcn}"
if [[ "$(uname -s)" == Linux* ]]; then
    OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-/mnt/e/DeepFakeData/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema_temporal_tcn}"
fi
echo "Temporal pooling: residual_tcn"
echo "Training output: $OUTPUT_DIR"

QALF_TEMPORAL_POOLING=residual_tcn \
QALF_TEMPORAL_BOTTLENECK="${QALF_TEMPORAL_BOTTLENECK:-48}" \
QALF_TRAIN_OUTPUT_DIR="$OUTPUT_DIR" \
    ./run_train.sh
