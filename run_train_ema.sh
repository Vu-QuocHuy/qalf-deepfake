#!/usr/bin/env bash
set -euo pipefail

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) STORAGE_ROOT='E:/DeepFakeData' ;;
    Linux*) STORAGE_ROOT='/mnt/e/DeepFakeData' ;;
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QALF_TRAIN_OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema}"
export QALF_EMA_DECAY="${QALF_EMA_DECAY:-0.999}"
export QALF_VALIDATION_WEIGHTS=ema
echo "EMA training: decay=$QALF_EMA_DECAY validation_weights=$QALF_VALIDATION_WEIGHTS"
exec "$PROJECT_ROOT/run_train.sh"
