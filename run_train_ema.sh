#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) STORAGE_ROOT='E:/DeepFakeData' ;;
    Linux*) STORAGE_ROOT='/mnt/e/DeepFakeData' ;;
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac

export QALF_EMA_DECAY='0.999'
export QALF_TRAIN_OUTPUT_DIR="$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_ema"
exec "$SCRIPT_DIR/run_train.sh"
