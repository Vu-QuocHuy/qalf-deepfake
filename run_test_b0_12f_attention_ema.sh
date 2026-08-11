#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) STORAGE_ROOT='E:/DeepFakeData' ;;
    Linux*) STORAGE_ROOT='/mnt/e/DeepFakeData' ;;
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac

export QALF_TEST_CHECKPOINT="$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_12f_attention_ema/best.pt"
export QALF_TEST_OUTPUT_DIR="$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_12f_attention_ema_to_celebdf_flip_tta_ffpp_threshold"
exec "$SCRIPT_DIR/run_test.sh"
