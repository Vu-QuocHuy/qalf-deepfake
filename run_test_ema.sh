#!/usr/bin/env bash
set -euo pipefail

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) STORAGE_ROOT='E:/DeepFakeData' ;;
    Linux*) STORAGE_ROOT='/mnt/e/DeepFakeData' ;;
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QALF_TEST_CHECKPOINT="${QALF_TEST_CHECKPOINT:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema/best.pt}"
export QALF_TEST_OUTPUT_DIR="${QALF_TEST_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema_to_celebdf_12f_3clips_mean_tta_ffpp_threshold}"
exec "$PROJECT_ROOT/run_test.sh"
