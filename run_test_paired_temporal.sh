#!/usr/bin/env bash
set -euo pipefail

# Evaluate the dual-rate temporal experiment with its checkpoint-recorded sampling protocol.
WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
WSL_PROJECT_ROOT='/mnt/e/DeepFakeData'

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) STORAGE_ROOT="$WINDOWS_PROJECT_ROOT" ;;
    Linux*) STORAGE_ROOT="$WSL_PROJECT_ROOT" ;;
    *)
        echo "ERROR: unsupported shell platform: $(uname -s)" >&2
        exit 1
        ;;
esac

EXPERIMENT_ROOT="${QALF_EXPERIMENT_ROOT:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_dual_rate_dual_rate_residual}"
export QALF_TEST_CHECKPOINT="${QALF_TEST_CHECKPOINT:-$EXPERIMENT_ROOT/best.pt}"
export QALF_TEST_OUTPUT_DIR="${QALF_TEST_OUTPUT_DIR:-${EXPERIMENT_ROOT}_to_celebdf}"
export QALF_TEST_TEXTURE_FRAMES="${QALF_TEST_TEXTURE_FRAMES:-8}"

exec "$PROJECT_ROOT/run_test.sh"
