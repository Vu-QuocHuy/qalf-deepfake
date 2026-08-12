#!/usr/bin/env bash
set -euo pipefail

# Canonical FF++ -> Celeb-DF texture-only SBI evaluation profile.
WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
WSL_PROJECT_ROOT='/mnt/e/DeepFakeData'
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) STORAGE_ROOT="$WINDOWS_PROJECT_ROOT" ;;
    Linux*) STORAGE_ROOT="$WSL_PROJECT_ROOT" ;;
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac

SEED="${QALF_SEED:-42}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: QALF_SEED must be a non-negative integer; got '$SEED'" >&2
    exit 2
fi
EXPERIMENT='qalf_ffpp4_effb0_160_8f_texture_sbi'
if [[ "$SEED" != '42' ]]; then
    EXPERIMENT="${EXPERIMENT}_seed${SEED}"
fi
export QALF_TEST_CHECKPOINT="$STORAGE_ROOT/experiments/$EXPERIMENT/best.pt"
export QALF_TEST_OUTPUT_DIR="$STORAGE_ROOT/experiments/${EXPERIMENT}_to_celebdf_12f_3clips_mean_tta_ffpp_threshold"
export QALF_TEXTURE_FRAMES=12

echo "Cross-dataset model: texture-only SBI"
echo "Seed: $SEED"
"$PROJECT_ROOT/run_test.sh"
