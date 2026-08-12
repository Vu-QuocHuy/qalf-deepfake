#!/usr/bin/env bash
set -euo pipefail

# Canonical FF++ -> Celeb-DF texture-only SBI training profile.
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
export QALF_TRAIN_OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-$STORAGE_ROOT/experiments/$EXPERIMENT}"

# run_train.sh owns the locked architecture/protocol.
"$PROJECT_ROOT/run_train.sh"
