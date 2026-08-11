#!/usr/bin/env bash
set -euo pipefail

WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
WSL_PROJECT_ROOT='/mnt/e/DeepFakeData'
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) STORAGE_ROOT="$WINDOWS_PROJECT_ROOT" ;;
    Linux*) STORAGE_ROOT="$WSL_PROJECT_ROOT" ;;
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac

PROFILE="${1:-dual_view}"
case "$PROFILE" in
    full_face)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_deterministic'
        TEST_TEXTURE_FRAMES=12
        ;;
    full_face_ema)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_ema'
        TEST_TEXTURE_FRAMES=12
        ;;
    full_face_mixstyle)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_mixstyle'
        TEST_TEXTURE_FRAMES=12
        ;;
    full_face_dynamics)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_dynamics'
        TEST_TEXTURE_FRAMES=12
        ;;
    dual_view)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_dual_view'
        TEST_TEXTURE_FRAMES=12
        ;;
    *)
        echo "ERROR: unknown profile '$PROFILE'" >&2
        echo 'Use: full_face, full_face_ema, full_face_mixstyle, full_face_dynamics, or dual_view' >&2
        exit 2
        ;;
esac

export QALF_TEST_CHECKPOINT="$STORAGE_ROOT/experiments/$EXPERIMENT/best.pt"
export QALF_TEST_OUTPUT_DIR="$STORAGE_ROOT/experiments/${EXPERIMENT}_to_celebdf_${TEST_TEXTURE_FRAMES}f_3clips_mean_tta_ffpp_threshold"
export QALF_TEXTURE_FRAMES="$TEST_TEXTURE_FRAMES"

echo "Cross-dataset profile: $PROFILE"
"$PROJECT_ROOT/run_test.sh"
