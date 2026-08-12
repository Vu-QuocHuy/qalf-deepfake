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

PROFILE="${1:-full_face_sbi}"
case "$PROFILE" in
    full_face_sbi)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi'
        TEST_TEXTURE_FRAMES=12
        ;;
    texture_only_sbi)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi_texture_only'
        TEST_TEXTURE_FRAMES=12
        ;;
    srm_sbi)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi_srm'
        TEST_TEXTURE_FRAMES=12
        ;;
    learned_srm)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi_learned_srm'
        TEST_TEXTURE_FRAMES=12
        ;;
    *)
        echo "ERROR: unknown profile '$PROFILE'" >&2
        echo 'Use: full_face_sbi, texture_only_sbi, srm_sbi, or learned_srm' >&2
        exit 2
        ;;
esac

SEED="${QALF_SEED:-42}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: QALF_SEED must be a non-negative integer; got '$SEED'" >&2
    exit 2
fi
if [[ "$SEED" != '42' ]]; then
    EXPERIMENT="${EXPERIMENT}_seed${SEED}"
fi

export QALF_TEST_CHECKPOINT="$STORAGE_ROOT/experiments/$EXPERIMENT/best.pt"
export QALF_TEST_OUTPUT_DIR="$STORAGE_ROOT/experiments/${EXPERIMENT}_to_celebdf_${TEST_TEXTURE_FRAMES}f_3clips_mean_tta_ffpp_threshold"
export QALF_TEXTURE_FRAMES="$TEST_TEXTURE_FRAMES"

echo "Cross-dataset profile: $PROFILE"
echo "Seed: $SEED"
"$PROJECT_ROOT/run_test.sh"
