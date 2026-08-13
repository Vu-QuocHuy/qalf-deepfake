#!/usr/bin/env bash
set -euo pipefail

# Evaluate the SRM-preprocessed checkpoint with the canonical eight-frame
# train/eval protocol. All paths can be overridden through QALF_TEST_* vars.
WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
CHECKPOINT="${QALF_TEST_CHECKPOINT:-$WINDOWS_PROJECT_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_srm_ema/best.pt}"
OUTPUT_DIR="${QALF_TEST_OUTPUT_DIR:-$WINDOWS_PROJECT_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_srm_ema_to_celebdf_8f_3clips_mean_tta}"
QALF_TEST_CHECKPOINT="$CHECKPOINT" \
QALF_TEST_OUTPUT_DIR="$OUTPUT_DIR" \
QALF_TEST_TEXTURE_FRAMES="${QALF_TEST_TEXTURE_FRAMES:-8}" \
    ./run_test.sh
