#!/usr/bin/env bash
set -euo pipefail

# Evaluate the seed-4 32-frame experiment with its matching sampling protocol.
WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
WSL_PROJECT_ROOT='/mnt/e/DeepFakeData'

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
        STORAGE_ROOT="$WINDOWS_PROJECT_ROOT"
        ;;
    Linux*)
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
        STORAGE_ROOT="$WSL_PROJECT_ROOT"
        ;;
    *)
        echo "ERROR: unsupported shell platform: $(uname -s)" >&2
        exit 1
        ;;
esac
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtual-environment Python not found: $PYTHON" >&2
    exit 1
fi

DATA_ROOT="$STORAGE_ROOT/data"
FFPP_FRAME_ROOT="${QALF_FFPP_FRAME_ROOT:-$DATA_ROOT/extracted/ffpp}"
FFPP_LANDMARK_OUTPUT_ROOT="${QALF_FFPP_32F_LANDMARK_OUTPUT:-$DATA_ROOT/landmarks/ffpp-landmark-32f-stride2}"
CELEBDF_FRAME_ROOT="${QALF_CELEBDF_FRAME_ROOT:-$DATA_ROOT/extracted/celebdf}"
CELEBDF_LANDMARK_OUTPUT_ROOT="${QALF_CELEBDF_32F_LANDMARK_OUTPUT:-$DATA_ROOT/landmarks/celebdf-landmark-32f-stride2}"
CHECKPOINT="${QALF_TEST_CHECKPOINT:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_32f5fps_16w_8f_texture_sbi_ema_seed4/best.pt}"
CLIPS_PER_VIDEO="${QALF_TEST_CLIPS_PER_VIDEO:-3}"
AGGREGATION="${QALF_TEST_AGGREGATION:-mean}"
TOP_K="${QALF_TEST_TOP_K:-1}"
THRESHOLD_SELECTION="${QALF_THRESHOLD_SELECTION:-eer}"
FLIP_TTA="${QALF_TEST_FLIP_TTA:-1}"
OUTPUT_DIR="${QALF_TEST_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_32f5fps_16w_8f_texture_sbi_ema_seed4_to_celebdf_8f_${CLIPS_PER_VIDEO}clips_${AGGREGATION}_${THRESHOLD_SELECTION}_tta_ffpp_threshold}"

TTA_ARGS=()
if [[ "$FLIP_TTA" == "1" ]]; then
    TTA_ARGS+=(--texture-flip-tta)
fi

"$PYTHON" scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --manifest "$CELEBDF_LANDMARK_OUTPUT_ROOT/manifests/celebdf_test_landmarks.jsonl" \
    --frame-root "$CELEBDF_FRAME_ROOT" \
    --landmark-root "$CELEBDF_LANDMARK_OUTPUT_ROOT/landmarks" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 8 \
    --num-workers 4 \
    --clips-per-video "$CLIPS_PER_VIDEO" \
    --aggregation "$AGGREGATION" \
    --top-k "$TOP_K" \
    --texture-frames 8 \
    "${TTA_ARGS[@]}" \
    --threshold-manifest "$FFPP_LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl" \
    --threshold-frame-root "$FFPP_FRAME_ROOT" \
    --threshold-landmark-root "$FFPP_LANDMARK_OUTPUT_ROOT/landmarks" \
    --threshold-clips-per-video 3 \
    --threshold-selection "$THRESHOLD_SELECTION"
