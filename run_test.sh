#!/usr/bin/env bash
set -euo pipefail

# Canonical texture-only + SBI + EMA evaluation entry point.
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
    echo 'Create a Windows venv for Git Bash or a separate Linux venv for WSL.' >&2
    exit 1
fi

DATA_ROOT="$STORAGE_ROOT/data"
FFPP_FRAME_ROOT="$DATA_ROOT/extracted/ffpp"
FFPP_LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/ffpp-landmark"
FFPP_LANDMARK_ROOT="$FFPP_LANDMARK_OUTPUT_ROOT/landmarks"
FFPP_VAL_MANIFEST="$FFPP_LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl"
CELEBDF_FRAME_ROOT="$DATA_ROOT/extracted/celebdf"
CELEBDF_LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/celebdf-landmark"
CELEBDF_LANDMARK_ROOT="$CELEBDF_LANDMARK_OUTPUT_ROOT/landmarks"
CELEBDF_TEST_MANIFEST="$CELEBDF_LANDMARK_OUTPUT_ROOT/manifests/celebdf_test_landmarks.jsonl"
CHECKPOINT="${QALF_TEST_CHECKPOINT:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema/best.pt}"
# Canonical baseline evaluates with the same eight texture frames used in
# training. Set QALF_TEST_TEXTURE_FRAMES=12 explicitly for a separate ablation.
TEXTURE_FRAMES="${QALF_TEST_TEXTURE_FRAMES:-8}"
CLIPS_PER_VIDEO="${QALF_TEST_CLIPS_PER_VIDEO:-3}"
AGGREGATION="${QALF_TEST_AGGREGATION:-mean}"
TOP_K="${QALF_TEST_TOP_K:-1}"
THRESHOLD_CLIPS_PER_VIDEO="${QALF_TEST_THRESHOLD_CLIPS_PER_VIDEO:-3}"
FLIP_TTA="${QALF_TEST_FLIP_TTA:-1}"
OUTPUT_DIR="${QALF_TEST_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema_to_celebdf_${TEXTURE_FRAMES}f_${CLIPS_PER_VIDEO}clips_${AGGREGATION}_tta_ffpp_threshold}"

TTA_ARGS=()
if [[ "$FLIP_TTA" == "1" ]]; then
    TTA_ARGS+=(--texture-flip-tta)
fi

"$PYTHON" scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --manifest "$CELEBDF_TEST_MANIFEST" \
    --frame-root "$CELEBDF_FRAME_ROOT" \
    --landmark-root "$CELEBDF_LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 8 \
    --num-workers 4 \
    --clips-per-video "$CLIPS_PER_VIDEO" \
    --aggregation "$AGGREGATION" \
    --top-k "$TOP_K" \
    --texture-frames "$TEXTURE_FRAMES" \
    "${TTA_ARGS[@]}" \
    --threshold-manifest "$FFPP_VAL_MANIFEST" \
    --threshold-frame-root "$FFPP_FRAME_ROOT" \
    --threshold-landmark-root "$FFPP_LANDMARK_ROOT" \
    --threshold-clips-per-video "$THRESHOLD_CLIPS_PER_VIDEO"
