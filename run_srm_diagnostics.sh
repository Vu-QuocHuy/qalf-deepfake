#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
        STORAGE_ROOT='E:/DeepFakeData'
        ;;
    Linux*)
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
        STORAGE_ROOT='/mnt/e/DeepFakeData'
        ;;
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac

SEED="${QALF_SEED:-42}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: QALF_SEED must be a non-negative integer" >&2
    exit 2
fi
SEED_SUFFIX=''
if [[ "$SEED" != '42' ]]; then SEED_SUFFIX="_seed$SEED"; fi

EXPERIMENTS_ROOT="$STORAGE_ROOT/experiments"
EXPERIMENT="qalf_ffpp4_effb0_160_8f_full_face_sbi_srm${SEED_SUFFIX}"
CHECKPOINT="$EXPERIMENTS_ROOT/$EXPERIMENT/best.pt"
EVALUATION_DIR="$EXPERIMENTS_ROOT/${EXPERIMENT}_texture_inference_to_celebdf_12f_3clips_mean_tta_ffpp_threshold"
HISTORY="$EXPERIMENTS_ROOT/$EXPERIMENT/history.json"
OUTPUT_PREFIX="$EXPERIMENTS_ROOT/srm_complementarity_seed$SEED"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: SRM checkpoint is missing: $CHECKPOINT" >&2
    exit 3
fi

if [[ -f "$EVALUATION_DIR/metrics.json" && -f "$EVALUATION_DIR/threshold_predictions.csv" ]]; then
    echo "Reusing texture-inference evaluation: $EVALUATION_DIR"
else
    echo "Evaluating SRM-trained texture branch with its own FF++ threshold"
    QALF_TEST_CHECKPOINT="$CHECKPOINT" \
    QALF_TEST_OUTPUT_DIR="$EVALUATION_DIR" \
    QALF_TEXTURE_FRAMES=12 \
    QALF_SCORE_BRANCH=texture \
        "$PROJECT_ROOT/run_test.sh"
fi

ANALYSIS_ARGS=(
    --evaluation-dir "$EVALUATION_DIR"
    --output-prefix "$OUTPUT_PREFIX"
)
if [[ -f "$HISTORY" ]]; then
    ANALYSIS_ARGS+=(--history "$HISTORY")
fi
"$PYTHON" scripts/analyze_srm_complementarity.py "${ANALYSIS_ARGS[@]}"

echo "========================================================================"
echo "SRM diagnostic complete. Do not run staged training until this report is reviewed."
