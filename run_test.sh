#!/usr/bin/env bash
set -euo pipefail

# =========================== EDIT CONFIGURATION HERE ===========================
WINDOWS_DATA_ROOT='F:/DeepFakedata/outputs_duong_huy'
WSL_DATA_ROOT='/mnt/f/DeepFakedata/outputs_duong_huy'

BATCH_SIZE=16
NUM_WORKERS=4
CLIPS_PER_VIDEO=2
AGGREGATION='mean'
TOP_K=1
RUN_DATA_AUDIT=1
# ==============================================================================

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
        DATA_ROOT="$WINDOWS_DATA_ROOT"
        ;;
    Linux*)
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
        DATA_ROOT="$WSL_DATA_ROOT"
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

FRAME_ROOT="$DATA_ROOT/data/extracted/celebdf"
LANDMARK_OUTPUT_ROOT="$DATA_ROOT/data/landmarks/celebdf-landmark"
LANDMARK_ROOT="$LANDMARK_OUTPUT_ROOT/landmarks"
TEST_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/celebdf_test_landmarks.jsonl"
CHECKPOINT="$DATA_ROOT/experiments/qalf_ffpp/best.pt"
OUTPUT_DIR="$DATA_ROOT/experiments/qalf_ffpp_to_celebdf"

for required_path in "$TEST_MANIFEST" "$CHECKPOINT" "$FRAME_ROOT" "$LANDMARK_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required path does not exist: $required_path" >&2
        exit 1
    fi
done

echo "Python: $PYTHON"
echo "Checkpoint: $CHECKPOINT"
echo "Evaluation output: $OUTPUT_DIR"

if [[ "$RUN_DATA_AUDIT" == '1' ]]; then
    "$PYTHON" scripts/audit_manifest.py \
        --manifest "$TEST_MANIFEST" \
        --frame-root "$FRAME_ROOT" \
        --landmark-root "$LANDMARK_ROOT" \
        --expected-frames 64
fi

"$PYTHON" scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --manifest "$TEST_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --clips-per-video "$CLIPS_PER_VIDEO" \
    --aggregation "$AGGREGATION" \
    --top-k "$TOP_K"
