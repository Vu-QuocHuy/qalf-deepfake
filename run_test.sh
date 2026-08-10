#!/usr/bin/env bash
set -euo pipefail

# Shared storage roots. Evaluation options are edited directly in the command below.
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
FRAME_ROOT="$DATA_ROOT/extracted/celebdf"
LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/celebdf-landmark"
LANDMARK_ROOT="$LANDMARK_OUTPUT_ROOT/landmarks"
TEST_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/celebdf_test_landmarks.jsonl"
CHECKPOINT="$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_clean_160_8f/best.pt"
OUTPUT_DIR="$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_clean_160_8f_to_celebdf"

for required_path in "$TEST_MANIFEST" "$CHECKPOINT" "$FRAME_ROOT" "$LANDMARK_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required path does not exist: $required_path" >&2
        exit 1
    fi
done

echo "Python: $PYTHON"
echo "Checkpoint: $CHECKPOINT"
echo "Evaluation output: $OUTPUT_DIR"

"$PYTHON" scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --manifest "$TEST_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 8 \
    --num-workers 4 \
    --clips-per-video 3 \
    --aggregation mean \
    --top-k 1
