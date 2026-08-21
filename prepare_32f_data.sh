#!/usr/bin/env bash
set -euo pipefail

# Derive 32-frame/approximately-5-FPS manifests and landmark caches by taking
# indices 0,2,...,62 from the canonical 64-frame/10-FPS data. JPEGs are reused.
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
FFPP_SOURCE="${QALF_FFPP_LANDMARK_SOURCE:-$DATA_ROOT/landmarks/ffpp-landmark}"
FFPP_OUTPUT="${QALF_FFPP_32F_LANDMARK_OUTPUT:-$DATA_ROOT/landmarks/ffpp-landmark-32f-stride2}"
CELEBDF_SOURCE="${QALF_CELEBDF_LANDMARK_SOURCE:-$DATA_ROOT/landmarks/celebdf-landmark}"
CELEBDF_OUTPUT="${QALF_CELEBDF_32F_LANDMARK_OUTPUT:-$DATA_ROOT/landmarks/celebdf-landmark-32f-stride2}"

derive_manifest() {
    local input_manifest="$1"
    local input_landmarks="$2"
    local output_manifest="$3"
    local output_landmarks="$4"
    if [[ ! -f "$input_manifest" ]]; then
        echo "ERROR: source manifest not found: $input_manifest" >&2
        exit 1
    fi
    "$PYTHON" scripts/subsample_sequences.py \
        --input-manifest "$input_manifest" \
        --input-landmark-root "$input_landmarks" \
        --output-manifest "$output_manifest" \
        --output-landmark-root "$output_landmarks" \
        --source-frames 64 \
        --stride 2 \
        --offset 0
}

for split in train val; do
    derive_manifest \
        "$FFPP_SOURCE/manifests/ffpp_${split}_landmarks.jsonl" \
        "$FFPP_SOURCE/landmarks" \
        "$FFPP_OUTPUT/manifests/ffpp_${split}_landmarks.jsonl" \
        "$FFPP_OUTPUT/landmarks"
done

derive_manifest \
    "$CELEBDF_SOURCE/manifests/celebdf_test_landmarks.jsonl" \
    "$CELEBDF_SOURCE/landmarks" \
    "$CELEBDF_OUTPUT/manifests/celebdf_test_landmarks.jsonl" \
    "$CELEBDF_OUTPUT/landmarks"

echo "32-frame derived data is ready"
echo "FF++: $FFPP_OUTPUT"
echo "Celeb-DF: $CELEBDF_OUTPUT"
