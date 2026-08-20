#!/usr/bin/env bash
set -euo pipefail

# Evaluation-only robustness protocol for the TextureSBI baseline.
WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
WSL_PROJECT_ROOT='/mnt/e/DeepFakeData'
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"; STORAGE_ROOT="$WINDOWS_PROJECT_ROOT" ;;
  Linux*) PYTHON="$PROJECT_ROOT/.venv/bin/python"; STORAGE_ROOT="$WSL_PROJECT_ROOT" ;;
  *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac
[[ -x "$PYTHON" ]] || { echo "ERROR: virtual-environment Python not found: $PYTHON" >&2; exit 1; }
DATA_ROOT="$STORAGE_ROOT/data"
FFPP_ROOT="$DATA_ROOT/landmarks/ffpp-landmark"
CELEBDF_ROOT="$DATA_ROOT/landmarks/celebdf-landmark"
CHECKPOINT="${QALF_ROBUSTNESS_CHECKPOINT:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema/best.pt}"
TEXTURE_FRAMES="${QALF_ROBUSTNESS_TEXTURE_FRAMES:-8}"
THRESHOLD_SELECTION="${QALF_THRESHOLD_SELECTION:-eer}"
OUTPUT="${QALF_ROBUSTNESS_OUTPUT:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema_${TEXTURE_FRAMES}f_${THRESHOLD_SELECTION}_robustness.json}"
"$PYTHON" scripts/evaluate_robustness.py \
  --checkpoint "$CHECKPOINT" \
  --manifest "$CELEBDF_ROOT/manifests/celebdf_test_landmarks.jsonl" \
  --frame-root "$DATA_ROOT/extracted/celebdf" \
  --landmark-root "$CELEBDF_ROOT/landmarks" \
  --output "$OUTPUT" --texture-frames "$TEXTURE_FRAMES" --clips-per-video 3 \
  --aggregation mean --top-k 1 --texture-flip-tta \
  --threshold-manifest "$FFPP_ROOT/manifests/ffpp_val_landmarks.jsonl" \
  --threshold-frame-root "$DATA_ROOT/extracted/ffpp" \
  --threshold-landmark-root "$FFPP_ROOT/landmarks" \
  --threshold-clips-per-video 3 \
  --threshold-selection "$THRESHOLD_SELECTION"
