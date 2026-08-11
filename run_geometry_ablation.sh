#!/usr/bin/env bash
set -euo pipefail

# Run isolated geometry ablations from the established EfficientNet-B0/full-face/SBI
# baseline. No profile inherits a change from another ablation profile.
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

MODE="${1:-all}"
case "$MODE" in
    train|test|all) ;;
    *)
        echo "ERROR: mode must be train, test, or all; got '$MODE'" >&2
        exit 2
        ;;
esac

PROFILES=(
    geometry_i1_attentive
    geometry_i2_reliability
    geometry_i3_attentive_reliability
)
BASELINE_PROFILE='full_face_sbi'
BASELINE_CHECKPOINT="$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_full_face_sbi/best.pt"

echo "Geometry ablation mode: $MODE"
echo "Baseline: $BASELINE_PROFILE"
echo "Profiles: ${PROFILES[*]}"

if [[ "$MODE" == train || "$MODE" == all ]]; then
    if [[ -f "$BASELINE_CHECKPOINT" ]]; then
        echo "========================================================================"
        echo "BASELINE: keeping existing checkpoint: $BASELINE_CHECKPOINT"
    else
        echo "========================================================================"
        echo "BASELINE: training $BASELINE_PROFILE"
        echo "========================================================================"
        "$PROJECT_ROOT/run_train_cross_dataset.sh" "$BASELINE_PROFILE"
    fi
fi

if [[ "$MODE" == test || "$MODE" == all ]]; then
    echo "========================================================================"
    echo "BASELINE TEST: $BASELINE_PROFILE"
    echo "========================================================================"
    "$PROJECT_ROOT/run_test_cross_dataset.sh" "$BASELINE_PROFILE"
fi

for profile in "${PROFILES[@]}"; do
    echo "========================================================================"
    echo "PROFILE: $profile"
    echo "========================================================================"
    if [[ "$MODE" == train || "$MODE" == all ]]; then
        "$PROJECT_ROOT/run_train_cross_dataset.sh" "$profile"
    fi
    if [[ "$MODE" == test || "$MODE" == all ]]; then
        "$PROJECT_ROOT/run_test_cross_dataset.sh" "$profile"
    fi
done

if [[ "$MODE" == test || "$MODE" == all ]]; then
    "$PYTHON" scripts/summarize_geometry_ablation.py \
        --experiments-root "$STORAGE_ROOT/experiments"
fi

echo "========================================================================"
echo "Geometry ablation suite complete."
