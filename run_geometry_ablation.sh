#!/usr/bin/env bash
set -euo pipefail

# Compare the retained candidate and required texture-only control against SBI.
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
    geometry_candidate
    texture_only_sbi
)
BASELINE_PROFILE='full_face_sbi'
SEED="${QALF_SEED:-42}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: QALF_SEED must be a non-negative integer; got '$SEED'" >&2
    exit 2
fi
BASELINE_EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi'
if [[ "$SEED" != '42' ]]; then
    BASELINE_EXPERIMENT="${BASELINE_EXPERIMENT}_seed${SEED}"
fi
BASELINE_CHECKPOINT="$STORAGE_ROOT/experiments/$BASELINE_EXPERIMENT/best.pt"

echo "P1 texture-control comparison mode: $MODE"
echo "Baseline: $BASELINE_PROFILE"
echo "Seed: $SEED"
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
        case "$profile" in
            geometry_candidate)
                PROFILE_EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_i3_attentive_reliability'
                ;;
            texture_only_sbi)
                PROFILE_EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi_texture_only'
                ;;
        esac
        if [[ "$SEED" != '42' ]]; then
            PROFILE_EXPERIMENT="${PROFILE_EXPERIMENT}_seed${SEED}"
        fi
        PROFILE_CHECKPOINT="$STORAGE_ROOT/experiments/$PROFILE_EXPERIMENT/best.pt"
        if [[ -f "$PROFILE_CHECKPOINT" ]]; then
            echo "PROFILE: keeping existing checkpoint: $PROFILE_CHECKPOINT"
        else
            "$PROJECT_ROOT/run_train_cross_dataset.sh" "$profile"
        fi
    fi
    if [[ "$MODE" == test || "$MODE" == all ]]; then
        "$PROJECT_ROOT/run_test_cross_dataset.sh" "$profile"
    fi
done

if [[ "$MODE" == test || "$MODE" == all ]]; then
    "$PYTHON" scripts/summarize_geometry_ablation.py \
        --experiments-root "$STORAGE_ROOT/experiments" \
        --seed "$SEED"
fi

echo "========================================================================"
echo "P1 texture-control comparison complete."
