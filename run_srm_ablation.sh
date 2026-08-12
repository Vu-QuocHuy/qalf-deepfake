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

MODE="${1:-all}"
case "$MODE" in train|test|all) ;; *) echo "Use: train, test, or all" >&2; exit 2 ;; esac
SEED="${QALF_SEED:-42}"
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: QALF_SEED must be a non-negative integer" >&2
    exit 2
fi

EXPERIMENTS_ROOT="$STORAGE_ROOT/experiments"
SEED_SUFFIX=''
if [[ "$SEED" != '42' ]]; then SEED_SUFFIX="_seed$SEED"; fi
TEXTURE_EXPERIMENT="qalf_ffpp4_effb0_160_8f_full_face_sbi_texture_only${SEED_SUFFIX}"
GEOMETRY_EXPERIMENT="qalf_ffpp4_effb0_160_8f_full_face_sbi${SEED_SUFFIX}"
SRM_EXPERIMENT="qalf_ffpp4_effb0_160_8f_full_face_sbi_srm${SEED_SUFFIX}"
EVALUATION_SUFFIX='_to_celebdf_12f_3clips_mean_tta_ffpp_threshold'

has_checkpoint() {
    [[ -f "$EXPERIMENTS_ROOT/$1/best.pt" ]]
}

complete_training() {
    has_checkpoint "$1" && [[ -f "$EXPERIMENTS_ROOT/$1/training_summary.json" ]]
}

has_evaluation() {
    [[ -f "$EXPERIMENTS_ROOT/${1}${EVALUATION_SUFFIX}/metrics.json" ]]
}

ensure_profile_available() {
    local profile="$1"
    local experiment="$2"
    if has_evaluation "$experiment"; then
        echo "$profile: completed evaluation already available"
    elif complete_training "$experiment"; then
        echo "$profile: keeping completed checkpoint: $EXPERIMENTS_ROOT/$experiment/best.pt"
    elif has_checkpoint "$experiment"; then
        echo "$profile: checkpoint available; training summary is optional for evaluation"
    else
        echo "$profile: neither evaluation nor checkpoint exists; training now"
        "$PROJECT_ROOT/run_train_cross_dataset.sh" "$profile"
    fi
}

require_checkpoint_for_evaluation() {
    if ! has_checkpoint "$1"; then
        echo "ERROR: evaluation is missing and no checkpoint is available:" >&2
        echo "  $EXPERIMENTS_ROOT/$1/best.pt" >&2
        exit 3
    fi
}

echo "SRM ablation mode: $MODE"
echo "Seed: $SEED"
echo "Controls: texture-only SBI, geometry baseline A"
echo "Candidate: fixed SRM residual bank + lightweight CNN, no dropout/reliability"

if [[ "$MODE" == train || "$MODE" == all ]]; then
    ensure_profile_available texture_only_sbi "$TEXTURE_EXPERIMENT"
    ensure_profile_available full_face_sbi "$GEOMETRY_EXPERIMENT"
    ensure_profile_available srm_sbi "$SRM_EXPERIMENT"
fi

if [[ "$MODE" == test || "$MODE" == all ]]; then
    for entry in \
        "texture_only_sbi:$TEXTURE_EXPERIMENT" \
        "full_face_sbi:$GEOMETRY_EXPERIMENT" \
        "srm_sbi:$SRM_EXPERIMENT"
    do
        profile="${entry%%:*}"
        experiment="${entry#*:}"
        metrics="$EXPERIMENTS_ROOT/${experiment}${EVALUATION_SUFFIX}/metrics.json"
        if [[ -f "$metrics" ]]; then
            echo "$profile: reusing existing evaluation: $metrics"
        else
            require_checkpoint_for_evaluation "$experiment"
            "$PROJECT_ROOT/run_test_cross_dataset.sh" "$profile"
        fi
    done
    "$PYTHON" scripts/summarize_srm_ablation.py \
        --experiments-root "$EXPERIMENTS_ROOT" \
        --seed "$SEED"
fi

echo "========================================================================"
echo "SRM ablation complete."
