#!/usr/bin/env bash
set -euo pipefail

# Isolate modality dropout from reliability supervision on the locked SBI baseline.
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
    *) echo "ERROR: mode must be train, test, or all; got '$MODE'" >&2; exit 2 ;;
esac

SEED="${QALF_SEED:-42}"
if [[ "$SEED" != '42' ]]; then
    echo "ERROR: this diagnostic is intentionally locked to QALF_SEED=42; got '$SEED'" >&2
    exit 2
fi

EXPERIMENTS_ROOT="$STORAGE_ROOT/experiments"
EVALUATION_SUFFIX='_to_celebdf_12f_3clips_mean_tta_ffpp_threshold'
BASELINE_EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi'
DROPOUT_EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_dropout_only'
COMBINED_EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_i2_reliability'

is_complete_checkpoint() {
    local experiment="$1"
    [[ -f "$EXPERIMENTS_ROOT/$experiment/best.pt" ]] && \
        [[ -f "$EXPERIMENTS_ROOT/$experiment/training_summary.json" ]]
}

require_complete_checkpoint() {
    local experiment="$1"
    local profile="$2"
    if ! is_complete_checkpoint "$experiment"; then
        echo "ERROR: completed checkpoint required for $profile:" >&2
        echo "  $EXPERIMENTS_ROOT/$experiment" >&2
        echo "Expected both best.pt and training_summary.json." >&2
        echo "If the historical checkpoint is unavailable, train it explicitly with:" >&2
        echo "  ./run_train_cross_dataset.sh $profile" >&2
        exit 3
    fi
}

has_counterfactual_diagnostics() {
    local metrics_path="$1"
    [[ -f "$metrics_path" ]] || return 1
    "$PYTHON" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
metrics = payload.get("metrics", payload)
required = {
    "zero_geometry_auc",
    "auc_gain_over_zero_geometry",
    "mean_abs_zero_geometry_score_shift",
    "median_geometry_weight",
    "p90_geometry_weight",
}
raise SystemExit(0 if required.issubset(metrics) else 1)
' "$metrics_path"
}

echo "Geometry failure diagnostic mode: $MODE"
echo "Seed: 42 (locked)"
echo "A: SBI baseline                 dropout=0.00 reliability=0.00"
echo "C: modality dropout only       dropout=0.15 reliability=0.00"
echo "D: dropout + reliability       dropout=0.15 reliability=0.10"

if [[ "$MODE" == all ]]; then
    # Fail before the only new training job if either historical control is absent.
    require_complete_checkpoint "$BASELINE_EXPERIMENT" full_face_sbi
    require_complete_checkpoint "$COMBINED_EXPERIMENT" geometry_reliability_combined
fi

if [[ "$MODE" == train || "$MODE" == all ]]; then
    if is_complete_checkpoint "$DROPOUT_EXPERIMENT"; then
        echo "C: keeping completed checkpoint: $EXPERIMENTS_ROOT/$DROPOUT_EXPERIMENT/best.pt"
    else
        "$PROJECT_ROOT/run_train_cross_dataset.sh" geometry_dropout_only
    fi
fi

if [[ "$MODE" == test || "$MODE" == all ]]; then
    require_complete_checkpoint "$BASELINE_EXPERIMENT" full_face_sbi
    require_complete_checkpoint "$DROPOUT_EXPERIMENT" geometry_dropout_only
    require_complete_checkpoint "$COMBINED_EXPERIMENT" geometry_reliability_combined

    BASELINE_METRICS="$EXPERIMENTS_ROOT/${BASELINE_EXPERIMENT}${EVALUATION_SUFFIX}/metrics.json"
    if ! has_counterfactual_diagnostics "$BASELINE_METRICS"; then
        echo "A: baseline diagnostics missing; evaluating existing checkpoint"
        "$PROJECT_ROOT/run_test_cross_dataset.sh" full_face_sbi
    else
        echo "A: reusing existing evaluation: $BASELINE_METRICS"
    fi

    "$PROJECT_ROOT/run_test_cross_dataset.sh" geometry_dropout_only
    "$PROJECT_ROOT/run_test_cross_dataset.sh" geometry_reliability_combined

    "$PYTHON" scripts/summarize_geometry_failure.py \
        --experiments-root "$EXPERIMENTS_ROOT"
fi

echo "========================================================================"
echo "Geometry failure diagnostic complete."
