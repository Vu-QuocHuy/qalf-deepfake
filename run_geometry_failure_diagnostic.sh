#!/usr/bin/env bash
set -euo pipefail

# Isolate dropout, reliability, and SBI-aware routing on the locked SBI baseline.
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
SBI_AWARE_EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_sbi_aware_reliability'

is_complete_checkpoint() {
    local experiment="$1"
    local checkpoint="$EXPERIMENTS_ROOT/$experiment/best.pt"
    local summary="$EXPERIMENTS_ROOT/$experiment/training_summary.json"
    [[ -f "$checkpoint" ]] || return 1
    if [[ ! -f "$summary" ]]; then
        echo "Recovering missing training summary from: $checkpoint"
        "$PYTHON" -c '
import json
import os
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
metrics = checkpoint.get("validation_metrics", {})
best_value = metrics.get("auc")
if best_value is None:
    raise SystemExit(f"Checkpoint has no validation AUC: {checkpoint_path}")
payload = {
    "status": "complete",
    "recovered_from_checkpoint": True,
    "completed_epochs": int(checkpoint.get("epoch", 0)),
    "best_epoch": int(checkpoint.get("epoch", 0)),
    "best_metric": str(checkpoint.get("best_metric", "val_auc")),
    "best_value": float(best_value),
    "best_threshold": float(checkpoint["threshold"]),
    "best_validation_metrics": metrics,
    "best_model": str(checkpoint_path),
}
temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
os.replace(temporary, summary_path)
' "$checkpoint" "$summary"
    fi
    [[ -f "$summary" ]]
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

evaluate_if_needed() {
    local experiment="$1"
    local profile="$2"
    local label="$3"
    local metrics_path="$EXPERIMENTS_ROOT/${experiment}${EVALUATION_SUFFIX}/metrics.json"
    if has_counterfactual_diagnostics "$metrics_path"; then
        echo "$label: reusing existing evaluation: $metrics_path"
    else
        echo "$label: diagnostics missing; evaluating existing checkpoint"
        "$PROJECT_ROOT/run_test_cross_dataset.sh" "$profile"
    fi
}

echo "Geometry failure diagnostic mode: $MODE"
echo "Seed: 42 (locked)"
echo "A: SBI baseline                 dropout=0.00 reliability=0.00"
echo "C: modality dropout only       dropout=0.15 reliability=0.00"
echo "D: dropout + reliability       dropout=0.15 reliability=0.10"
echo "E: SBI-aware reliability       dropout=0.15 reliability=0.10 SBI excluded"

if [[ "$MODE" == all ]]; then
    # The locked baseline must already exist; missing diagnostic controls are
    # trained below so a single `all` command remains self-contained.
    require_complete_checkpoint "$BASELINE_EXPERIMENT" full_face_sbi
fi

if [[ "$MODE" == train || "$MODE" == all ]]; then
    if is_complete_checkpoint "$DROPOUT_EXPERIMENT"; then
        echo "C: keeping completed checkpoint: $EXPERIMENTS_ROOT/$DROPOUT_EXPERIMENT/best.pt"
    else
        echo "C: completed checkpoint missing; training modality-dropout-only control"
        "$PROJECT_ROOT/run_train_cross_dataset.sh" geometry_dropout_only
    fi

    if is_complete_checkpoint "$COMBINED_EXPERIMENT"; then
        echo "D: keeping completed checkpoint: $EXPERIMENTS_ROOT/$COMBINED_EXPERIMENT/best.pt"
    else
        echo "D: completed checkpoint missing; training dropout-plus-reliability control"
        "$PROJECT_ROOT/run_train_cross_dataset.sh" geometry_reliability_combined
    fi

    if is_complete_checkpoint "$SBI_AWARE_EXPERIMENT"; then
        echo "E: keeping completed checkpoint: $EXPERIMENTS_ROOT/$SBI_AWARE_EXPERIMENT/best.pt"
    else
        echo "E: completed checkpoint missing; training SBI-aware reliability control"
        "$PROJECT_ROOT/run_train_cross_dataset.sh" geometry_sbi_aware_reliability
    fi
fi

if [[ "$MODE" == test || "$MODE" == all ]]; then
    require_complete_checkpoint "$BASELINE_EXPERIMENT" full_face_sbi
    require_complete_checkpoint "$DROPOUT_EXPERIMENT" geometry_dropout_only
    require_complete_checkpoint "$COMBINED_EXPERIMENT" geometry_reliability_combined
    require_complete_checkpoint "$SBI_AWARE_EXPERIMENT" geometry_sbi_aware_reliability

    evaluate_if_needed "$BASELINE_EXPERIMENT" full_face_sbi A
    evaluate_if_needed "$DROPOUT_EXPERIMENT" geometry_dropout_only C
    evaluate_if_needed "$COMBINED_EXPERIMENT" geometry_reliability_combined D
    evaluate_if_needed "$SBI_AWARE_EXPERIMENT" geometry_sbi_aware_reliability E

    "$PYTHON" scripts/summarize_geometry_failure.py \
        --experiments-root "$EXPERIMENTS_ROOT"
fi

echo "========================================================================"
echo "Geometry failure diagnostic complete."
