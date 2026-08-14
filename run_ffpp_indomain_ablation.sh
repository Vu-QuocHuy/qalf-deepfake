#!/usr/bin/env bash
set -euo pipefail

# Evaluate already-trained ablation checkpoints on the official FF++ test split.
# This script deliberately does not train. It filters the target and calibration
# data to the four registered methods and therefore excludes FaceShifter.

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
    *) echo "ERROR: unsupported shell platform: $(uname -s)" >&2; exit 1 ;;
esac
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: virtual-environment Python not found: $PYTHON" >&2
    exit 1
fi

DATA_ROOT="${QALF_DATA_ROOT:-$STORAGE_ROOT/data}"
ABLATION_ROOT="${QALF_ABLATION_ROOT:-$STORAGE_ROOT/experiments/ablation}"
OUTPUT_ROOT="${QALF_FFPP_INDOMAIN_ROOT:-$ABLATION_ROOT/ffpp_test}"
TEST_MANIFEST="${QALF_FFPP_TEST_MANIFEST:-$DATA_ROOT/landmarks/ffpp-landmark/manifests/ffpp_test_landmarks.jsonl}"
VAL_MANIFEST="${QALF_FFPP_VAL_MANIFEST:-$DATA_ROOT/landmarks/ffpp-landmark/manifests/ffpp_val_landmarks.jsonl}"
FRAME_ROOT="${QALF_FFPP_FRAME_ROOT:-$DATA_ROOT/extracted/ffpp}"
LANDMARK_ROOT="${QALF_FFPP_LANDMARK_ROOT:-$DATA_ROOT/landmarks/ffpp-landmark/landmarks}"
SEEDS_RAW="${QALF_FFPP_INDOMAIN_SEEDS:-0 17 42 73 123}"
TEXTURE_FRAMES="${QALF_FFPP_TEXTURE_FRAMES:-8}"
CLIPS_PER_VIDEO="${QALF_FFPP_CLIPS_PER_VIDEO:-3}"
AGGREGATION="${QALF_FFPP_AGGREGATION:-mean}"
TOP_K="${QALF_FFPP_TOP_K:-1}"
THRESHOLD_SELECTION="${QALF_THRESHOLD_SELECTION:-youden_j}"
THRESHOLD_CLIPS="${QALF_FFPP_THRESHOLD_CLIPS_PER_VIDEO:-3}"
FLIP_TTA="${QALF_FFPP_FLIP_TTA:-1}"
NUM_WORKERS="${QALF_FFPP_NUM_WORKERS:-4}"
BATCH_SIZE="${QALF_FFPP_BATCH_SIZE:-8}"
FORCE_EVAL="${QALF_FFPP_FORCE_EVAL:-0}"
BOOTSTRAP="${QALF_FFPP_BOOTSTRAP:-1}"
BOOTSTRAP_REPS="${QALF_FFPP_BOOTSTRAP_REPS:-2000}"
if [[ "$FLIP_TTA" == 1 ]]; then
    TTA_SUFFIX="tta"
else
    TTA_SUFFIX="no_tta"
fi
EVAL_SUFFIX="_ffpp_test_${TEXTURE_FRAMES}f_${CLIPS_PER_VIDEO}clips_${AGGREGATION}_${THRESHOLD_SELECTION}_${TTA_SUFFIX}"

FAKE_METHODS=(Deepfakes Face2Face FaceSwap NeuralTextures)
read -r -a SEEDS <<< "$SEEDS_RAW"
if (( ${#SEEDS[@]} == 0 )); then
    echo "ERROR: QALF_FFPP_INDOMAIN_SEEDS is empty" >&2
    exit 1
fi
for required in "$TEST_MANIFEST" "$VAL_MANIFEST" "$FRAME_ROOT" "$LANDMARK_ROOT"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: required FF++ input is missing: $required" >&2
        exit 1
    fi
done
if [[ ! -f "$TEST_MANIFEST" ]]; then
    echo "ERROR: official FF++ test manifest was not found: $TEST_MANIFEST" >&2
    echo "Set QALF_FFPP_TEST_MANIFEST to the official FF++ test JSONL." >&2
    echo "Do not substitute ffpp_val_landmarks.jsonl: validation is used for threshold calibration." >&2
    exit 1
fi
"$PYTHON" - "$TEST_MANIFEST" <<'PY'
import sys
from qalf.data.manifest import load_manifest

records = load_manifest(sys.argv[1])
datasets = sorted({record.dataset for record in records})
splits = sorted({record.split for record in records})
if datasets != ["ffpp"] or splits != ["test"]:
    raise SystemExit(
        "ERROR: --manifest must be the official FF++ test split; "
        f"got datasets={datasets}, splits={splits}."
    )
print(f"Validated FF++ test manifest: videos={len(records)}")
PY

declare -a RUN_SPECS=()
for spec in \
    "baseline:$ABLATION_ROOT/baseline_seed" \
    "no_sbi:$ABLATION_ROOT/no_sbi_seed" \
    "no_ema:$ABLATION_ROOT/no_ema_seed" \
    "no_pretrain:$ABLATION_ROOT/no_pretrain_seed" \
    "no_aug:$ABLATION_ROOT/no_aug_seed" \
    "sbi_half:$ABLATION_ROOT/sbi_half_seed"; do
    profile="${spec%%:*}"
    prefix="${spec#*:}"
    for seed in "${SEEDS[@]}"; do
        if [[ "$profile" == no_pretrain || "$profile" == no_aug || "$profile" == sbi_half ]] && [[ "$seed" != "42" ]]; then
            continue
        fi
        RUN_SPECS+=("$profile:$seed:$prefix")
    done
done

echo "QALF FF++ in-domain ablation evaluation"
echo "Target: $TEST_MANIFEST"
echo "Threshold calibration: $VAL_MANIFEST ($THRESHOLD_SELECTION)"
echo "Fake methods: ${FAKE_METHODS[*]} (FaceShifter excluded)"
echo "Output: $OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT"
SUMMARY_ARGS=()
for run in "${RUN_SPECS[@]}"; do
    profile="${run%%:*}"
    rest="${run#*:}"
    seed="${rest%%:*}"
    prefix="${rest#*:}"
    checkpoint="${prefix}${seed}/best.pt"
    output_dir="$OUTPUT_ROOT/${profile}_seed${seed}${EVAL_SUFFIX}"
    SUMMARY_ARGS+=(--run "${profile}_seed${seed}=${output_dir}")
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: checkpoint missing for ${profile}/seed${seed}: $checkpoint" >&2
        exit 1
    fi
    if [[ "$FORCE_EVAL" != 1 && -f "$output_dir/metrics.json" ]]; then
        echo "[${profile}/seed${seed}] metrics exist; skipping"
    else
        echo "[${profile}/seed${seed}] evaluating"
        args=(
            scripts/evaluate.py
            --checkpoint "$checkpoint"
            --manifest "$TEST_MANIFEST"
            --frame-root "$FRAME_ROOT"
            --landmark-root "$LANDMARK_ROOT"
            --output-dir "$output_dir"
            --batch-size "$BATCH_SIZE"
            --num-workers "$NUM_WORKERS"
            --clips-per-video "$CLIPS_PER_VIDEO"
            --texture-frames "$TEXTURE_FRAMES"
            --aggregation "$AGGREGATION"
            --top-k "$TOP_K"
            --threshold-manifest "$VAL_MANIFEST"
            --threshold-frame-root "$FRAME_ROOT"
            --threshold-landmark-root "$LANDMARK_ROOT"
            --threshold-clips-per-video "$THRESHOLD_CLIPS"
            --threshold-selection "$THRESHOLD_SELECTION"
            --fake-methods "${FAKE_METHODS[@]}"
            --threshold-fake-methods "${FAKE_METHODS[@]}"
        )
        if [[ "$FLIP_TTA" == 1 ]]; then
            args+=(--texture-flip-tta)
        fi
        "$PYTHON" "${args[@]}"
    fi
    if [[ "$BOOTSTRAP" == 1 ]]; then
        predictions="$output_dir/predictions.csv"
        ci_output="$output_dir/bootstrap_ci.json"
        if [[ ! -f "$predictions" ]]; then
            echo "ERROR: predictions missing for ${profile}/seed${seed}: $predictions" >&2
            exit 1
        fi
        if [[ "$FORCE_EVAL" == 1 || ! -f "$ci_output" ]]; then
            "$PYTHON" scripts/bootstrap_ci.py \
                --predictions "$predictions" \
                --output "$ci_output" \
                --repetitions "$BOOTSTRAP_REPS" \
                --seed 0
        fi
    fi
done

"$PYTHON" scripts/summarize_in_domain_ablation.py \
    "${SUMMARY_ARGS[@]}" \
    --output-stem "$OUTPUT_ROOT/summary_${THRESHOLD_SELECTION}"
echo "FF++ in-domain ablation evaluation complete."
