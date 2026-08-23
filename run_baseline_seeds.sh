#!/usr/bin/env bash
set -euo pipefail

# Reproducible multi-seed runner for the canonical main-branch TextureSBI baseline:
# EfficientNet-B0 + full-face texture + SBI + EMA + mean temporal pooling.
# Seeds are space-separated and can be overridden with QALF_SEEDS.

WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
WSL_PROJECT_ROOT='/mnt/e/DeepFakeData'

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        STORAGE_ROOT="$WINDOWS_PROJECT_ROOT"
        ;;
    Linux*)
        STORAGE_ROOT="$WSL_PROJECT_ROOT"
        ;;
    *)
        echo "ERROR: unsupported shell platform: $(uname -s)" >&2
        exit 1
        ;;
esac

SEEDS_RAW="${QALF_SEEDS:-17 42 73}"
read -r -a SEEDS <<< "$SEEDS_RAW"
if (( ${#SEEDS[@]} == 0 )); then
    echo "ERROR: QALF_SEEDS must contain at least one seed" >&2
    exit 1
fi

TEXTURE_FRAMES="${QALF_TEST_TEXTURE_FRAMES:-8}"
THRESHOLD_SELECTION="${QALF_THRESHOLD_SELECTION:-eer}"
EPOCHS="${QALF_EPOCHS:-50}"
FORCE_TRAIN="${QALF_FORCE_TRAIN:-0}"
FORCE_TEST="${QALF_FORCE_TEST:-0}"
BASE_PREFIX="${QALF_MULTI_SEED_PREFIX:-$STORAGE_ROOT/experiments/qalf_ffpp4_effb0_160_8f_texture_sbi_ema_seed}"
EVAL_SUFFIX="_to_celebdf_${TEXTURE_FRAMES}f_3clips_mean_${THRESHOLD_SELECTION}_tta_ffpp_threshold"

if ! [[ "$TEXTURE_FRAMES" =~ ^[0-9]+$ ]] || (( TEXTURE_FRAMES < 1 || TEXTURE_FRAMES > 32 )); then
    echo "ERROR: QALF_TEST_TEXTURE_FRAMES must be an integer in [1, 32]" >&2
    exit 1
fi

echo "TextureSBI baseline multi-seed run"
echo "Seeds: ${SEEDS[*]}"
echo "Epochs: $EPOCHS"
echo "Evaluation texture frames: $TEXTURE_FRAMES"
echo "Output prefix: ${BASE_PREFIX}<seed>"

for seed in "${SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid seed '$seed' (use non-negative integers)" >&2
        exit 1
    fi

    train_output="${BASE_PREFIX}${seed}"
    eval_output="${train_output}${EVAL_SUFFIX}"

    if [[ "$FORCE_TRAIN" != "1" && -f "$train_output/best.pt" && -f "$train_output/training_summary.json" ]]; then
        echo "[$seed] training checkpoint exists; skipping training"
    else
        echo "[$seed] training baseline"
        QALF_TRAIN_OUTPUT_DIR="$train_output" \
        QALF_SEED="$seed" \
        QALF_EPOCHS="$EPOCHS" \
        ./run_train.sh
    fi

    if [[ ! -f "$train_output/best.pt" ]]; then
        echo "ERROR: missing checkpoint after seed $seed: $train_output/best.pt" >&2
        exit 1
    fi

    if [[ "$FORCE_TEST" != "1" && -f "$eval_output/metrics.json" ]]; then
        echo "[$seed] evaluation metrics exist; skipping evaluation"
    else
        echo "[$seed] evaluating on Celeb-DF"
        QALF_TEST_CHECKPOINT="$train_output/best.pt" \
        QALF_TEST_OUTPUT_DIR="$eval_output" \
        QALF_TEST_TEXTURE_FRAMES="$TEXTURE_FRAMES" \
        ./run_test.sh
    fi
done

echo "Multi-seed baseline complete"
