#!/usr/bin/env bash
set -euo pipefail

# Resumable training/evaluation suite for the locked main-branch baseline.
# Core comparisons use five seeds; lightweight controls use one seed. Every
# profile has an independent output directory, so an interrupted run can be
# restarted without overwriting completed checkpoints or metrics.

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

MODE="${QALF_ABLATION_MODE:-all}"
case "$MODE" in
    all) DO_TRAIN=1; DO_EVAL=1; DO_ROBUSTNESS=1 ;;
    train) DO_TRAIN=1; DO_EVAL=0; DO_ROBUSTNESS=0 ;;
    eval) DO_TRAIN=0; DO_EVAL=1; DO_ROBUSTNESS=0 ;;
    robustness) DO_TRAIN=0; DO_EVAL=0; DO_ROBUSTNESS=1 ;;
    *) echo "ERROR: QALF_ABLATION_MODE must be all, train, eval, or robustness" >&2; exit 1 ;;
esac

DATA_ROOT="$STORAGE_ROOT/data"
CONFIG="$PROJECT_ROOT/configs/ffpp_to_celebdf.json"
ABLATION_ROOT="${QALF_ABLATION_ROOT:-$STORAGE_ROOT/experiments/ablation}"
BASELINE_PREFIX="${QALF_ABLATION_BASELINE_PREFIX:-$ABLATION_ROOT/baseline_seed}"
CORE_SEEDS_RAW="${QALF_ABLATION_CORE_SEEDS:-0 17 42 73 123}"
CONTROL_SEED="${QALF_ABLATION_CONTROL_SEED:-42}"
EPOCHS="${QALF_EPOCHS:-50}"
FORCE_TRAIN="${QALF_ABLATION_FORCE_TRAIN:-0}"
FORCE_EVAL="${QALF_ABLATION_FORCE_EVAL:-0}"
BOOTSTRAP_REPS="${QALF_BOOTSTRAP_REPS:-2000}"
TEXTURE_FRAMES=8
THRESHOLD_SELECTION="${QALF_THRESHOLD_SELECTION:-eer}"
EVAL_SUFFIX="_to_celebdf_8f_3clips_mean_${THRESHOLD_SELECTION}_tta_ffpp_threshold"
ALL_PROFILES=(baseline no_sbi no_ema texture_only no_pretrain no_aug sbi_half)
PROFILES_RAW="${QALF_ABLATION_PROFILES:-${ALL_PROFILES[*]}}"
PROFILE_FILTERED=0
if [[ -n "${QALF_ABLATION_PROFILES:-}" ]]; then
    PROFILE_FILTERED=1
fi

read -r -a CORE_SEEDS <<< "$CORE_SEEDS_RAW"
if (( ${#CORE_SEEDS[@]} == 0 )); then
    echo "ERROR: QALF_ABLATION_CORE_SEEDS must contain at least one seed" >&2
    exit 1
fi
read -r -a SELECTED_PROFILES <<< "$PROFILES_RAW"
if (( ${#SELECTED_PROFILES[@]} == 0 )); then
    echo "ERROR: QALF_ABLATION_PROFILES must contain at least one profile" >&2
    exit 1
fi
profile_enabled() {
    local requested="$1" selected
    for selected in "${SELECTED_PROFILES[@]}"; do
        if [[ "$selected" == "$requested" ]]; then
            return 0
        fi
    done
    return 1
}
for selected in "${SELECTED_PROFILES[@]}"; do
    known=0
    for profile in "${ALL_PROFILES[@]}"; do
        if [[ "$selected" == "$profile" ]]; then
            known=1
            break
        fi
    done
    if (( ! known )); then
        echo "ERROR: unknown ablation profile '$selected'; expected one of: ${ALL_PROFILES[*]}" >&2
        exit 1
    fi
done
mkdir -p "$ABLATION_ROOT"
export CUBLAS_WORKSPACE_CONFIG=':4096:8'

train_profile() {
    local profile="$1"
    local seed="$2"
    local prefix="$3"
    shift 3
    local output_dir="${prefix}${seed}"

    if [[ "$FORCE_TRAIN" != "1" && -f "$output_dir/best.pt" && -f "$output_dir/training_summary.json" ]]; then
        echo "[$profile/$seed] training checkpoint exists; skipping"
        return
    fi
    echo "[$profile/$seed] training"
    "$PYTHON" scripts/train.py \
        --config "$CONFIG" \
        --train-manifest "$DATA_ROOT/landmarks/ffpp-landmark/manifests/ffpp_train_landmarks.jsonl" \
        --val-manifest "$DATA_ROOT/landmarks/ffpp-landmark/manifests/ffpp_val_landmarks.jsonl" \
        --frame-root "$DATA_ROOT/extracted/ffpp" \
        --landmark-root "$DATA_ROOT/landmarks/ffpp-landmark/landmarks" \
        --output-dir "$output_dir" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch-size 8 \
        --num-workers 4 \
        --learning-rate 0.0003 \
        --backbone-learning-rate 0.00003 \
        --weight-decay 0.0003 \
        --early-stop-patience 5 \
        --num-frames 32 \
        --texture-frames 8 \
        --image-size 160 \
        --eval-clips-per-video 3 \
        --fake-methods Deepfakes Face2Face FaceSwap NeuralTextures \
        --texture-backbone efficientnet_b0 \
        --texture-mode full_face \
        --embedding-dim 192 \
        --dropout 0.3 \
        --deterministic \
        "$@"
}

evaluate_profile() {
    local profile="$1"
    local seed="$2"
    local prefix="$3"
    local train_dir="${prefix}${seed}"
    local checkpoint="$train_dir/best.pt"
    local output_dir="${train_dir}${EVAL_SUFFIX}"
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: missing checkpoint for $profile/$seed: $checkpoint" >&2
        exit 1
    fi
    if [[ "$FORCE_EVAL" != "1" && -f "$output_dir/metrics.json" ]]; then
        echo "[$profile/$seed] evaluation metrics exist; skipping"
        return
    fi
    echo "[$profile/$seed] evaluating on Celeb-DF"
    QALF_TEST_CHECKPOINT="$checkpoint" \
    QALF_TEST_OUTPUT_DIR="$output_dir" \
    QALF_TEST_TEXTURE_FRAMES=8 \
    QALF_TEST_CLIPS_PER_VIDEO=3 \
    QALF_TEST_AGGREGATION=mean \
    QALF_TEST_TOP_K=1 \
    QALF_TEST_THRESHOLD_CLIPS_PER_VIDEO=3 \
    QALF_TEST_FLIP_TTA=1 \
        ./run_test.sh
}

bootstrap_profile() {
    local profile="$1"
    local seed="$2"
    local prefix="$3"
    local eval_dir="${prefix}${seed}${EVAL_SUFFIX}"
    local predictions="$eval_dir/predictions.csv"
    local output="$eval_dir/bootstrap_ci.json"
    if [[ ! -f "$predictions" ]]; then
        echo "ERROR: missing aggregated predictions for $profile/$seed: $predictions" >&2
        exit 1
    fi
    if [[ "$FORCE_EVAL" != "1" && -f "$output" ]]; then
        echo "[$profile/$seed] bootstrap CI exists; skipping"
        return
    fi
    echo "[$profile/$seed] bootstrap CI (${BOOTSTRAP_REPS} video resamples)"
    "$PYTHON" scripts/bootstrap_ci.py \
        --predictions "$predictions" \
        --output "$output" \
        --repetitions "$BOOTSTRAP_REPS" \
        --seed 0
}

summarize_profile() {
    local profile="$1"
    local prefix="$2"
    shift 2
    local summary_stem="$ABLATION_ROOT/${profile}_summary"
    "$PYTHON" scripts/summarize_seed_runs.py \
        --train-prefix "$prefix" \
        --eval-suffix "$EVAL_SUFFIX" \
        --seeds "$@" \
        --output-stem "$summary_stem"
}

run_profile_seed() {
    local profile="$1"
    local seed="$2"
    local prefix="$3"
    shift 3
    if (( DO_TRAIN )); then
        train_profile "$profile" "$seed" "$prefix" "$@"
    fi
    if (( DO_EVAL )); then
        evaluate_profile "$profile" "$seed" "$prefix"
        bootstrap_profile "$profile" "$seed" "$prefix"
    fi
}

echo "TextureSBI ablation suite"
echo "Mode: $MODE"
echo "Profiles: ${SELECTED_PROFILES[*]}"
echo "Core seeds: ${CORE_SEEDS[*]}"
echo "Control seed: $CONTROL_SEED"
echo "Epochs: $EPOCHS"
echo "Output root: $ABLATION_ROOT"

# Core five-seed comparisons forming the complete SBI x EMA grid.
if profile_enabled baseline; then
    for seed in "${CORE_SEEDS[@]}"; do
        run_profile_seed baseline "$seed" "$BASELINE_PREFIX" --sbi --ema-decay 0.999 --validation-weights ema
    done
fi
if profile_enabled no_sbi; then
    for seed in "${CORE_SEEDS[@]}"; do
        run_profile_seed no_sbi "$seed" "$ABLATION_ROOT/no_sbi_seed" --no-sbi --ema-decay 0.999 --validation-weights ema
    done
fi
if profile_enabled no_ema; then
    for seed in "${CORE_SEEDS[@]}"; do
        run_profile_seed no_ema "$seed" "$ABLATION_ROOT/no_ema_seed" --sbi --ema-decay 0 --validation-weights raw
    done
fi
if profile_enabled texture_only; then
    for seed in "${CORE_SEEDS[@]}"; do
        run_profile_seed texture_only "$seed" "$ABLATION_ROOT/texture_only_seed" --no-sbi --ema-decay 0 --validation-weights raw
    done
fi

# One-seed controls: useful for the paper's implementation table without
# multiplying the expensive core grid.
if profile_enabled no_pretrain; then
    run_profile_seed no_pretrain "$CONTROL_SEED" "$ABLATION_ROOT/no_pretrain_seed" \
        --sbi --ema-decay 0.999 --validation-weights ema --no-texture-pretrained
fi
if profile_enabled no_aug; then
    run_profile_seed no_aug "$CONTROL_SEED" "$ABLATION_ROOT/no_aug_seed" \
        --sbi --ema-decay 0.999 --validation-weights ema --no-texture-augmentation
fi
if profile_enabled sbi_half; then
    run_profile_seed sbi_half "$CONTROL_SEED" "$ABLATION_ROOT/sbi_half_seed" \
        --sbi --sbi-mixture 0.25 0.25 0.50 --ema-decay 0.999 --validation-weights ema
fi

if (( DO_EVAL )); then
    profile_enabled baseline && summarize_profile baseline "$BASELINE_PREFIX" "${CORE_SEEDS[@]}"
    profile_enabled no_sbi && summarize_profile no_sbi "$ABLATION_ROOT/no_sbi_seed" "${CORE_SEEDS[@]}"
    profile_enabled no_ema && summarize_profile no_ema "$ABLATION_ROOT/no_ema_seed" "${CORE_SEEDS[@]}"
    profile_enabled texture_only && summarize_profile texture_only "$ABLATION_ROOT/texture_only_seed" "${CORE_SEEDS[@]}"
    profile_enabled no_pretrain && summarize_profile no_pretrain "$ABLATION_ROOT/no_pretrain_seed" "$CONTROL_SEED"
    profile_enabled no_aug && summarize_profile no_aug "$ABLATION_ROOT/no_aug_seed" "$CONTROL_SEED"
    profile_enabled sbi_half && summarize_profile sbi_half "$ABLATION_ROOT/sbi_half_seed" "$CONTROL_SEED"
fi

if (( DO_EVAL )) && (( ! PROFILE_FILTERED )); then
    # Evaluation-only protocol ablations use the seed-42 baseline checkpoint.
    BASELINE_SEED42="${BASELINE_PREFIX}42/best.pt"
    if [[ ! -f "$BASELINE_SEED42" ]]; then
        echo "ERROR: seed-42 baseline checkpoint required for protocol ablations: $BASELINE_SEED42" >&2
        exit 1
    fi
    run_eval_case() {
        local name="$1" frames="$2" clips="$3" aggregation="$4" flip="$5"
        local output_dir="$ABLATION_ROOT/eval_${name}"
        if [[ "$FORCE_EVAL" != "1" && -f "$output_dir/metrics.json" ]]; then
            echo "[eval/$name] metrics exist; skipping"
            return
        fi
        echo "[eval/$name] evaluating"
        QALF_TEST_CHECKPOINT="$BASELINE_SEED42" \
        QALF_TEST_OUTPUT_DIR="$output_dir" \
        QALF_TEST_TEXTURE_FRAMES="$frames" \
        QALF_TEST_CLIPS_PER_VIDEO="$clips" \
        QALF_TEST_AGGREGATION="$aggregation" \
        QALF_TEST_TOP_K=1 \
        QALF_TEST_THRESHOLD_CLIPS_PER_VIDEO=3 \
        QALF_TEST_FLIP_TTA="$flip" \
            ./run_test.sh
    }
    run_eval_case frames4 4 3 mean 1
    run_eval_case frames8 8 3 mean 1
    run_eval_case frames12 12 3 mean 1
    run_eval_case clips1 8 1 mean 1
    run_eval_case clips3 8 3 mean 1
    run_eval_case clips5 8 5 mean 1
    run_eval_case median 8 3 median 1
    run_eval_case no_tta 8 3 mean 0
fi

if (( DO_ROBUSTNESS )) && (( ! PROFILE_FILTERED )); then
    BASELINE_SEED42="${BASELINE_PREFIX}42/best.pt"
    if [[ ! -f "$BASELINE_SEED42" ]]; then
        echo "ERROR: seed-42 baseline checkpoint required for robustness: $BASELINE_SEED42" >&2
        exit 1
    fi
    robustness_output="$ABLATION_ROOT/baseline_robustness.json"
    if [[ "$FORCE_EVAL" == "1" || ! -f "$robustness_output" ]]; then
        QALF_ROBUSTNESS_CHECKPOINT="$BASELINE_SEED42" \
        QALF_ROBUSTNESS_OUTPUT="$robustness_output" \
        QALF_ROBUSTNESS_TEXTURE_FRAMES=8 \
            ./run_robustness.sh
    else
        echo "[robustness] report exists; skipping"
    fi
fi

echo "========================================================================"
echo "Ablation suite complete"
echo "Reports: $ABLATION_ROOT"
