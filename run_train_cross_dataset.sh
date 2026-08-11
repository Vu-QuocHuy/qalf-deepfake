#!/usr/bin/env bash
set -euo pipefail

# Focused cross-dataset profiles built from the best full-face configuration.
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

PROFILE="${1:-full_face_sbi}"
PROFILE_ARGS=()
case "$PROFILE" in
    full_face)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_deterministic'
        DESCRIPTION='established 0.8209 AUC full-face baseline'
        PROFILE_ARGS+=(--texture-mode full_face)
        ;;
    full_face_sbi)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_sbi'
        DESCRIPTION='locked full-face baseline with 50/25/25 SBI hybrid training'
        PROFILE_ARGS+=(--texture-mode full_face --sbi)
        ;;
    geometry_g1_balanced)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_g1_balanced'
        DESCRIPTION='SBI plus class-balanced supervised geometry loss'
        PROFILE_ARGS+=(--texture-mode full_face --sbi --geometry-class-balanced)
        ;;
    geometry_g2_attentive)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_g2_attentive'
        DESCRIPTION='G1 plus attentive temporal statistics pooling'
        PROFILE_ARGS+=(
            --texture-mode full_face
            --sbi
            --geometry-class-balanced
            --geometry-architecture tcn_attentive
        )
        ;;
    geometry_g3_graph)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_g3_graph'
        DESCRIPTION='G2 plus clip-local landmark graph message passing'
        PROFILE_ARGS+=(
            --texture-mode full_face
            --sbi
            --geometry-class-balanced
            --geometry-architecture graph_attentive
        )
        ;;
    geometry_g4_two_stream)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_g4_two_stream'
        DESCRIPTION='G3 plus rigid and non-rigid geometry streams'
        PROFILE_ARGS+=(
            --texture-mode full_face
            --sbi
            --geometry-class-balanced
            --geometry-mode aligned_motion_rigid_3d
            --geometry-architecture graph_rigid_attentive
        )
        ;;
    geometry_g5_self_supervised)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_g5_self_supervised'
        DESCRIPTION='G4 plus geometry augmentation-consistency self-supervision'
        PROFILE_ARGS+=(
            --texture-mode full_face
            --sbi
            --geometry-class-balanced
            --geometry-mode aligned_motion_rigid_3d
            --geometry-architecture graph_rigid_attentive
            --geometry-consistency-noise-std 0.10
            --geometry-self-supervision-loss-weight 0.05
        )
        ;;
    geometry_g6_reliability)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_sbi_geometry_g6_reliability'
        DESCRIPTION='G5 plus modality dropout and supervised reliability routing'
        PROFILE_ARGS+=(
            --texture-mode full_face
            --sbi
            --geometry-class-balanced
            --geometry-mode aligned_motion_rigid_3d
            --geometry-architecture graph_rigid_attentive
            --geometry-consistency-noise-std 0.10
            --geometry-self-supervision-loss-weight 0.05
            --modality-dropout-probability 0.15
            --reliability-gate-loss-weight 0.10
        )
        ;;
    full_face_ema)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_ema'
        DESCRIPTION='full-face baseline with EMA checkpoint selection'
        PROFILE_ARGS+=(--texture-mode full_face --ema-decay 0.999)
        ;;
    full_face_mixstyle)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_mixstyle'
        DESCRIPTION='full-face baseline with video-coherent MixStyle'
        PROFILE_ARGS+=(
            --texture-mode full_face
            --texture-mixstyle-probability 0.5
            --texture-mixstyle-alpha 0.1
            --texture-mixstyle-layers 1 2
        )
        ;;
    full_face_dynamics)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_full_face_dynamics'
        DESCRIPTION='full-face baseline with temporal dynamics pooling'
        PROFILE_ARGS+=(--texture-mode full_face --texture-temporal-pooling dynamics)
        ;;
    dual_view)
        EXPERIMENT='qalf_ffpp4_effb0_160_8f_dual_view'
        DESCRIPTION='shared-backbone fusion of full-face and canonical-skin views'
        PROFILE_ARGS+=(--texture-mode dual_view)
        ;;
    *)
        echo "ERROR: unknown profile '$PROFILE'" >&2
        echo 'Use: full_face, full_face_sbi, geometry_g1_balanced, geometry_g2_attentive, geometry_g3_graph, geometry_g4_two_stream, geometry_g5_self_supervised, geometry_g6_reliability, full_face_ema, full_face_mixstyle, full_face_dynamics, or dual_view' >&2
        exit 2
        ;;
esac

DATA_ROOT="$STORAGE_ROOT/data"
FRAME_ROOT="$DATA_ROOT/extracted/ffpp"
LANDMARK_OUTPUT_ROOT="$DATA_ROOT/landmarks/ffpp-landmark"
LANDMARK_ROOT="$LANDMARK_OUTPUT_ROOT/landmarks"
TRAIN_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_train_landmarks.jsonl"
VAL_MANIFEST="$LANDMARK_OUTPUT_ROOT/manifests/ffpp_val_landmarks.jsonl"
OUTPUT_DIR="${QALF_TRAIN_OUTPUT_DIR:-$STORAGE_ROOT/experiments/$EXPERIMENT}"
BATCH_SIZE="${QALF_BATCH_SIZE:-8}"
NUM_WORKERS="${QALF_NUM_WORKERS:-4}"

export CUBLAS_WORKSPACE_CONFIG=':4096:8'
echo "Python: $PYTHON"
echo "Profile: $PROFILE ($DESCRIPTION)"
echo "Training output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
"$PYTHON" -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"

"$PYTHON" scripts/train.py \
    --config configs/ffpp_to_celebdf.json \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --frame-root "$FRAME_ROOT" \
    --landmark-root "$LANDMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --epochs 35 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
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
    --geometry-hidden 128 \
    --geometry-layers 3 \
    --embedding-dim 192 \
    --dropout 0.3 \
    --geometry-mode aligned_motion_3d \
    --fusion-mode quality \
    --geometry-loss-weight 0.25 \
    --texture-loss-weight 0.25 \
    --texture-gate-bias 0.0 \
    --deterministic \
    "${PROFILE_ARGS[@]}"
