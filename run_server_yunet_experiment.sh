#!/usr/bin/env bash
set -euo pipefail

# Kịch bản chạy thí nghiệm đối chứng YuNet trên Server (PyTorch) cho video thô
# Thực hiện cả hai điều kiện: C1 (CPU) và C2 (GPU)

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Thiết lập đường dẫn cơ bản (có thể ghi đè qua biến môi trường)
WINDOWS_PROJECT_ROOT='E:/DeepFakeData'
WSL_PROJECT_ROOT='/mnt/e/DeepFakeData'

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        STORAGE_ROOT="${QALF_STORAGE_ROOT:-$WINDOWS_PROJECT_ROOT}"
        PYTHON=".venv/Scripts/python.exe"
        ;;
    Linux*)
        STORAGE_ROOT="${QALF_STORAGE_ROOT:-$WSL_PROJECT_ROOT}"
        PYTHON=".venv/bin/python"
        ;;
    *)
        echo "ERROR: unsupported shell platform: $(uname -s)" >&2
        exit 1
        ;;
esac

# Đường dẫn đến dữ liệu Celeb-DF-v2
MANIFEST="${QALF_TEST_MANIFEST:-F:/DeepFakedata/Celeb_DFv2/List_of_testing_videos.txt}"
VIDEO_ROOT="${QALF_TEST_VIDEO_ROOT:-F:/DeepFakedata/Celeb_DFv2}"

# Tham số mô hình
SEED="${QALF_SEED:-42}"
CHECKPOINT="${QALF_TEST_CHECKPOINT:-$STORAGE_ROOT/experiments/ablation/baseline_seed${SEED}/best.pt}"

# Thư mục đầu ra
OUTPUT_DIR_GPU="${QALF_TEST_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_server_yunet_seed${SEED}_gpu}"
OUTPUT_DIR_CPU="${QALF_TEST_OUTPUT_DIR:-$STORAGE_ROOT/experiments/qalf_server_yunet_seed${SEED}_cpu}"

echo "======================================================"
echo " CHẠY THÍ NGHIỆM ĐỐI CHỨNG: SERVER + YUNET + PYTORCH"
echo "======================================================"
echo "Manifest: $MANIFEST"
echo "Video Root: $VIDEO_ROOT"
echo "Checkpoint: $CHECKPOINT"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found at $MANIFEST"
    exit 1
fi

# Chạy C2: PyTorch GPU
echo ""
echo ">>> Bắt đầu chạy C2: PyTorch GPU"
if [[ -f "$OUTPUT_DIR_GPU/metrics.json" ]]; then
    echo "Đã có kết quả GPU tại $OUTPUT_DIR_GPU, bỏ qua..."
else
    "$PYTHON" scripts/evaluate_server_yunet_e2e.py \
        --manifest "$MANIFEST" \
        --video-root "$VIDEO_ROOT" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$OUTPUT_DIR_GPU" \
        --device "cuda" \
        --clips-per-video 3 \
        --texture-frames 8
fi

# Chạy C1: PyTorch CPU
echo ""
echo ">>> Bắt đầu chạy C1: PyTorch CPU"
if [[ -f "$OUTPUT_DIR_CPU/metrics.json" ]]; then
    echo "Đã có kết quả CPU tại $OUTPUT_DIR_CPU, bỏ qua..."
else
    "$PYTHON" scripts/evaluate_server_yunet_e2e.py \
        --manifest "$MANIFEST" \
        --video-root "$VIDEO_ROOT" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$OUTPUT_DIR_CPU" \
        --device "cpu" \
        --clips-per-video 3 \
        --texture-frames 8
fi

echo "======================================================"
echo "HOÀN THÀNH!"
echo "Vui lòng kiểm tra metrics.json trong các thư mục đầu ra để lấy AUC."
