#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# BASH SCRIPT: TOÀN BỘ QUY TRÌNH KIỂM THỬ VÀ ĐO HIỆU NĂNG TRÊN RASPBERRY PI 4
# ==============================================================================

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "/mnt/usb_data/venv_qalf/bin/activate" ]]; then
    source /mnt/usb_data/venv_qalf/bin/activate
fi

USB_ROOT="/mnt/usb_data"
TEST_DATA_DIR="$USB_ROOT/celebdf_test_518"
OUTPUT_SYNTHESIS_JSON="$USB_ROOT/profile_synthesis_mtcnn.json"
OUTPUT_REAL_JSON="$USB_ROOT/profile_youtube_real_mtcnn.json"

echo "=============================================================================="
echo "    BẮT ĐẦU CHẠY TOÀN BỘ QUY TRÌNH THỰC NGHIỆM TRÊN RASPBERRY PI 4"
echo "=============================================================================="
echo "Mô hình ONNX      : models/qalf.onnx"
echo "Tập Dữ Liệu       : $TEST_DATA_DIR"
echo "Cấu hình          : 3 clips x 8 frames across 32f @ 10 FPS, MTCNN, Flip-TTA"
echo "CPU Threads       : 4"
echo "=============================================================================="

export PYTHONUNBUFFERED=1

# 1. Quét toàn bộ 340 video Fake (Celeb-synthesis)
echo ""
echo ">>> [GIAI ĐOẠN 1/2] Đang quét toàn bộ 340 Video Fake (Celeb-synthesis)..."
python -u scripts/infer_video.py \
    --video-dir "$TEST_DATA_DIR/Celeb-synthesis" \
    --onnx "models/qalf.onnx" \
    --clips 3 \
    --cpu-threads 4 \
    --output-json "$OUTPUT_SYNTHESIS_JSON"

# 2. Quét toàn bộ video Real (YouTube-real)
echo ""
echo ">>> [GIAI ĐOẠN 2/2] Đang quét toàn bộ Video Real (YouTube-real)..."
python -u scripts/infer_video.py \
    --video-dir "$TEST_DATA_DIR/YouTube-real" \
    --onnx "models/qalf.onnx" \
    --clips 3 \
    --cpu-threads 4 \
    --output-json "$OUTPUT_REAL_JSON"

# 3. Tính toán tổng hợp chỉ số Khoa học (AUC, EER, Accuracy)
echo ""
echo "=============================================================================="
echo "    TỔNG HỢP KẾT QUẢ KHOA HỌC CUỐI CÙNG (FINAL SCIENTIFIC METRICS)"
echo "=============================================================================="

python3 - << 'EOF'
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

with open("/mnt/usb_data/profile_synthesis_mtcnn.json") as f:
    fakes = json.load(f)
with open("/mnt/usb_data/profile_youtube_real_mtcnn.json") as f:
    reals = json.load(f)

y_true = [1] * len(fakes) + [0] * len(reals)
y_scores = [r["detection"]["fake_probability"] for r in fakes] + [r["detection"]["fake_probability"] for r in reals]
threshold = fakes[0]["detection"]["threshold"]
y_preds = [1 if s >= threshold else 0 for s in y_scores]

auc = roc_auc_score(y_true, y_scores)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
fnr = 1 - tpr
idx_eer = np.nanargmin(np.absolute((fnr - fpr)))
eer = fpr[idx_eer]
acc = accuracy_score(y_true, y_preds)
cm = confusion_matrix(y_true, y_preds)

print(f"Tổng số video kiểm thử       : {len(y_true)} ({len(fakes)} Fake + {len(reals)} Real)")
print(f"Ngưỡng quyết định (Youden-J) : {threshold:.4f}")
print("-" * 65)
print(f"AUC-ROC Score                : {auc * 100:.2f} %")
print(f"Equal Error Rate (EER)       : {eer * 100:.2f} %")
print(f"Overall Accuracy             : {acc * 100:.2f} % ({sum(y_t == y_p for y_t, y_p in zip(y_true, y_preds))}/{len(y_true)})")
print(f"Recall Fake (Sensitivity)    : {cm[1, 1] / len(fakes) * 100:.2f} % ({cm[1, 1]}/{len(fakes)})")
print(f"Recall Real (Specificity)    : {cm[0, 0] / len(reals) * 100:.2f} % ({cm[0, 0]}/{len(reals)})")
print("=" * 65)
EOF

echo ""
echo "[HOÀN TẤT 100%] Toàn bộ dữ liệu thực nghiệm đã được lưu trữ an toàn."
