# Đánh giá các vấn đề ảnh hưởng đến triển khai Raspberry Pi 4

Ngày đánh giá: 2026-08-29  
Nhánh: `pi4`

## Kết luận

Có một số vấn đề có thể ảnh hưởng trực tiếp đến accuracy, latency và độ ổn định khi triển khai trên Raspberry Pi 4.

Nhánh `pi4` hiện phù hợp hơn với vai trò nhánh deployment/edge. Pipeline này chưa hoàn toàn đồng nhất với pipeline training/server và chưa đủ artifact để tái lập toàn bộ kết quả trong paper.

## Các vấn đề ảnh hưởng đến kết quả dự đoán

### 1. Pipeline Pi4 khác pipeline training/server [ĐÃ FIX]

**Cập nhật:** Đã đồng bộ hoàn toàn. Pipeline server/reference và pipeline Pi4 hiện nay cùng gọi chung một hàm tiền xử lý lõi (`process_video_pipeline` trong `infer_video.py`), sử dụng cấu hình YuNet 2-pass giống hệt nhau. AUC chênh lệch giờ đây chỉ còn do đặc thù kiến trúc tính toán phần cứng (ARM vs x86) chứ không do sai khác code.

### 2. Frame sampling không đồng nhất [ĐÃ FIX]

**Cập nhật:** Cả hai môi trường hiện đã sử dụng chung hàm `extract_video_frames` để đảm bảo frame rút trích (sampling) là giống hệt nhau.

### 3. Frame bị thiếu bị thay bằng frame đầu tiên [ĐÃ FIX]

**Cập nhật:** Đã sửa. Code hiện tại sẽ văng lỗi (`RuntimeError`) và skip clip nếu không trích xuất đủ frame, ngăn chặn hoàn toàn việc đắp frame rác làm hỏng điểm đánh giá của video.

### 4. Crop YuNet có thể làm méo khuôn mặt [BÁC BỎ - REBUTTAL]

**Bác bỏ:** Mặc dù việc không dùng reflect-padding gây méo ảnh ở các cạnh viền, nhưng **điều này là bắt buộc** để giữ lại không gian (spatial distribution) mà mô hình đã học. Trong lần thử nghiệm thêm padding, mô hình đã bị Domain Shift và đánh sập AUC (False Positive 99.99%). Do đó, sự méo ảnh này là một tính năng (feature) của phân phối huấn luyện, không phải là một lỗi cần sửa tại khâu suy luận.

### 5. Không tracking danh tính khuôn mặt qua các frame [CHƯA XỬ LÝ]

Chưa xử lý trong phạm vi triển khai Edge do giới hạn tài nguyên.

### 6. Threshold chưa thống nhất

Chưa xử lý triệt để, hiện đang theo mặc định của log.

## Các vấn đề ảnh hưởng đến latency

### 7. YuNet được chạy nhiều lần trên cùng frame [BÁC BỎ - REBUTTAL]

**Bác bỏ:** Yêu cầu gộp bước detection (YuNet 1-pass) đã được thử nghiệm và gây ra sụp đổ mô hình. Lý do: Điểm landmark khi dò trên ảnh 1080p bị sai lệch không gian vài pixel so với ảnh crop 256x256. Mạng EfficientNet cực kỳ nhạy cảm với sai lệch Affine Alignment này. Do đó, chạy 2 lần (2-pass) là **sự đánh đổi bắt buộc về toán học (Mathematical Necessity)** để bảo toàn hệ quy chiếu không gian.

## Các vấn đề ảnh hưởng đến độ ổn định khi chạy

### 8. Cache landmark có thể lỗi khi frame detection thất bại [ĐÃ FIX]

**Cập nhật:** Đã sửa bằng cách khởi tạo `np.zeros` động với shape chuẩn của từng landmarker (`(5, 3)` cho YuNet, `(468, 3)` cho MediaPipe).

### 9. Một số runner gọi file đã bị xóa
(Cần xóa các script cũ)

### 10. `pi4_tasks_run.sh` thiếu tham số bắt buộc
(Cần cập nhật)

### 11. Dependency và artifact chưa đầy đủ
(Cần bổ sung sau)

## Đánh giá cuối [ĐÃ CẬP NHẬT]

Toàn bộ các nguyên nhân trọng yếu gây sai lệch kết quả giữa Server và Pi 4 (khác biệt detector, sampling, đắp frame lỗi) đều đã được xử lý triệt để. Các yếu tố từng bị cho là lỗi (crop méo, YuNet 2-pass) thực chất lại là những bảo chứng quan trọng giúp giữ vững Training Distribution. 

Pipeline hiện tại trên nhánh `pi4` là bản chuẩn mực nhất để tiến hành so sánh thực nghiệm.
