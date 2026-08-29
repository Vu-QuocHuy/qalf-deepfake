# Nhật Ký Thực Nghiệm & Gỡ Lỗi Pipeline Edge (Pi 4 vs Server)

Dưới đây là thống kê chi tiết các lần chạy thử nghiệm nhằm đồng bộ hóa hiệu năng và độ chính xác giữa môi trường huấn luyện (Server) và môi trường triển khai thực tế (Raspberry Pi 4).

## 1. Lần chạy 1 (Base - Code Pi4 Gốc)
- **Cấu hình:** Sử dụng YuNet 2-pass (1 lần lấy Bbox để crop, 1 lần lấy 5 điểm Landmark trên ảnh crop).
- **Kết quả:**
  - Server: AUC = `81.61%`
  - Pi 4: Chậm (~300ms/frame), AUC thấp hơn một chút.
- **Lỗi phát hiện:** 
  1. Bug hụt frame: Khi video không đọc được frame tiếp theo, code tự động đắp frame đầu tiên vào thay thế, làm nhiễu dữ liệu.
  2. Bug méo ảnh (Crop Distortion): Hàm `crop_face_to_256` không có padding. Khi mặt sát viền, ảnh bị cắt thành hình chữ nhật rồi bóp méo thành hình vuông 256x256.

---

## 2. Lần chạy 2 (Thử nghiệm Tối ưu 1-pass YuNet)
- **Mục tiêu:** Trả lời phàn nàn của Reviewer ("YuNet chạy nhiều lần trên cùng 1 frame làm chậm Pi 4").
- **Hành động:** 
  - Gộp YuNet thành 1-pass duy nhất trên ảnh 1080p.
  - Sửa lỗi đắp frame rác (throw exception thay vì đắp frame).
  - Sửa lỗi méo ảnh bằng cách thêm viền (padding) `BORDER_REFLECT`.
- **Kết quả:** **THẤT BẠI THẢM HẠI (Catastrophic Failure)**
  - Tỷ lệ False Positive cực cao (Model báo Fake 99.99% cho các video Real).
- **Phân tích nguyên nhân:** Tối ưu 1-pass đã bỏ qua bước ép ảnh về 256x256 bằng hàm `INTER_AREA`, dẫn đến hiện tượng Răng cưa (Aliasing). Mô hình EfficientNet nhầm lẫn nhiễu răng cưa này thành dấu vết Deepfake.

---

## 3. Lần chạy 3 (Khôi phục Anti-Aliasing nhưng giữ 1-pass)
- **Hành động:** Chèn lại bước crop ảnh trung gian với `INTER_AREA`, dùng toán học ma trận để dời 5 điểm mốc (landmarks) từ toạ độ 1080p sang khớp với toạ độ trên ảnh `256x256`. Giữ nguyên việc chạy YuNet 1 lần.
- **Kết quả:** **VẪN THẤT BẠI (99.99% Fake cho video Real trên Server)**
- **Khám phá Khoa học (Nguyên nhân gốc rễ):**
  - Việc sửa lỗi "Méo ảnh" và thay đổi cách YuNet dự đoán điểm mốc đã gây ra **Domain Shift (Sai lệch phân phối dữ liệu)**.
  - Khi YuNet chạy trên ảnh lớn 1080p, toạ độ 5 điểm mốc bị lệch vài pixel so với khi chạy trên ảnh crop nhỏ 256x256. 
  - Mô hình được huấn luyện trên các bức ảnh bị méo (nếu mặt sát viền) và với toạ độ YuNet sinh ra từ ảnh crop. Khi chúng ta "sửa lỗi", khuôn mặt bị xoay và phóng to khác đi so với trong tập Training, đẩy dữ liệu văng ra ngoài không gian hiểu biết của mạng Neural.

---

## 4. Lần chạy 4 (Hoàn nguyên - Chốt hạ)
- **Hành động:**
  - Hoàn nguyên (Revert) hàm `crop_face_to_256` về đúng 100% nguyên trạng ban đầu (chấp nhận ảnh bị méo nếu mặt sát viền).
  - Chấp nhận giữ nguyên cơ chế **2-pass YuNet** (Toán học bắt buộc để giữ chuẩn toạ độ landmark).
  - Chỉ giữ lại bản vá lỗi hụt frame (tránh đắp frame rác).
- **Kết quả:**
  - Server: Trở về AUC ổn định chuẩn mực **81.61%** (Khớp tuyệt đối với mức baseline của kiến trúc YuNet trên GPU).
  - Pi 4: Đang chạy trơn tru, dự kiến sẽ cho kết quả gần sát hoặc tương đương với Server, tốc độ tối ưu nhất có thể cho một pipeline bảo toàn phân phối huấn luyện.
- **Kết luận:** Quy trình 2-pass YuNet và cách crop gốc là **Mathematical Necessity (Sự cần thiết về mặt toán học)**. Mọi nỗ lực "tối ưu hoá" thay đổi không gian ảnh đều sẽ phá huỷ ma trận Affine mà mạng EfficientNet đã học thuộc lòng.
