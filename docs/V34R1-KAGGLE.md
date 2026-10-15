# Chạy v34r1 trên Kaggle theo yêu cầu người dùng

Người dùng yêu cầu dừng benchmark trên laptop và cung cấp tài khoản Kaggle để chạy GPU. Các tiến trình TEST cục bộ đã dừng; không dùng kết quả đang dang dở làm kết quả hoàn tất và không ghép chúng vào lượt Kaggle.

## Giao thức giữ nguyên

Không đổi A2 policy, controller303401, task weights,21 profile, shortlist/budget/guard, ID giữ lại, nhãn, kích thước128/320/960 hay khung/chỉ số nguồn. Chỉ dùng QP30/35/40/45. Không dùng điểm TEST cục bộ để điều chỉnh chính sách.

Các thay đổi thực thi được tách rõ: Linux path, FFmpeg build và thư viện nền Kaggle. Mỗi anchor và phương pháp mới trong lượt Kaggle dùng cùng môi trường. Không so sánh chéo byte từ build Windows với build Linux.

## Kiểm tra GPU đã hoàn tất

- Python3.13.15; torch2.11.0+cu128; torchvision0.26.0+cu128.
- Hai TeslaT4, mỗi GPU khoảng15GiB. Phép kiểm tra CUDA nhỏ trả tổng3.0 đúng.
- imageio-ffmpeg0.6.0 cung cấp FFmpeg7.0.2-static trên Linux.
- SHA-256 binary: `e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99`.
- Docker image: `gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`.

Đây là kiểm tra tương thích, không phải kết quả AR/OD/OT.

## Ba job chính

Mỗi job riêng tư ghim immutable Git SHA và Docker image, bật GPU+Internet, không chứa khóa API trong notebook hoặc Git. Repo và dữ liệu trung gian ở `/kaggle/temp`; chỉ log, protocol, paired rows, summary và archive kết quả ở `/kaggle/working`.

1. Cài các dependency bổ sung có phiên bản cụ thể; kiểm tra torch/torchvision và hash FFmpeg.
2. Chạy bộ tests các khối trên Kaggle, xác minh artifact v34 lịch sử.
3. Khôi phục đúng nguồn TEST từ mount hoặc endpoint công khai; kiểm tra từng SHA-256. Không thay source bị thiếu/hỏng bằng source khác.
4. Chạy `adaptive_vcm.v34r1.runner` bằng `configs/v34r1-kaggle.json`.
5. Chỉ đánh dấu complete sau khi đủ lưới và ghi summary. Luôn giữ log và trạng thái lỗi để chẩn đoán.

Manifest portable trong [v34r1-cloud-data-manifest.json.gz](../configs/v34r1-cloud-data-manifest.json.gz) chỉ chứa metadata/hash, không chứa media hay thông tin đăng nhập. Một số video giữ hash cả JPEG xen kẽ đã tải trước đó; model chỉ đọc đúng16 JPEG ở `provider_selected_paths`.

Tổng hợp cuối phải có đủ AR128 video, OD64 ảnh, OT256 frame và cả hai codec. Job được đẩy thành công hoặc báo RUNNING không đồng nghĩa đã đạt BD-rate hay đã hoàn thành.
