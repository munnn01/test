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

## Job và xử lý chờ nguồn

Mã đo bất biến: `5a369016e4214b8339f89768534fed19cbcf194e`. Các job riêng tư được gửi qua Kaggle API; không đưa token vào notebook. OD và OT đã ghi `75 passed` trên cùng Docker image và xác nhận hash các artifact lịch sử.

- AR ban đầu: `shungg05/v34r1-ar-test-qp4`, gắn dataset provider23GB. Sau nhiều lần trạng thái RUNNING nhưng chưa có log mã chạy, kiểm tra riêng trên Kaggle đã tải một JPEG qua endpoint version1, SHA-256 khớp trong0.775s.
- AR dự phòng: `baooo25r/v34r1-ar-direct-test-qp4`, notebook giống từng byte, cùng Docker/GPU/config/commit và manifest; không gắn toàn dataset. Script khôi phục có sẵn tải từng tệp đúng tên/hash. Không thay ID hoặc chọn nguồn theo độ chính xác. Đây là thay đổi vận chuyển dữ liệu, không phải một chính sách khác.
- OD: `dieulinhh/v34r1-od-test-qp4`.
- OT: `huolgggnuyen/v34r1-ot-test-qp4`.

Nếu cả hai job AR hoàn tất, chọn job hoàn tất trước theo thời gian, không chọn theo BD-rate. Phân biệt rõ lượt lặp khỏi số lượng video độc lập; không ghép hai lượt thành256 video.

OD hoàn tất đủ1024 hàng ghép cặp: H.264−7.8396124194%; H.265 không hợp lệ do quality không đơn điệu theo rate. Đây là kết quả chưa đạt mục tiêu, vẫn được giữ nguyên.

AR direct version1 dừng khi khôi phục JPEG vì HTTP404, trước suy luận TEST. Bộ tải cục bộ trước đó có retry mọi lỗi request; bộ tải cloud chỉ retry429. Bổ sung retry có giới hạn cho lỗi HTTP/timeout, ghi tên tệp khi hết lượt, vẫn bắt buộc đúng SHA-256 và không thay nguồn. Hai regression test CPU tái hiện HTTP404/timeout và lỗi kéo dài đã fail trước sửa, pass sau sửa. Chỉ script vận chuyển nguồn/tests/docs thay đổi; `adaptive_vcm` và toàn bộ `configs` giữ nguyên từng byte. Job AR tiếp theo kiểm tra lại toàn bộ77 tests trên Kaggle trước khi đo.

Các lỗi HTTP vẫn lặp ở direct version2. Đã chuyển3053 JPEG đóng băng có sẵn thành cache riêng tư `baooo25r/v34r1-frozen-ar-source-cache` version1; không đưa cache vào Git hay phát hành public. Archive43,795,301byte có SHA-256 `727e1b582943e77cdf509b0a27986af52bdb13a7d6dd046de2ad72fc03caac19`; metadata giữ license `unknown` như provider, không tự gán quyền phân phối lại.

Job `baooo25r/v34r1-ar-cache-test-qp4` kiểm tra hash3053 tệp ngay sau mount và chép đúng đường portable trước khi chạy script chuẩn. Nó ghi `77 passed`, xác minh128 record và manifest `08bce7123f7289e8a5722c11b7ea74b63e4cb54ce2be060e77a5c9998dab9352`, rồi mới bắt đầu TEST. Repo commit b3f30bcd1c111b69311180f9a5259f0541eee817 chỉ sửa transport/tests/docs so với5a36901; thuật toán/config còn nguyên. Cả ba job đều dùng cùng Docker/T4×2, FFmpeg, weights và các hash khoa học.
