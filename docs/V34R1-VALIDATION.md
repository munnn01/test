# Kiểm chứng các khối v34r1

Ngày2026-10-15, trên Windows/Python3.12 và RTX4060 Laptop8GB. Commit mã đã đóng băng trước TEST: `fd345382d2363b3bc926c1103c7fe8387e0fd1f4`.

## Bằng chứng đã chạy

`D:\TEST\.venv\Scripts\python.exe -m pytest D:\TEST\tests -q`: **75 passed**. Bộ tests gồm cả legacy, codec thật, TrackEval integration và các test mới. `scripts/reproduce_results.py --verify-only` tiếp tục xác nhận hash runtime, protocol, annotation/exclusion, ba checkpoint AR và bốn archive ghép cặp đã công bố.

Các test numpy/scipy cho ngưỡng, bộ lọc, inventory dữ liệu, báo cáo và bootstrap cũng chạy trong môi trường người dùng chỉ định `D:\App\Anconda\envs\python-tools`. Môi trường đó chưa có torch/torchvision/supervision; các test detector/tracker và phép đo GPU dùng `.venv` riêng.

| Khối | Kiểm tra có ý nghĩa |
|---|---|
| Bộ lọc | Identity đúng từng pixel;13 profile cũ trùng pixel legacy; uint8/geometry hữu hạn; lõi object được giữ khi hệ số lõi bằng0; map/profile sai bị từ chối |
| QP và mục tiêu | Chỉ30/35/40/45; đúng−15% không đạt; thiếu bất kỳ task/codec không đạt; không ngoại suy; đường không đơn điệu/không có overlap không hợp lệ |
| OD nguồn | Guard hoạt động với `targets` là thuộc tính ném lỗi nếu bị đọc; không truyền GT vào detector; đường supervised cũ vẫn hoạt động |
| AR nguồn | Phạt đổi top-1 so với dự đoán nguồn, không cần trường nhãn GT; phép đo DEV A1/A2 dùng đúng cùng16 video |
| Chọn ứng viên | Byte đo thật; loại NaN, rate vượt budget, source damage quá hạn; identity hữu hạn bắt buộc; extra profile là ID cố định ngoài13 neural outputs |
| Tracking | Đổi thứ tự trial không đổi cost; reject không commit; commit chỉ một lần; tiếp tục frame IDs qua cửa sổ; đổi sequence reset; hai arm có quá khứ khác nhau giữ cost/trạng thái khác nhau |
| Dữ liệu | File bị sửa hash phải lỗi; đường vượt root phải lỗi; JPEG không khai báo phải lỗi; OT phải đủ hash các file thực sự dùng; đọc16 frame khai báo cho pixel đúng hệt đọc32 frame stride2 |
| Ghép cặp | Thiếu/trùng/thừa nguồn–codec–QP–arm bị lỗi; byte elementary+header phải bằng tổng packet |
| Tổng hợp | Từ chối DEV, TEST bị limit, endpoint trùng, config/checkpoint/encoder/runtime/manifest/task-model drift; đủ sáu point không đồng nghĩa đủ sáu CI |
| Bootstrap | Ghép cặp theo video qua cả4QP; đường có rate×.8 giữ nghiệm−20% ở draw hữu hạn; draw không hợp lệ phải trả vào đuôi bất lợi, không bị bỏ để xác nhận CI |

## Lỗi thực tế đã tìm và xử lý

- `source_only=True` trong nhánh OD cũ vẫn đi vào GT loss. Chỉ sửa trong namespace mới để không phá hash công bố lịch sử.
- `deepcopy` tracker wrapper lỗi vì chứa module `supervision`; hiện chỉ chia sẻ handle module bất biến và clone state có thể thay đổi.
- Guard OT reset theo cửa sổ, receiver giữ state; nhánh mới dùng common-prefix trials và commit. Arm ablation có state riêng.
- DEV smoke OT16frame ban đầu chấm trên GT256frame; kết quả smoke đó không dùng trong bảng so sánh. Bản sửa chấm đúng prefix quan sát; các DEV32frame và TEST256frame dùng bản sửa.
- FFmpeg hệ thống2026 đã bỏ tùy chọn `-vsync` mà codec legacy gọi. Bản chạy này dùng FFmpeg7.1 từ imageio-ffmpeg0.6.0 và ghi hash binary, thay vì sửa runtime legacy.
- Chỉ kiểm tra hash các JPEG đã khai báo vẫn có thể bỏ lọt JPEG thêm vào làm lệch center/stride. Kiểm tra exact inventory và đọc indices tường minh ngăn lỗi này.
- CI chỉ lấy quantile trên draw hợp lệ có thể xác nhận sai ngưỡng. Ví dụ10 video/2000draw cho55 draw không hợp lệ: bản cũ cho cận trên≈−20%; bản sửa trả cận trên không hữu hạn và không xác nhận.

Rà soát mã độc lập không còn vấn đề Critical/Important sau các sửa đổi này. Kiểm thử, review và phản biện AutoResearchClaw không được dùng làm bằng chứng đạt BD-rate. Kết luận hiệu năng phải dựa trên các phép đo TEST đầy đủ.
