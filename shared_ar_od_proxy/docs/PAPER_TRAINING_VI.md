# Phân tích nguyên nhân và thí nghiệm cải thiện — 23/10/2026

## Kết luận từ kết quả đã kiểm chứng

Tăng NPP lên 9,90M và 12,04M tham số chưa tạo ra cải thiện. Hai nhánh đều chọn checkpoint bước 0; tensor NPP và mọi bitstream DEV trùng identity. Vì thế BD-rate 0% là kết quả của việc quay về identity, không chứng minh mô hình lớn hoàn toàn không có khả năng học. `last.pt` đã thay đổi, gradient khác 0 và không có OOM.

| NPP | Cập nhật giữ lại /512 | Cập nhật được nhận rồi bị rollback | Panel đạt /121 | Checkpoint chọn |
|---|---:|---:|---:|---:|
| 36.531 tham số | 65 | 281 | 14 | bước 312 |
| 9.895.548 tham số | 40 | 275 | 7 | identity |
| 12.039.798 tham số | 26 | 303 | 4 | identity |

Nhánh nhỏ có AR BD-rate +11,21% với H.264 và +1,80% với H.265; OD bằng 0%. Dấu dương là xấu hơn. Đây không phải mức tăng dung lượng ở cùng QP: bitrate cùng QP gần như không đổi, nhưng accuracy giảm tại một số điểm làm đường rate–accuracy xấu đi. Audit đã kiểm chứng 12.288 bitstream, checkpoint và phép tính BD-rate; không phải lỗi đọc nhầm checkpoint.

## Nguyên nhân: mức độ bằng chứng khác nhau

1. **Quy trình rollback đang giữ mô hình quanh identity — bằng chứng trực tiếp.** Chỉ 40/512 cập nhật của NPP9,90M được giữ lại. Panel thất bại tập trung vào hai video TRAIN `ZnIOBBpSryE` và `F2uebcZluts`; không phải vượt ngưỡng RMS hay hết VRAM. Điều kiện mọi mẫu đều có task loss tăng không quá 2%, cộng cấm mất dự đoán AR đúng, là ràng buộc mạnh. Hủy cả khối cập nhật còn xóa tiến bộ ở các mẫu khác. Tuy nhiên, các vi phạm accuracy là thật; bỏ rollback không tự động bảo đảm cải thiện.
2. **Gradient task qua proxy chưa đủ tin cậy — bằng chứng lịch sử, cần đo lại trên NPP lớn.** Audit trước đó có 33 phép thử: bước theo gradient task tại RMS0,5/255 chỉ làm loss codec thật giảm ở 19/33 trường hợp dù proxy dự đoán giảm cả 33. Gradient rate thường hữu ích hơn. Vì vậy scalar rate fit tốt hoặc reconstruction L1 thấp không đủ kết luận Jacobian đúng. Audit mới ghi riêng theo codec/QP, trước và sau huấn luyện, ở hai cỡ bước.
3. **Tăng số tham số chưa giải quyết chênh lệch giữa thuật toán huấn luyện và bài nền.** Mô hình lớn chỉ được thử 512 bước với phần lớn cập nhật bị hủy. Đây là pilot, không tương đương ngân sách hàng trăm nghìn bước và lượng dữ liệu của bài ảnh. Hiện chưa có bằng chứng thiếu capacity là nút thắt chính.
4. **AR/OD có mục tiêu và nhãn khác nhau.** Video có AR ground truth nhưng OD dùng teacher trên nguồn; COCO có OD ground truth, không có AR. Loss detector là surrogate, không phải mAP. Chuẩn hóa loss theo identity giúp cân thang đo nhưng không loại bỏ xung đột gradient giữa hai task. Chưa thể coi kết quả này là chứng minh trên video có đủ nhãn chung AR+OD.

## Bám hai bài nền như thế nào

**Preprocessing Enhanced Image Compression for Machine Vision:** giữ NPP trước codec chuẩn, forward bằng codec thật và backward qua proxy, task network đóng băng; tối ưu rate + task + distortion. Lịch fixed-QP rồi adaptive-QP là cơ sở cho nhánh curriculum. Bài dùng NPP khoảng9,42M, proxy học được dựa trên learned image compression, BPG và QP28/31/34/37/41; bản của chúng ta dùng NPP9,90M, video codec và QP30/35/40/45 theo yêu cầu. Objective hiện tại giữ thêm chuẩn hóa loss, soft penalty, semantic/temporal regularization của pilot; không gọi đây là tái lập nguyên xi loss hoặc proxy của bài. [Mã chính thức của tác giả](https://github.com/XingtongGe/PreprocessingICM).

**A Preprocessing Framework for Video Machine Vision under Compression:** giữ xử lý cả không gian và thời gian, codec ảo khả vi, rate–distortion–task và analyzer đóng băng. Tuy nhiên bài huấn luyện qua virtual codec có prediction/transform/quantization/entropy, đánh giá AR và object tracking; không phải một mô hình chung AR+OD. Proxy ConvGRU + calibrated rate head của chúng ta chưa tái tạo đầy đủ cấu trúc virtual codec đó. Mạng causal, một bitstream chung cho AR+OD và cầu nối forward thật là thiết kế kết hợp hai bài. [Bài gốc](https://arxiv.org/abs/2512.15331).

## Thay đổi đã triển khai trong mã

Hai nhánh dùng cùng NPP9.895.548 tham số, initialization, calibrated proxy, task heads, dữ liệu, loss, optimizer và ngân sách512 bước:

| Nhánh | Lịch QP | Mục đích |
|---|---|---|
| continuous | 30/35/40/45 từ đầu | Đánh giá huấn luyện liên tục sau khi bỏ quality rollback/replay |
| curriculum | QP35 trong128 bước đầu, sau đó cả bốn QP | Tách ảnh hưởng lịch fixed→adaptive theo bài ảnh |

Hai nhánh mới chỉ khác lịch QP. Mốc guarded9,90M cũ là tham chiếu lịch sử: ngoài rollback, replay và cách chọn checkpoint cũng thay đổi nên không quy toàn bộ chênh lệch với mốc cũ cho riêng rollback.

- Giữ32 nguồn TRAIN/task để tối ưu, thêm8 nguồn TRAIN/task kế tiếp để chọn checkpoint. Tập chọn không nằm trong dữ liệu đã fit proxy. DEV128 nguồn/task chỉ chạy sau khi chọn xong; TEST không dùng.
- Cập nhật Adam liên tục, gradient clipping và kiểm tra finite. Không hủy bước/khối vì loss một mẫu tăng. LR giảm10 lần ở bước385 cho cả hai nhánh.
- Mỗi128 bước, đo codec thật trên toàn bộ128 case của tập chọn. Mỗi nhóm codec/QP/task-family phải có mean relative task loss tăng không quá2%; số AR dự đoán đúng không giảm ở từng codec/QP; RMS đầu ra từng mẫu không quá6/255. Trong các checkpoint đạt, chọn mean rate ratio thấp nhất; giữ identity làm fallback. Kiểm tra checkpoint không thay đổi quỹ đạo huấn luyện.
- Lưu cả `last.pt`, `selected.pt`, optimizer và lý do checkpoint không đạt. Chọn identity không được báo là đã cải thiện.
- Audit trên NPP đầu và NPP cuối: sai số rate, L1/PSNR proxy so với decode thật, PSNR decode thật so với nguồn, gradient rate/task và tác động thật ở RMS0,5/255 cùng2/255. Audit dùng một nguồn/task qua đủ8 điểm; là chẩn đoán nhỏ, chưa đại diện toàn bộ tập.
- Đánh giá cuối vẫn dùng bitstream thật, AR top1 và COCO AP; không thay bằng proxy metric. BD-rate không hợp lệ do đường cong không đơn điệu hoặc thiếu khoảng giao sẽ được báo rõ.

## Cách đọc kết quả kế tiếp

Nếu `last` thay đổi mạnh nhưng tất cả checkpoint không đạt, xem task direction audit trước khi tăng bước hoặc capacity. Nếu gradient rate đúng mà gradient task thường sai, ưu tiên proxy có prediction/transform/quantization theo bài video hoặc hiệu chỉnh Jacobian trên đầu ra NPP. Nếu continuous có lợi còn curriculum không hơn, không giữ curriculum chỉ vì giống bài. Nếu cả hai có lợi trên TRAIN selection nhưng DEV không tốt, ưu tiên mở rộng dữ liệu và kiểm tra overfit trước khi huấn luyện dài hơn.

Mục tiêu giảm BD-rate15% chưa được chứng minh. Pilot này kiểm tra hướng cải thiện có cơ sở; chỉ kết quả DEV qua codec thật mới xác nhận mức cải thiện.
