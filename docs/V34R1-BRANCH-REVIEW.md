# Đánh giá các nhánh nguồn

Đã clone đầy đủ lịch sử và tạo local tracking branch cho cả29 nhánh nguồn. [Inventory commit chính xác](V34R1-BRANCHES.md). Phần này là đánh giá tĩnh từ commit, cấu hình, giao thức và bằng chứng đã lưu; không gọi đó là đã chạy lại29 mô hình.

| Họ nhánh | Điểm đáng giữ | Giới hạn đối với mục tiêu mới |
|---|---|---|
| v14 dual/example/moderate/motion safety | Các ràng buộc task-regret và an toàn theo ví dụ, so với cân bằng loss đơn thuần | Tên nhánh không xác định đủ trạng thái mới nhất; `v14-search` đã chứa thiết kế V19. Các screen không chứng minh AR/OD/OT×2codec dưới−15% |
| v15 feasible-hinge/rate-pressure/rate-report | Tách sức ép rate, feasibility và kiểm tra protocol/checkpoint | Thay loss hoặc hệ số không chứng minh tiết kiệm byte thật hay duy trì độ chính xác độc lập |
| v16 pixel-release/semantic-anchor | Khảo sát ràng buộc reconstruction và đặc trưng ngữ nghĩa | Cần kiểm tra codec mismatch và suy giảm task sau giải mã; không dùng surrogate loss làm BD-rate |
| v17 log-rate/common-rate/calibration | Đối chiếu proxy rate, scale gradient và byte quan sát | Proxy tốt hơn không bảo đảm thứ hạng ứng viên hoặc lợi ích trên TEST |
| v18 rate weights/common-rate | Screen cố định các mức sức ép rate | Hệ số loss khác nhau phải được chọn trên DEV, không chọn bằng kết quả TEST mới |
| v19 residual/common-filter/configs | Lọc Gaussian residual và căn chỉnh với tài liệu VCM | Cần pixel mới→codec thật mới; không được lấy byte/guard cũ cho ảnh đã đổi |
| v25 policy | Đo all-action, học safety và log-byte ranking; identity là fallback; phân biệt oracle với phương pháp triển khai | TRAIN resubstitution và recall@K không phải chất lượng tác vụ độc lập; giao thức cũ có5QP |
| v26 utility/v27 portfolio | Utility có điều kiện và danh mục ứng viên bổ sung, giúp nghiên cứu khoảng cách shortlist với action bank | So sánh cần cùng ngân sách thử K; không gán lợi ích của tìm nhiều ứng viên cho mạng học tốt hơn |
| v28 motion renderer | Motion bù dịch chuyển camera, cut/reset, Gaussian/block/DC experts, lõi semantic giữ pixel | Một số bằng chứng component là fixture nhiễu/màu tổng hợp, không phải accuracy/mAP/HOTA hoặc BD-rate |
| v29 semantic | Expert/strength distillation; tách variant và calibration khỏi fitting | DEV đã quan sát nhiều lần mang tính exploratory; guard nguồn có thể chấp nhận tổn hại mAP thật |
| v30 adaptive | Conditional admission, profile bank mở rộng, temporal/chroma expert, lineage đo codec | Mục tiêu cũ−10%,5QP và nhiều audit/GT-oracle không chuyển thành xác nhận mục tiêu mới−15%,4QP |
| v31 actions | Action registry rõ, giữ OD empty hợp lệ, kiểm tra tiếp tục stage | Sửa tính đúng đắn và tiến trình vẫn cần được xác nhận bằng kết quả task độc lập |
| archive/original-main | Mốc lịch sử giữ để truy nguyên | Không dùng thay cho snapshot v34 đã có checkpoint/protocol/rows được hash |
| v34-r1/main | Controller CNN+GRU nhỏ, bộ lọc theo vùng, codec chuẩn, kết quả AR1024 và OD64 có thể kiểm tra byte/hash | AR/OD cũ chưa đạt mục tiêu mới; OT thiếu xác nhận độc lập. Giao thức5QP phải được tách khỏi thử nghiệm4QP mới |

## Quyết định cho v34r1

Chọn `source/main` tại `307b77a05fc1084970ac2c3b1b1d74efb83fc806` làm nền vì có checkpoint và bằng chứng công bố truy nguyên được. Không trộn các nhánh thử nghiệm cũ thành một mô hình mới rồi giữ nguyên tên kết quả.

Các ý tưởng được tiếp thu trong phạm vi có thể đo lại: mở rộng bank có giới hạn, guard trên ảnh đã decode và byte thật, phân biệt ứng viên học được với ứng viên cố định, giữ identity, tách state của từng tracking arm, và kiểm tra đủ lưới/provenance. Mã gốc và số liệu lịch sử được giữ để tái lập; namespace mới có giao thức4QP và ngưỡng−15% riêng.

Kết quả của các nhánh có tập dữ liệu, codec build, số QP, ngân sách thử và evaluator khác nhau không tạo thành bảng xếp hạng trực tiếp. Chỉ phép đo anchor/phương pháp mới trên cùng tập TEST, cùng giao thức đóng băng mới được dùng để đánh giá mục tiêu lần này.
