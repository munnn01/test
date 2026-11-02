# Đợt cải thiện NPP giữ nguyên pipeline — 02/11/2026

Pipeline khi chạy: ảnh → NPP gốc (Conv1×1 + U-Net, điều kiện lambda/8) → H.264/H.265 thật → task net đóng băng. SAM 2 đóng băng tạo cache offline; proxy phục vụ backward. 42 file lõi gốc giữ nguyên.

Các thay đổi chỉ nằm ở cách huấn luyện: Gaussian sigma=0,8 làm mục tiêu warm-up cho NPP; hiệu chỉnh decoder proxy trên hỗn hợp ảnh gốc/NPP/Gaussian; so sánh hai nhánh cùng khởi tạo, một nhánh giữ hệ số task, một nhánh cân bằng gradient theo QP. Boundary weight=0 trong đợt này. Không thêm module Gaussian vào suy luận.

| Task | Codec | Tài khoản | Trạng thái đã xác minh |
|---|---|---|---|
| Instance segmentation | H.264 | tranthihongdieu | RUNNING; warm-up đã qua, đang fit proxy |
| Instance segmentation | H.265 | dngbolm | RUNNING; đang khởi tạo |
| Object detection | H.264 | hoangminhhuy123 | RUNNING; đang khởi tạo |
| Object detection | H.265 | ngynanhthuw | RUNNING; đang khởi tạo |

Tất cả notebook/đầu vào/quyền riêng tư đã được kiểm tra trên server. Cùng source SHA ff796936fa09639368b1dc1dc6a678c401faf035884e8f78a55a2d2586618dcf. Toàn bộ 112 file nguồn GitHub khớp byte với capsule Kaggle.

IS–H.264: warm-up 1.000 bước, teacher PSNR 38,3708 dB và source PSNR 31,4688 dB trên 8 ảnh TRAIN-dev; qua ngưỡng 35/20 dB. Đây là phép đo fidelity, chưa phải kết quả BD-rate.

QP 30/35/40/45; 1.536 cập nhật task mỗi nhánh sau warm-up; chọn checkpoint trên 32 ảnh TRAIN riêng, đánh giá cuối trên 256 ảnh VAL mới loại toàn bộ 64 ảnh VAL cũ. So sánh BD-rate trên miền AP chung. COCO được mã hóa thành các luồng intra một khung hình độc lập. Tối đa10.800 giây/worker; dừng theo budget/gate được ghi là chưa hoàn tất.

Kiểm tra:230 kiểm thử pytest đã qua; review độc lập phát hiện 2 lỗi Important đã sửa bằng 8 kiểm thử hồi quy RED→GREEN. Smoke codec thật đủ 8 tổ hợp codec/QP, đối soát byte và gradient tới NPP; proxy/task trong smoke cục bộ là toy do Windows thiếu CompressAI. Proxy CompressAI thật đã khôi phục thành công trên canary Kaggle; đánh giá task/BD-rate cuối còn đang chờ.

Mã và outputs: D:/STUDY/LAB/SAM. Nguồn đã công bố tại commit 35da98c759177f807f3b2b55d835ca0ada9f99ac của munnn01/test. Registry:SAM_TEACHER_RUNS_20261102.json; các receipt trong outputs/sam_teacher_20261102. Tracking để đợt riêng.

Quyết định tích hợp: thư mục SAM là bản sao nguồn không có .git; dùng manifest SHA, kiểm tra 42 file gốc và cập nhật nhánh GitHub với expected-head thay worktree. Nếu đối chiếu không đúng sẽ có rủi ro lệch phiên bản;112 file nguồn đã được xác minh khớp.
