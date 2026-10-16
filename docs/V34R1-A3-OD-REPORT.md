# V34r1 A3 OD: kết quả DEV và hướng tiếp tục

**Chưa đạt mục tiêu.** Hai proxy mới đều không qua validation;512 ảnh TEST dự phòng chưa được suy luận/đánh giá. Không điều chỉnh hay bỏ điểm QP để tạo BD-rate hợp lệ.

Kaggle chạy commit `33ef4181abb9afb6b829de6c6094a2e015b6346a`:128 ảnh mới,64FIT/64validation,21 cấu hình lọc ×H.264/H.265 ×QP30/35/40/45 =**21.504 hàng**. Có84 kiểm thử qua. Thời gian cloud36.42phút;21.058 lần mã hóa thực (kể cả2 kiểm tra lặp), các hàng trùng pixel được giữ trong lưới nhưng dùng lại phép tính. Hai kiểm tra lặp cho packet/pixel giống hệt và sai khác pseudo-loss0; hash trọng số detector không đổi.

| Chính sách | FIT H.264 | FIT H.265 | Validation H.264 | Validation H.265 |
|---|---:|---:|---:|---:|
| A2 đã khóa | -31.20% | không xác định | không xác định | -4.78% |
| Cả21 cấu hình, cùng guard | -39.12% | không xác định | không xác định | -3.70% |
| Pseudo-AP | 0% / identity | không xác định | không xác định | 0% / identity |
| Pseudo task-loss, ngưỡng.025 | -21.41% | không xác định | không xác định | không xác định |

Pseudo-AP không có ngưỡng FIT nào đạt ràng buộc AP nên khóa identity. Pseudo task-loss chọn.025 từ FIT, sau đó không qua validation. Đây là kết quả âm đã được giữ nguyên.

Anchor H.264 validation có AP QP30=27.8521%,QP35=28.5908%: chất lượng tăng khi giảm bitrate. Anchor H.265 FIT có AP QP35=25.1689%,QP40=25.6135%, cũng đảo chiều. Theo giao thức đã khóa, những đường này không có BD-rate hợp lệ. Lọc nhiều cấu hình làm giảm byte cùngQP nhưng chưa chứng minh giảm bitrate tại cùng AP: H.265 validation của bộ21 chỉ−3.70%. Không có cấu hình lọc cố định nào trong21 cấu hình đạt−15% ở H.265 validation; điều này chưa chứng minh mọi cách trộn thích ứng đều thất bại.

Đã đối chiếu đủ lưới128×8×21, byte=elementary+header, hash nguồn và không trùng ảnh giữa các split, phát lại quyết định source-only, chọn ngưỡng FIT và tính lại PCHIP từ AP đã đo. Phần kiểm tra trên máy chỉ đọc/hash/tính số đã có; không chạy lại detector, codec hoặc COCO benchmark. Toàn bộ COCO AP trong bảng do Kaggle đo trên cả tập ảnh, không phải trung bình AP từng ảnh. Chưa có khoảng tin cậy cho các điểm BD này.

Hướng kế tiếp: kiểm tra thêm dung lượng cải thiện của bank bằng chẩn đoán GT trên DEV và mức lệch giữa proxy với GT, trước khi chọn chính sách học mới. Bất kỳ chẩn đoán dùng GT nào cũng không phải bộ chọn có thể triển khai ở phía gửi. OT đang có thí nghiệm riêng về guard HOTA theo lịch sử đã quan sát, với split theo chuỗi và TEST giữ riêng. AR A2 giữ nguyên kết quả và giới hạn đã công bố.

Nguồn đo: [Kaggle OD A3](https://www.kaggle.com/code/dieulinhh/v34r1-a3-od-dev128-proxy21), [giao thức đã khóa](https://github.com/munnn01/test/blob/33ef4181abb9afb6b829de6c6094a2e015b6346a/configs/v34r1-a3-od-probe.json). Hash gói cloud:`683f16675e19b07ede7bce06444b3df203c31c261a4b495e51d908f61beddea0`. Hash policy khóa từ FIT:`c7d279f6022e1cbdd85c05dc3f4727c3af1762219072da48ce8ca502b5a62465`. Dữ liệu công bố gồm hàng đo/prediction/protocol/log/hash; không gồm ảnh gốc, trọng số mô hình hoặc thông tin đăng nhập.
