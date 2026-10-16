# V34r1: OD trên DEV mới và OT A5

**Chưa đạt mục tiêu BD-rate < −15% cho cả ba tác vụ và hai codec.** Chỉ dùng QP30/35/40/45. Mọi mô hình, codec và đánh giá tác vụ đều chạy trên Kaggle; kiểm tra trên máy chỉ đọc tệp, đối chiếu byte/hash và tính lại số đã đo.

AR giữ điểm TEST A2: H.264 −31,35%, H.265 −21,22%. Hai lần chạy lại cho đường cong giống hệt; chúng dùng cùng128 video nên không tăng cỡ mẫu. Khoảng tin cậy bootstrap chưa xác nhận ngưỡng, đặc biệt ở H.265.

OD A3 dùng128 ảnh DEV mới,21 bộ lọc và21.504 hàng. Hai proxy mới không qua validation;512 ảnh TEST chưa đánh giá. Chẩn đoán GT sau đó cho thấy tương quan mô tả của các proxy với tổn hại AP từng ảnh chỉ khoảng0,16–0,30. Bộ chọn có nhãn thật đạt −28,58% ở H.265 validation với slack.02, nhưng đây là chẩn đoán không thể triển khai, không phải kết quả phương pháp và cũng không phải cận trên AP toàn tập. H.264 validation có anchor không đơn điệu, nên BD-rate vẫn không xác định.

OT A3 chạy4 video MOT17 DEV tách theo video,64 khung/video, đúng FPS gốc. Có88 kiểm thử qua,772 hàng ứng viên thực,128 điều kiện video/cửa sổ/codec/QP; cloud chạy37,15 phút. Kiểm tra đủ lưới, byte trả gồm header, hash nguồn, phép tổng hợp HOTA đã lưu và BD-rate đều khớp. Không chạy lại tracker hay GT trên máy.

| OT trên validation A3 | H.264 | H.265 |
|---|---:|---:|
| A2 ở FPS gốc | −2,55% | −6,66% |
| Guard HOTA toàn lịch sử, chính sách FIT đã khóa | 0% / identity | 0% / identity |
| Guard HOTA32 khung gần nhất, chính sách FIT đã khóa | 0% / identity | 0% / identity |

Cả12 ứng viên FIT bị loại bởi ràng buộc giảm HOTA tại cùngQP. Một số ứng viên vẫn giảm byte và cải thiện HOTA trung bình; vì vậy A4 kiểm tra một mục tiêu chọn theo sự đánh đổi bitrate/chất lượng. A3 giữ nguyên kết quả âm. A4 đã khóa duy nhất guard32 khung với ngưỡng.02 từ FIT, trước khi chạy lại calibration. Mục tiêu chọn gần đúng không phải BD-rate; chỉ phép PCHIP hợp lệ từ đủ4 điểm thô mới được dùng kết luận. Validation cũ được gọi rõ là calibration tái sử dụng, không phải kiểm chứng độc lập mới. Ba video TEST dự phòng chưa được đánh giá.

Với OD, A4 đã huấn luyện dự đoán rủi ro từ DEV cũ và chạy xác nhận trên64 ảnh mới; kết quả chi tiết ở cuối báo cáo. Nhãn huấn luyện mới đo ảnh hưởng của một ảnh tới AP cả tập, để tính đến thứ tự độ tin cậy giữa các ảnh. Các kiểm tra tổng hợp đã xác nhận khớp COCO AP chính thức, xử lý ảnh không có đối tượng và không sửa dữ liệu prediction đầu vào.64 ảnh DEV mới đã được dành riêng trước huấn luyện và được đánh giá một lần sau khóa mô hình.


## A4 OT đã đo

Kaggle CPU chạy21 kiểm thử và phát lại cấu hình đã khóa, không mã hóa hay suy luận detector mới. Byte của48 tệp đầu ra tracker khớp lần chạy gốc ở hai anchor và ứng viên FIT; phép tính HOTA/BD-rate đã lưu cũng khớp. Thời gian cloud146,07 giây.

| OT, calibration tái sử dụng | H.264 | H.265 |
|---|---:|---:|
| A2 ở FPS gốc | −2,55% | −6,66% |
| A4 guard32 khung, ngưỡng.02 | −8,23% | −9,94% |

A4 cải thiện so với A2 trong phép đo này nhưng **chưa đạt −15%**. Cả hai đường FIT của A4 không đơn điệu nên BD-rate FIT không xác định. Calibration đã được tái sử dụng cho nghiên cứu, chưa là bằng chứng độc lập. TEST chưa được đánh giá.


## A4 OD đã huấn luyện, chờ DEV mới

Kaggle đã tạo21.504 nhãn marginal AP, với32 phép thay thế kiểm tra lại bằng COCOeval đầy đủ đều sai lệch0. Nhãn FIT và calibration được tính riêng. Hai mô hình ExtraTrees chỉ học từ64 ảnh FIT;302 đặc trưng phía gửi không chứa GT hay ID. Lần huấn luyện thành công chạy22 kiểm thử, hoàn tất trong171,42 giây. Lần đầu lỗi chuẩn bị thư mục trước khi huấn luyện đã được lưu riêng.

Chính sách khóa là ExtraTrees depth10,128 cây, ngưỡng0,005. Hai bản xuất JSON khớp dự đoán sklearn trên toàn21.504 hàng với sai lệch tối đa5,6×10⁻¹⁷. Kiểm toán tệp và7.168 quyết định đã lưu đều khớp; không chạy lại mô hình trên laptop.

Trên calibration tái sử dụng, BD-rate H.264 **không xác định** vì anchor không đơn điệu; H.265 **−11,93%**, chưa đạt−15%. Mục tiêu gần đúng dùng để chọn mô hình là−0,26388, **không phải BD-rate**. Đã khóa mô hình/ngưỡng/hash và chuẩn bị kiểm chứng một lần trên64 ảnh DEV mới với đúng QP30/35/40/45; chưa suy luận hoặc đánh giá TEST512.


## A4 OD: đã kiểm chứng trên64 ảnh DEV mới

Mô hình depth10/ngưỡng0,005 đã khóa trước khi suy luận64 ảnh, đủ10.752 hàng21 bộ lọc ×2 codec ×4 QP. Hai lần lặp anchor khớp packet/pixel; kiểm tra trạng thái detector ở cuối worker không đổi. Kiểm toán đối chiếu hash, đủ lưới,2.048 quyết định đã lưu, byte/header, tái sử dụng và các phép tính BD/resource đều qua.

| Phương pháp, cùng64 ảnh mới | H.264 | H.265 |
|---|---:|---:|
| A2 shortlist + guard cũ | −7,53% | Không xác định: không đơn điệu |
| Toàn21 bộ lọc + guard cũ | −5,37% | −9,82% |
| Critic A4 đã khóa | **−8,21%** | **−11,89%** |

**Chưa đạt−15%.** Cả hai đường critic tính được BD, nhưng điều kiện mở TEST cũng thất bại: utility H.264−0,11292 và H.265−0,06140, đều chưa thấp hơn log(.85)=−0,16252. Utility này không phải BD-rate. Không chỉnh mô hình/ngưỡng trên64 ảnh để gọi lại đó là xác nhận độc lập;512 ảnh TEST chưa đánh giá. Chưa có khoảng tin cậy BD cho phép đo64 ảnh này.

Kaggle chạy774,53 giây trên2 T4. Có10.538 lần encode và10.538 decode thật, gồm2 lượt lặp;10.499 forward detector trên bản giải mã,64 trên nguồn và512 forward controller A2. Các hàng trùng packet/pixel được ghi riêng;10.752 hàng logic không phải10.752 mẫu độc lập. Chi phí CPU phía gửi: tạo đặc trưng35,73 giây, critic0,47 giây, chọn/lưu0,33 giây; không cộng các timer lồng nhau. Tổng byte truyền luôn gồm40 byte header/gói.

OT A5 đã qua23 kiểm thử tổng hợp và review độc lập, đang chạy Kaggle CPU. Nó kiểm tra bộ chọn theo log-bitrate cộng tổn hại HOTA nguồn, chỉ chọn lambda trên FIT rồi khóa trước một lần calibration cũ. Job này không mở TEST. AutoResearchClaw đã có phản biện A4; lượt phản biện A5 chưa thành công do lỗi khởi động ACP, không tính là phản biện hoàn thành.
